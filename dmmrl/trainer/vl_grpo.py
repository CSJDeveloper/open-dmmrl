"""
The implementation of the training with the Group Relative Policy Optimization (GRPO).

For the multimodal of the Qwen-VL models, especially the text, image and video scenarios, we refer to the instruct in https://github.com/QwenLM/Qwen2.5-VL/tree/main.

For the multimodal for the vllm package, we refer to the instruct in
https://docs.vllm.ai/en/latest/serving/multimodal_inputs.html.

"""

from typing import List


import torch
from transformers import Trainer, PreTrainedModel
from trl.models import unwrap_model_for_generation
from trl.trainer import GRPOConfig, GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.trainer.utils import pad, selective_log_softmax
from accelerate.utils import broadcast_object_list, gather, gather_object
from qwen_vl_utils import process_vision_info

import wandb


class VLGRPOTrainer(GRPOTrainer):
    """
    Trainer supported by the Group Relative Policy Optimization under the multimodal scenarios.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        args: GRPOConfig,
        processing_class,
        reward_functions,
        reward_processing_classes=None,
        additional_args=None,  # Additional customized arguments
        train_dataset=None,
        eval_dataset=None,
        callbacks=None,
        optimizers=(None, None),
        peft_config=None,
    ):
        super().__init__(
            model,
            args=args,
            processing_class=processing_class,
            reward_funcs=reward_functions,
            reward_processing_classes=reward_processing_classes,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # We should note that here is one potential issue in the
        # self.ref_model because within the `GRPOTrainer`, when
        # is_deepspeed_zero3_enabled(), the self.ref_model is initialized
        # directly with AutoModelForCausalLM. Thus, self.ref_model may not
        # align with the model.
        # self.refer_model = create_reference_model(model)

        self.additional_args = additional_args

    def _get_per_token_logps(self, model, logp_inputs, logits_to_keep):
        """
        Get the per-token log probabilities for the completions for the model and the reference model.

        :param processed_prompt_completions: This should contain the ids of the prompt and the completions, attention masks that have masked out tokens after the EOS in the completion, and the multimodal info.
        """

        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(
            **logp_inputs,
            logits_to_keep=logits_to_keep + 1,
        ).logits

        # (B, L-1, V), exclude the last logit: it corresponds to the next token prediction
        logits = logits[:, :-1, :]

        input_ids = logp_inputs["input_ids"][:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]

        #  compute logprobs for the input tokens
        return selective_log_softmax(logits, input_ids)

    def _prepare_inputs(self, inputs: List[dict[str]]):
        """
        Prepare the experience for the subsequent optimization of models.

        This part is to:
        1). Perform the generation for each prompt multiple times, leading to
            a group as mentioned in the paper.
        2). Compute the rule-based rewards.
        3). Estimate the advantage for step-wise reasoning process.

        :param inputs: Each item of inputs present one example, which is a dict containing necessary terms, especially the `prompt`.
        """

        device = self.accelerator.device
        # Get the prompts of one batch of examples
        prompts = [ipt["prompt"] for ipt in inputs]

        ####################################################
        ## Start the experience sampling process in which involves inference
        # input samples through the \theta_{old} to get experiences.
        # This aligns with the GRPO paper's Sample 𝐺 outputs for each question.

        # Process the prompts to be the target format
        # the function will capture the keyword, such as the `prompt` in the example and replace its value with the formatted prompt.
        format_prompts = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        # In this part, we need to process either the text or the image-text prompt
        image_inputs, video_inputs = process_vision_info(prompts)
        prompt_inputs = {"text": format_prompts}
        multimodal_inputs = {}
        if image_inputs is not None:
            multimodal_inputs["images"] = image_inputs
        if video_inputs is not None:
            multimodal_inputs["videos"] = video_inputs
        prompt_inputs.update(multimodal_inputs)

        # Get the processed inputs containing
        #  - input_ids, attention_mask, ..
        #  - pixel_values, image_grid_thw
        processed_inputs = self.processing_class(
            **prompt_inputs,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )

        processed_inputs = Trainer._prepare_inputs(self, processed_inputs)
        input_ids, input_mask = (
            processed_inputs["input_ids"],
            processed_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            input_ids = input_ids[:, -self.max_prompt_length :]
            input_mask = input_mask[:, -self.max_prompt_length :]
            processed_inputs["input_ids"] = input_ids
            processed_inputs["attention_mask"] = input_mask

        processed_multimodal_inputs = dict(
            filter(
                lambda item: item[0] not in ["input_ids", "attention_mask"],
                processed_inputs.items(),
            )
        )

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            # Gather format_prompts from all processes and thus get a list
            # of format_prompts
            all_prompts_text = gather_object(format_prompts)
            all_multimodal_inputs = gather_object(multimodal_inputs)
            # Set the prompts to the vllm_inputs
            vllm_inputs = all_prompts_text
            # Convert the vllm_inputs to the multimodal case.
            if all_multimodal_inputs:
                # Different from the processor in the transformer, the vllm's
                # generate receives the key words, such as "image"...
                vllm_inputs = [
                    {
                        "prompt": p,
                        "multi_modal_data": {
                            key[:-1] if key.endswith("s") else key: value
                            for key, value in mm.items()
                        },
                    }
                    for p, mm in zip(all_prompts_text, all_multimodal_inputs)
                ]

            if self.accelerator.is_main_process:
                outputs = self.llm.generate(
                    vllm_inputs,
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                completion_ids = [
                    out.token_ids
                    for completions in outputs
                    for out in completions.outputs
                ]
            else:
                completion_ids = [None] * len(all_prompts_text)

            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            prompt_completion_ids = torch.cat([input_ids, completion_ids], dim=1)
        else:
            # Forward generation without using the vLLM
            with unwrap_model_for_generation(
                self.model, self.accelerator
            ) as unwrapped_model:

                # The output will be the combination of the input prompt and
                # a sequence of completion tokens
                prompt_completion_ids = unwrapped_model.generate(
                    **processed_inputs,
                    generation_config=self.generation_config,
                )

            # Compute prompt length and extract completion ids
            prompt_length = input_ids.size(1)
            input_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # After the forward generation, we have obtained the
        # `prompt_completion_ids` that contains the ids of the prompt and the completion.

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        # (B*G, P+C)
        attention_mask = torch.cat([input_mask, completion_mask], dim=1)
        # Only need to compute the logits for the completion tokens
        logits_to_keep = completion_ids.size(1)

        # To compute the KL divergence, we need to compute the logits from the # reference model or the peft model (only disable the adapters).
        with torch.inference_mode():
            # Organize the inputs for the model logits computations.
            # We have
            #   - processed_inputs holding all processed original data
            #       <-- input_ids, attention mask
            #       <-- or pixel_values, image_grid_thw
            #   - prompt_completion_ids holding prompt and completion ids
            #   - attention_mask that has been processed to mask out tokens
            #       after he EOS in the completion
            # We need to organize:
            logp_inputs = {
                "input_ids": prompt_completion_ids,
                "attention_mask": attention_mask,
            }
            logp_inputs.update(processed_multimodal_inputs)

            # input_ids=input_ids,
            # attention_mask=attention_mask,
            # pixel_values=pixel_values,
            # image_grid_thw=image_grid_thw,

            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    logp_inputs,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        logp_inputs,
                        logits_to_keep,
                    )

        ####################################################
        ## Start the reward computation process in which the reward functions are used to evaluate the
        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # Module instead of PretrainedModel for compat with compiled models
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [
                        {"messages": p + c} for p, c in zip(prompts, completions)
                    ]
                    texts = [
                        apply_chat_template(x, reward_processing_class)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    # Shape (B*G,)
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {
                    key: [example[key] for example in inputs] for key in keys
                }
                output_reward_func = reward_func(
                    completions=completions, **reward_kwargs
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(
            dim=1
        )

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(format_prompts),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": input_ids,
            "prompt_mask": input_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "processed_multimodal_inputs": processed_multimodal_inputs,
        }

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):

        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # we only need to compute the logits for the completion tokens
        logits_to_keep = completion_ids.size(1)

        logp_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        logp_inputs.update(inputs["processed_multimodal_inputs"])
        per_token_logps = self._get_per_token_logps(model, logp_inputs, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        ref_per_token_logps = inputs["ref_per_token_logps"]
        per_token_kl = (
            torch.exp(ref_per_token_logps - per_token_logps)
            - (ref_per_token_logps - per_token_logps)
            - 1
        )

        # x - x.detach() allows for preserving gradients from x
        advantages = inputs["advantages"]
        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()
        ) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = (
            (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()

        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)

        mean_kl = (
            (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
        ).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss
