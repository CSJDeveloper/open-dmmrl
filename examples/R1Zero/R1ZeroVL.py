"""
Main function to run the training of the DeepSeek-R1-Zero based on
Group Relative Policy Optimization. --> Visual Language dataset.
"""

from projinit import config
from projinit.platform_init import InitializePlatforms, ProjectInfo

import torch
from trl import get_peft_config
from trl.trainer import GRPOConfig, ModelConfig
from transformers import AutoModelForImageTextToText, AutoProcessor

from dmmrl.trainer import vl_grpo
from dmmrl.data_process import map_vl_sample
from dmmrl.dataset import registry as data_registry
from dmmrl.rewarder import registry as reward_registry


def main():
    """Main Running."""

    #########################################################
    ## Define the project and set the platforms
    InitializePlatforms().login_accounts()
    # Create the project information
    proj_info = ProjectInfo()
    wandb_run = proj_info.create_wandb(entity="LatentPlanReasoner")

    torch.cuda.empty_cache()
    #########################################################
    ## Define and load the model
    model_name = proj_info.model_config["model_name"]

    model = AutoModelForImageTextToText.from_pretrained(model_name)
    vl_processor = AutoProcessor.from_pretrained(
        model_name,
        revision=proj_info.model_config["model_revision"],
        trust_remote_code=proj_info.model_config["trust_remote_code"],
        min_pixels=proj_info.model_config["min_pixels"],
        max_pixels=proj_info.model_config["max_pixels"],
    )
    vl_processor.pad_token_id = vl_processor.tokenizer.pad_token_id
    vl_processor.eos_token_id = vl_processor.tokenizer.eos_token_id
    lora_config = (
        {} if "lora" not in proj_info.train_config else proj_info.train_config["lora"]
    )
    model_args = ModelConfig(
        model_name_or_path=model_name,
        model_revision=proj_info.model_config["model_revision"],
        torch_dtype=proj_info.model_config["torch_dtype"],
        attn_implementation=proj_info.model_config["attn_implementation"],
        use_peft=proj_info.train_config["use_peft"],
        **lora_config
    )

    #########################################################
    ## Load and format the data
    data_config = proj_info.data_config
    train_dataset = data_registry.get(data_config["data_name"], split="train")
    test_dataset = data_registry.get(data_config["data_name"], split="test")

    def process_func(x):
        return map_vl_sample(
            x,
            system_prompt=(
                None
                if "system_prompt" in proj_info.model_config
                else proj_info.model_config["system_prompt"]
            ),
            to_format="prompt_completion",
            add_answer=False,
            maintain_columns=["groundtruth"],
        )

    train_dataset.lm_format_function = process_func
    test_dataset.lm_format_function = process_func

    #########################################################
    ## Define the training arguments
    reward_functions = reward_registry.get(
        proj_info.train_config["GRPO"]["reward_functions"]
    )
    reward_process_classes = reward_registry.get_process_class(reward_functions)

    batch_size = proj_info.train_config["per_device_train_batch_size"]
    gradient_size = proj_info.train_config["gradient_accumulation_steps"]
    eval_batch_size = proj_info.eval_config["per_device_eval_batch_size"]

    grpo_config = GRPOConfig(
        reward_weights=proj_info.train_config["GRPO"]["reward_weights"],
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_size,
        warmup_ratio=proj_info.train_config["warmup_ratio"],
        bf16=proj_info.train_config["bf16"],
        num_train_epochs=proj_info.train_config["epoch"],
        learning_rate=proj_info.train_config["learning_rate"],
        logging_steps=proj_info.log_config["logging_steps"],
        max_steps=proj_info.train_config["max_steps"],
        logging_first_step=proj_info.log_config["logging_first_step"],
        logging_strategy=proj_info.log_config["logging_strategy"],
        save_steps=proj_info.log_config["save_steps"],
        weight_decay=proj_info.train_config["weight_decay"],
        lr_scheduler_type=proj_info.train_config["lr_scheduler"],
        seed=proj_info.env_config["seed"],
        do_eval=proj_info.eval_config["do_eval"],
        per_device_eval_batch_size=eval_batch_size,
        output_dir=proj_info.log_config["checkpoint_path"],
        gradient_checkpointing_kwargs=proj_info.train_config[
            "gradient_checkpointing_kwargs"
        ],
        report_to="wandb",
        disable_tqdm=True,
        # GRPO block:
        use_vllm=proj_info.train_config["GRPO"]["use_vllm"],
        vllm_device=proj_info.train_config["GRPO"]["vllm_device"],
        vllm_gpu_memory_utilization=proj_info.train_config["GRPO"][
            "vllm_gpu_memory_utilization"
        ],
        temperature=proj_info.train_config["GRPO"]["temperature"],
        beta=proj_info.train_config["GRPO"]["beta"],
        max_prompt_length=proj_info.train_config["GRPO"]["max_prompt_length"],
        max_completion_length=proj_info.train_config["GRPO"]["max_completion_length"],
        num_generations=proj_info.train_config["GRPO"]["num_generations"],
        log_completions=proj_info.log_config["log_completions"],
        ds3_gather_for_generation=proj_info.train_config["GRPO"][
            "ds3_gather_for_generation"
        ],
    )

    trainer = vl_grpo.VLGRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=vl_processor,
        reward_functions=reward_functions,
        reward_processing_classes=reward_process_classes,
        additional_args=None,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        peft_config=get_peft_config(model_args),
    )
    trainer_stats = trainer.train()

    # Finish wandb run
    # The detailed run history is generated when we finish the Weights & Biases run.
    wandb_run.finish()
    config.Config.set_records(status="Completed")


if __name__ == "__main__":
    main()
