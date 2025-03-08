"""
The code to process the data to be the desired format.


For the support of the multiple images:
https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#multi-image-inference

https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl

Currently, for one content, we can only support one image.

The support for the multiple images will be added in the future.
"""

from dmmrl.dataset import base


def map_samples(
    samples: base.TextSample,
    proj: dict,
    to_format: str = "messages",
    add_answer: bool = True,
):
    """
    Create the messages from the samples.

    The samples will be replaced by messages named `name`.
    The original content of the samples will be automatically maintained.
    """

    assert to_format in ["messages", "prompt_completion"]

    qs = samples["question"]
    ans = samples["cot_answer"]

    qs = [qs] if isinstance(qs, str) else qs
    ans = [ans] if isinstance(ans, str) else ans

    messages = []
    prompts = []
    completions = []
    for q, a in zip(qs, ans):
        msg = [{"role": "user", "content": q}]
        if add_answer:
            msg.append({"role": "assistant", "content": a})

        if "system_prompt" in proj.model_config:
            msg.insert(
                0, {"role": "system", "content": proj.model_config["system_prompt"]}
            )
        messages.append(msg)
        # Only contain the system and user content for the prompt
        prompts.append(msg[:2])
        completions.append(msg[-1])

    base_output = {}
    if to_format == "messages":
        base_output = {"messages": messages}

    if to_format == "prompt_completion":
        if add_answer:
            base_output = {"prompt": prompts, "completion": completions}
        else:
            base_output = {"prompt": prompts}
    return base_output


def map_vl_samples(
    sample: base.VisualTextSample,
    proj: dict,
    to_format: str = "message",
    add_answer: bool = True,
    maintain_columns: list = None,
):
    """
    Create the message from the visual-language samples.

    The samples will be replaced by message named `name`.
    The original content of the samples will be automatically maintained.


    Note that for the visual-language samples, we do not support the batch of
    samples as the .map process will automatically align contents from user, system, and assistant.
    """
    assert to_format in ["message", "prompt_completion"]

    q = sample["question"]
    q_imgs = sample["question_images"]
    an = sample["cot_answer"]

    message = []
    # Add the system prompt if it exists
    if "system_prompt" in proj.model_config:
        message.append(
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": proj.model_config["system_prompt"]}
                ],
            }
        )

    # Add the user prompt
    user_content = [{"type": "text", "text": q}]
    main_image = q_imgs[0][-1]
    if main_image is not None:
        user_content.insert(0, {"type": "image", "image": main_image})

    message.append(
        {
            "role": "user",
            "content": user_content,
        }
    )
    # Add the assistant content, i.e., answer
    if add_answer:
        message.append({"role": "assistant", "content": [{"type": "text", "text": an}]})

    # Only contain the system and user content for the prompt
    prompt = message[:2]
    completion = message[-1] if add_answer else None

    base_output = {}
    if to_format == "message":
        base_output = {"groundtruth": sample["groundtruth"], "message": message}

    elif to_format == "prompt_completion":
        if add_answer:
            base_output = {"prompt": prompt, "completion": completion}
        else:
            base_output = {"prompt": prompt}
    else:
        raise ValueError("The format is not supported.")

    if main_image is not None:
        base_output["image"] = main_image

    if maintain_columns is not None:
        for col in maintain_columns:
            base_output[col] = sample[col]

    return base_output
