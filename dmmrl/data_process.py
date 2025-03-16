"""
The code to process the data to be the desired format.


For the support of the multiple images:
https://huggingface.co/docs/transformers/main/en/model_doc/llava_next#multi-image-inference

https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl

Currently, for one content, we can only support one image.

The support for the multiple images will be added in the future.
"""

from dmmrl.dataset import base


def map_sample(
    sample: base.TextSample,
    system_prompt: str = None,
    to_format: str = "messages",
    add_answer: bool = True,
    maintain_columns: list = None,
):
    """
    Create the messages from the samples.

    The samples will be replaced by messages named `name`.
    The original content of the samples will be automatically maintained.
    """

    assert to_format in ["messages", "prompt_completion"]

    q = sample["question"]
    an = sample["cot_answer"]

    message = [{"role": "user", "content": q}]
    if add_answer:
        message.append({"role": "assistant", "content": an})

    if system_prompt is not None:
        message.insert(0, {"role": "system", "content": system_prompt})

    # Only contain the system and user content for the prompt
    prompt = message[:2]
    completion = message[-1] if add_answer else None

    base_output = {}
    if to_format == "message":
        base_output = {"message": message}

    elif to_format == "prompt_completion":
        if add_answer:
            base_output = {"prompt": prompt, "completion": completion}
        else:
            base_output = {"prompt": prompt}
    else:
        raise ValueError("The format is not supported.")

    if maintain_columns is not None:
        for col in maintain_columns:
            base_output[col] = sample[col]

    return base_output


def map_vl_sample(
    sample: base.VisualTextSample,
    system_prompt: str = None,
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
    if system_prompt is not None:
        message.append(
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}],
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
        base_output = {"message": message}

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
