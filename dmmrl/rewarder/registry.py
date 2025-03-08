"""
An interface to register rewards based on the configuration.
"""

import logging
from transformers import PreTrainedModel, AutoProcessor, AutoTokenizer

from dmmrl.rewarder.rule_rewards import (
    accuracy_reward,
    format_reward,
    reasoning_steps_reward,
    len_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    code_reward,
)

from dmmrl.rewarder.visual_rewards import iou_reward


rule_rewards = {
    "accuracy_reward": accuracy_reward,
    "format_reward": format_reward,
    "reasoning_steps_reward": reasoning_steps_reward,
    "len_reward": len_reward,
    "get_cosine_scaled_reward": get_cosine_scaled_reward,
    "get_repetition_penalty_reward": get_repetition_penalty_reward,
    "code_reward": code_reward,
    "iou_reward": iou_reward,
}


def get(reward_names: list):
    """Registry the reward functions based on the configuration."""
    reward_functions = []
    for reward_name in reward_names:
        reward_functions.append(rule_rewards[reward_name])

    logging.info("--> Defined Reward functions: %s", reward_functions)

    return reward_functions


def get_process_class(reward_functions):
    """Get the reward processing class."""
    reward_processing_classes = []
    for function in reward_functions:
        if isinstance(function, PreTrainedModel):
            if reward_processing_class is None:
                if "vision_config" in function.config:
                    reward_processing_class = AutoProcessor.from_pretrained(
                        function.config._name_or_path
                    )
                    reward_processing_class.eos_token = (
                        reward_processing_class.tokenizer.eos_token_id
                    )
            else:
                reward_processing_class = AutoTokenizer.from_pretrained(
                    function.config._name_or_path
                )
            if reward_processing_class.pad_token_id is None:
                reward_processing_class.pad_token = reward_processing_class.eos_token
            # The reward model computes the reward for the latest non-padded token in the input sequence.  So it's important to set the pad token ID to the padding token ID of the processing class.
            function.config.pad_token_id = reward_processing_class.pad_token_id
            reward_processing_classes.append(reward_processing_class)
        else:
            # Do not need to process anything
            reward_processing_classes.append(None)

    logging.info(
        "--> Defined Reward Processing Classes: %s",
        reward_processing_classes,
    )

    return reward_processing_classes
