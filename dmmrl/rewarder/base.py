"""
Base implementation of the rewards
"""

from typing import Union, Callable

from transformers import PreTrainedTokenizerBase, ProcessorMixin, PreTrainedModel

# For the reward functions to be used.
BaseRewardFunction = Union[PreTrainedModel, Callable[[list, list], list[float]]]
RewardFunctions = Union[BaseRewardFunction, list[BaseRewardFunction]]

# For the reward processing class, we support
RewardProcessingClasses = Union[
    # For the text reward
    PreTrainedTokenizerBase,
    list[PreTrainedTokenizerBase],
    # for the text and visual reward]
    ProcessorMixin,
    list[ProcessorMixin],
]
