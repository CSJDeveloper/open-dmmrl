"""
Interface of the GSM8K dataset.
"""

from datasets import load_dataset

from dmmrl.dataset.base import TextSample
from dmmrl.tools import re_utility


class GSM8KDataset:
    """A consistent interface for the GSM8k dataset."""

    def __init__(self):
        super().__init__()
        self.hf_dataset = load_dataset("openai/gsm8k", "main")
        ori_columns = self.hf_dataset["test"][0].keys()
        self.hf_dataset = self.hf_dataset.map(
            self.to_format,
            remove_columns=ori_columns,
            keep_in_memory=True,
            load_from_cache_file=False,
        )

    def to_format(self, sample):
        """Get the sample from the given idx."""
        # Create the sample
        groundtruth_sol = re_utility.extract_content(sample["answer"], marker="####")
        groundtruth_sol = "" if groundtruth_sol is None else groundtruth_sol
        return TextSample(
            question=sample["question"],
            cot_answer=sample["answer"],
            groundtruth=groundtruth_sol,
            data_info={"dataset": "gsm8k"},
        )
