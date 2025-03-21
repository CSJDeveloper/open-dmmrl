"""
Interface of the GSM8K dataset.
"""

from datasets import load_dataset


from dmmrl.tools import re_utility
from dmmrl.identifier import SOLKEY
from dmmrl.dataset.base import TextSample, VisualTextBase


class GSM8KDataset(VisualTextBase):
    """A consistent interface for the GSM8k dataset."""

    def __init__(self, split="train"):
        super().__init__(split=split)
        self.hf_dataset = load_dataset("openai/gsm8k", "main", split=split)

        # Use the visit index as the sample ID
        self.idx = 0

        # Make the sample to be the desired format defined
        # in the dataset.base class
        self.hf_dataset = self.hf_dataset.map(
            self.to_format,
            batch_size=1,
            load_from_cache_file=True,
            remove_columns=self.hf_dataset.column_names,
        )

    def to_format(self, sample):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        groundtruth_sol = re_utility.extract_content(sample["answer"], marker="####")
        groundtruth_sol = "" if groundtruth_sol is None else groundtruth_sol
        problem = sample["question"]
        question = f"{problem} (Place final solution within {SOLKEY})."
        return TextSample(
            main_id=f"{self.split}-ID{self.idx}",
            question=question,
            cot_answer=sample["answer"],
            groundtruth=groundtruth_sol,
            data_info={"dataset": "gsm8k"},
        )
