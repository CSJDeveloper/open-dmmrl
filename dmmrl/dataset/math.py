"""
Interface of the MATH dataset.
"""

from datasets import load_dataset

from dmmrl.dataset.base import TextSample
from dmmrl.tools import re_utility


class MATHDataset:
    """A consistent interface for the MATH dataset."""

    def __init__(self):
        super().__init__()
        self.hf_dataset = load_dataset("DigitalLearningGmbH/MATH-lighteval")
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
        cot_answer = sample["solution"]
        opt = re_utility.extract_format_equations(cot_answer, target_format="\\boxed")
        groundtruth_sol = "" if opt is None else opt[-1]

        return TextSample(
            question=sample["problem"],
            cot_answer=cot_answer,
            groundtruth=groundtruth_sol,
            data_info={
                "dataset": "MATH-lighteval",
                "level": sample["problem"],
                "type": sample["type"],
            },
        )
