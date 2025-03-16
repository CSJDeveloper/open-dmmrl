"""
Interface of the MATH dataset.
"""

from datasets import load_dataset
from math_verify import LatexExtractionConfig, parse


from dmmrl.dataset.base import TextSample, VisualTextBase
from dmmrl.identifier import SOLKEY


class MATHDataset(VisualTextBase):
    """A consistent interface for the MATH dataset."""

    def __init__(self, split="train"):
        super().__init__(split=split)
        self.hf_dataset = load_dataset(
            "DigitalLearningGmbH/MATH-lighteval", split=split
        )

    def to_format(self, sample):
        """Get the sample from the given idx."""
        # Create the sample
        cot_answer = sample["solution"]
        # opt = re_utility.extract_format_equations(cot_answer, target_format="\\boxed")
        # groundtruth_sol = "" if opt is None else opt[-1]
        # The parsed item will be a list holding a value and a str value
        groundtruth_sol = parse(
            cot_answer,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        groundtruth_sol = "" if len(groundtruth_sol) == 0 else groundtruth_sol[-1]
        problem = sample["problem"]
        question = f"{problem} (Place final solution within {SOLKEY})."

        return TextSample(
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth_sol,
            data_info={
                "dataset": "MATH-lighteval",
                "level": sample["problem"],
                "type": sample["type"],
            },
        )
