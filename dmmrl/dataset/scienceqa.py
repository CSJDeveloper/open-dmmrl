"""
Interface of the ScienceQA dataset.
"""

import os
from datasets import load_dataset

from projinit.config import Config

from dmmrl.dataset.base import VisualTextSample, VisualTextBase
from dmmrl.identifier import SOLKEY


class ScienceQADataset(VisualTextBase):
    """A consistent interface for the ScienceQA dataset."""

    def __init__(self, split="train"):
        super().__init__(split=split)
        self.hf_dataset = load_dataset("derek-thomas/ScienceQA", split=split)

        self.data_path = Config().data.data_path
        self.image_path = f"{self.data_path}/images"
        os.makedirs(self.image_path, exist_ok=True)

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

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        question = sample["question"]
        question = f"{question} (Place final selected option within {SOLKEY})."
        options = sample["choices"]
        image_data = sample["image"]
        q_image = None

        filename = f"{self.split}-Image-ID{self.idx}"
        filepath = f"{self.image_path}/{filename}.jpg"
        if os.path.exists(filepath):
            q_image = filepath
        else:
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                q_image = save_path

        if options is None or len(options) == 0:
            question = f"{question}\n"
        else:
            option_letters = [chr(ord("A") + num) for num in range(len(options))]
            options_str = [
                f"({letter}): {choice}"
                for choice, letter in zip(options, option_letters)
            ]
            options_str = "\n".join(options_str)

            question = f"{question}\nOptions:\n{options_str}"

        groundtruth = chr(ord("A") + int(sample["answer"]))
        lecture = sample["lecture"]
        solution = sample["solution"]
        cot_answer = f"{lecture}\n{solution}"

        return VisualTextSample(
            main_id=f"{self.split}-ID{self.idx}",
            question=question,
            cot_answer=cot_answer,
            groundtruth=groundtruth,
            question_images=[("image", q_image)],
            data_info={
                "dataset": "derek-thomas/ScienceQA",
                "grade": sample["grade"],
                "subject": sample["subject"],
                "topic": sample["topic"],
                "category": sample["category"],
                "skill": sample["skill"],
                "lecture": sample["lecture"],
                "hint": sample["hint"],
            },
        )
