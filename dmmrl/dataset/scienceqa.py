"""
Interface of the ScienceQA dataset.
"""

import os
from datasets import load_dataset

from projinit.config import Config

from dmmrl.dataset.base import VisualTextSample, VisualTextBase


class ScienceQADataset(VisualTextBase):
    """A consistent interface for the ScienceQA dataset."""

    def __init__(self):
        super().__init__()
        self.hf_dataset = load_dataset("derek-thomas/ScienceQA")

        self.data_path = Config().data.data_path
        self.image_path = f"{self.data_path}/images"
        os.makedirs(self.image_path, exist_ok=True)

        ori_columns = self.hf_dataset["test"][0].keys()

        self.idx = 0
        self.hf_dataset = self.hf_dataset.map(
            self.to_format,
            remove_columns=ori_columns,
            keep_in_memory=True,
            load_from_cache_file=False,
        )
        self.valid_splits = ["dev", "test", "validation"]

    def to_format(self, sample: dict):
        """Get the sample from the given idx."""
        self.idx += 1

        # Create the sample
        question = sample["question"]
        options = sample["choices"]
        image_data = sample["image"]
        q_image = None

        filename = f"Image{self.idx}"
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

        cot_answer = sample["solution"]

        return VisualTextSample(
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
