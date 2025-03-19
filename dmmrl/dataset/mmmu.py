"""
Interface of the MMMU dataset.
"""

import os
import ast
from datasets import load_dataset

from projinit.config import Config

from dmmrl.identifier import SOLKEY
from dmmrl.dataset.base import VisualTextSample, VisualTextBase


class MMMUDataset(VisualTextBase):
    """A consistent interface for the MMMU dataset."""

    def __init__(self, split="train"):
        super().__init__(split=split)
        self.hf_dataset = load_dataset("lmms-lab/MMMU", split=split)

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
        sample_id = sample["id"]
        # Create the sample
        question = sample["question"]
        question = f"{question} (Place final selected option within {SOLKEY})."
        options = sample["options"]

        question_images = []
        for i in range(1, 8):
            image_name = f"image-{i}"
            q_image_token = f"image-{i}"
            filename = f"Image-ID{sample_id}-{image_name}"
            filepath = f"{self.image_path}/{filename}.png"
            if os.path.exists(filepath):
                question_images.append((q_image_token, filepath))
                continue

            image_data = sample[image_name]
            save_path = self.save_pil_image(image_data, self.image_path, filename)
            if save_path is not None:
                question_images.append((q_image_token, save_path))

        if options is None or len(options) == 0:
            question = f"{question}\n"
        else:
            options = ast.literal_eval(options)
            option_letters = [chr(ord("A") + num) for num in range(len(options))]
            options_str = [
                f"({letter}): {choice}"
                for choice, letter in zip(options, option_letters)
            ]
            options_str = "\n".join(options_str)
            question = f"{question}\nOptions:\n{options_str}"

        cot_answer = sample["explanation"]

        return VisualTextSample(
            main_id=sample_id,
            question=question,
            cot_answer=cot_answer,
            groundtruth=sample["answer"],
            question_images=question_images,
            data_info={
                "dataset": "lmms-lab/MMMU",
                "question_type": sample["question_type"],
                "subfield": sample["subfield"],
                "topic_difficulty": sample["topic_difficulty"],
                "img_type": sample["img_type"],
            },
        )
