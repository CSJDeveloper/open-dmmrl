"""
Interface of the base dataset.
"""

from typing import List, Tuple
from dataclasses import dataclass
from transformers.utils import ModelOutput as FieldFrozenContainer


@dataclass
class TextSample(FieldFrozenContainer):
    """
    The sample of the unimodal dataset.
    """

    # The question to be answered
    question: str = None
    # Answer in the chain of thought format
    cot_answer: str = None
    # groundtruth solution
    groundtruth: str = None

    # Additional field to hold the information
    # of this sample
    data_info: dict = None


@dataclass
class VisualTextSample(TextSample):
    """
    The sample of the unimodal dataset.
    """

    # Images involved in the question
    # Each item is a tuple holding the image's token
    # name in the question and the path
    # For example, if the question is:
    #   <image 1> For company B, the revenue is $6,000,000.
    # Then the item to be added is ("image 1", "path/to/image")
    question_images: List[Tuple[str, str]] = None

    # Images involved in the answer
    # Each item is a list holding (image token name, images' path), which
    # is same as the one used by the `question_images`.
    # for the corresponding reasoning step, i.e.,
    # one thought in the cot
    cot_images: List[List[Tuple[str, str]]] = None


class VisualTextBase:
    """A base class for the visual-text dataset."""

    def save_pil_image(self, image_data, path: str, filename: str):
        """A function to save the PIL image to a file."""
        save_path = None
        if image_data is not None:
            img_format = image_data.format if image_data.format is not None else "PNG"
            extension = img_format.lower()
            if img_format.upper() == "JPEG":
                extension = "jpg"

            filename = f"{filename}.{extension}"
            save_path = f"{path}/{filename}"
            image_data.save(save_path, img_format)

            return save_path

        return save_path
