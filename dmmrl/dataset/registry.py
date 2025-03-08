"""
An interface to registry the datasets.
"""

import logging
from dmmrl.dataset import gsm8k, math, mmmu, scienceqa


data_factory = {
    "gsm8k": gsm8k.GSM8KDataset,
    "math": math.MATHDataset,
    "mmmu": mmmu.MMMUDataset,
    "scienceqa": scienceqa.ScienceQADataset,
}


def get(name: str):
    """Get the dataset."""

    logging.info("--> Get the dataset: %s", name)

    return data_factory[name.lower()]()
