import os
from typing import Any, Dict, List, Tuple

import click
import numpy as np

import datasets
import engines
import models
import trainers
import utils

LOGGER = utils.get_logger(__name__)


@click.command()
@click.option('-m', '--competition', type=str, default='quora')
@click.option('-lm', '--load-model', type=bool, default=True)
@click.option('-sm', '--save-model', type=bool, default=False)
def main(submit: bool) -> pd.DataFrame:

    PARAMS = {
        "train_path": "inputs/bengali_grapheme/train-folds.csv",
        "test_path": "inputs/bengali_grapheme",
        "pickle_path": "inputs/bengali_grapheme/pickled_images",
        "image_height": 137,
        "image_width": 236,
        "train_batch_size": 64,
        "test_batch_size": 64,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.239, 0.225),
        "epochs": 5,
        "train_folds": [0],
        "val_folds": [4],
        "test_loops": 5
    }

    model = models.ResNet34(pretrained=False)
    trainer = trainers.BengaliTrainer(model=model)
    bengali = engines.BengaliEngine(name='bengali-engine',
                                    trainer=trainer,
                                    params=PARAMS)
    submission = bengali.run_inference_engine()
    LOGGER.info(submission.head())
    # WIP


if __name__ == "__main__":
    main()
