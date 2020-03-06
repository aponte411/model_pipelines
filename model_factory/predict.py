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
    ENGINE_PARAMS = {
        "train_path": "inputs/train-folds.csv",
        "test_path": "inputs",
        "pickle_path": "inputs/pickled_images",
        "model_dir": "trained_models",
        "train_folds": [0],
        "val_folds": [4],
        "train_batch_size": 64,
        "test_batch_size": 32,
        "epochs": 3,
        "test_loops": 5,
        "image_height": 137,
        "image_width": 236,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.239, 0.225)
    }
    model = models.ResNet34(pretrained=True)
    trainer = trainers.BengaliTrainer(model=model, model_name='restnet34')
    bengali = engines.engaliEngine(trainer=trainer, params=ENGINE_PARAMS)
    submission = bengali.run_inference_engine()
    LOGGER.info(submission)
    # WIP


if __name__ == "__main__":
    main()
