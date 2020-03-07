import os
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

import datasets
import engines
import models
import trainers
import utils
from dispatcher import MODEL_DISPATCHER

LOGGER = utils.get_logger(__name__)

CREDENTIALS = {}
CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
CREDENTIALS['bucket'] = os.environ.get("bucket")


@click.command()
@click.option('-m', '--model-name', type=str, default='resnet34')
def main(model_name: str) -> Optional:
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
    model = MODEL_DISPATCHER.get(model_name)
    trainer = trainers.BengaliTrainer(model=model, model_name=model_name)
    bengali = engines.BengaliEngine(trainer=trainer, params=ENGINE_PARAMS)
    submission = bengali.run_inference_engine(
        model_dir=ENGINE_PARAMS['model_dir'],
        to_csv=True,
        output_dir=ENGINE_PARAMS['submission_dir'],
        load_from_s3=True,
        creds=CREDENTIALS)
    LOGGER.info(submission)


if __name__ == "__main__":
    main()
