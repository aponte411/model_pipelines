import datetime
import os
from typing import Optional

import click

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

TRAINING_PARAMS = {
    1: {
        "train": [0, 1, 2, 3],
        "val": [4]
    },
    2: {
        "train": [0, 1, 2, 4],
        "val": [3]
    },
    3: {
        "train": [0, 1, 3, 4],
        "val": [2]
    },
    4: {
        "train": [0, 2, 3, 4],
        "val": [1]
    }
}


@click.command()
@click.option('-m', '--model-name', type=str, default='resnet34')
@click.option('-tr', '--train', type=bool, default=True)
@click.option('-inf', '--inference', type=bool, default=False)
@click.option('-train',
              '--train-path',
              type=str,
              default='inputs/train-folds.csv')
@click.option('-test', '--test-path', type=str, default='inputs')
@click.option('-pickle',
              '--pickle-path',
              type=str,
              default='inputs/pickled_images')
@click.option('-sub', '--submission-dir', type=str, default='inputs')
@click.option('-model', '--model-dir', type=str, default='trained_models')
@click.option('-trainb', '--train-batch-size', type=int, default=64)
@click.option('-testb', '--test-batch-size', type=int, default=32)
@click.option('-ep', '--epochs', type=int, default=5)
def run_bengali_engine(model_name: str, train: bool, inference: bool,
                       train_path: str, test_path: str, pickle_path: str,
                       submission_dir: str, model_dir: str,
                       train_batch_size: int, test_batch_size: int,
                       epochs: int) -> Optional:
    if train:
        timestamp = utils.generate_timestamp()
        LOGGER.info(f'Training started {timestamp}')
        for loop, fold_dict in TRAINING_PARAMS.items():
            LOGGER.info(f'Training loop: {loop}')
            ENGINE_PARAMS = {
                "train_path": train_path,
                "test_path": test_path,
                "pickle_path": pickle_path,
                "model_dir": model_dir,
                "submission_dir": submission_dir,
                "train_folds": fold_dict['train'],
                "val_folds": fold_dict['val'],
                "train_batch_size": train_batch_size,
                "test_batch_size": test_batch_size,
                "epochs": epochs,
                "image_height": 137,
                "image_width": 236,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.239, 0.225),
                # 1 loop per test parquet file
                "test_loops": 5,
            }
            model = MODEL_DISPATCHER.get(model_name)
            trainer = trainers.BengaliTrainer(model=model,
                                              model_name=model_name)
            bengali = engines.BengaliEngine(trainer=trainer,
                                            params=ENGINE_PARAMS)
            bengali.run_training_engine(save_to_s3=True, creds=CREDENTIALS)
        LOGGER.info(f'Training complete!')
    if inference:
        timestamp = utils.generate_timestamp()
        LOGGER.info(f'Inference started {timestamp}')
        submission = bengali.run_inference_engine(
            model_dir=ENGINE_PARAMS['model_dir'],
            to_csv=True,
            output_dir=ENGINE_PARAMS['submission_dir'],
            load_from_s3=True,
            creds=CREDENTIALS)
        LOGGER.info(f'Inference complete!')
        LOGGER.info(submission)


if __name__ == "__main__":
    run_bengali_engine()
