from typing import Optional

import click

from engines import BengaliEngine
from trainers import BengaliTrainer
from models import ResNet34

import utils

LOGGER = utils.get_logger(__name__)


@click.command()
@click.option('-d', '--data', type=str, default='bengali')
def runner(data: str) -> Optional:
    if data == 'bengali':
        TRAINING_PARAMS = {
            0: {
                "train": [0, 1, 2, 3],
                "val": [4]
            },
            1: {
                "train": [0, 1, 2, 4],
                "val": [3]
            },
            2: {
                "train": [0, 1, 3, 4],
                "val": [2]
            },
            3: {
                "train": [0, 2, 3, 4],
                "val": [1]
            }
        }
        for loop, fold_dict in TRAINING_PARAMS.items():
            PARAMS = {
                "train_path": "inputs/train-folds.csv",
                "test_path": "inputs",
                "pickle_path": "inputs/pickled_images",
                "image_height": 137,
                "image_width": 236,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.239, 0.225),
                "train_folds": fold_dict['train'],
                "val_folds": fold_dict['val'],
                "test_loops": 5,
                "train_batch_size": 64,
                "test_batch_size": 32,
                "epochs": 3
            }
            model = ResNet34(pretrained=True)
            trainer = BengaliTrainer(model=model, model_name='restnet34')
            bengali = BengaliEngine(name='bengali-engine',
                                    trainer=trainer,
                                    params=PARAMS)
            bengali.run_training_engine(model_dir='trained_models')


if __name__ == "__main__":
    runner()
