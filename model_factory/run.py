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
        for train_fold in range(4):
            PARAMS = {
                "train_path": "inputs/bengali_grapheme/train-folds.csv",
                "test_path": "inputs/bengali_grapheme",
                "pickle_path": "inputs/bengali_grapheme/pickled_images",
                "epochs": 5,
                "image_height": 137,
                "image_width": 236,
                "batch_size": 64,
                "test_batch_size": 32,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.239, 0.225),
                "train_folds": [train_fold],
                "val_folds": [4],
                "test_loops": 5
            }
            model = ResNet34(pretrained=True)
            trainer = BengaliTrainer(model=model)
            bengali = BengaliEngine(name='bengali-engine',
                                    trainer=trainer,
                                    params=PARAMS)
            bengali.run_training_engine()


if __name__ == "__main__":
    runner()
