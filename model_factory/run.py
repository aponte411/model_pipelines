import datetime
from typing import Optional

import click

import utils
from engines import BengaliEngine
from models import ResNet34
from trainers import BengaliTrainer

LOGGER = utils.get_logger(__name__)


@click.command()
@click.option('-d', '--data', type=str, default='bengali')
def main(data: str) -> Optional:
    if data == 'bengali':
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
        timestamp = datetime.datetime.today().strftime("%B %d, %Y %H:%M")
        LOGGER.info(f'Training started {timestamp}')
        for loop, fold_dict in TRAINING_PARAMS.items():
            LOGGER.info(f'Training loop: {loop}')
            ENGINE_PARAMS = {
                "train_path": "inputs/train-folds.csv",
                "test_path": "inputs",
                "pickle_path": "inputs/pickled_images",
                "model_dir": "trained_models",
                "train_folds": fold_dict['train'],
                "val_folds": fold_dict['val'],
                "train_batch_size": 64,
                "test_batch_size": 32,
                "epochs": 3,
                "test_loops": 5,
                "image_height": 137,
                "image_width": 236,
                "mean": (0.485, 0.456, 0.406),
                "std": (0.229, 0.239, 0.225)
            }
            model = ResNet34(pretrained=True)
            trainer = BengaliTrainer(model=model, model_name='restnet34')
            bengali = BengaliEngine(trainer=trainer, params=ENGINE_PARAMS)
            bengali.run_training_engine()
        LOGGER.info(f'Training complete!')


if __name__ == "__main__":
    main()
