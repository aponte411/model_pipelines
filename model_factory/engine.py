from typing import Dict, List, Optional, Tuple

import click
import torch
from torch.utils.data import DataLoader

import utils
from datasets import BengaliDataSetTrain
from trainers import BengaliTrainer

LOGGER = utils.get_logger(__name__)


def get_loader(train_path: str, folds: List[int], params: Dict) -> DataLoader:
    """Converts a datasets object into a DataLoader.

    Arguments:
        train_path {str} -- path to train-fold.csv file
        folds {List[int]} -- the key for the fold_mapping dictionary
        params {Dict} -- parameter dictionary

    Returns:
        DataLoader -- torch.utils.data.DataLoader object
    """
    dataset = BengaliDataSetTrain(train_path=train_path,
                                  folds=folds,
                                  image_height=params["image_height"],
                                  image_width=params["image_width"],
                                  mean=params["mean"],
                                  std=params["std"])
    return DataLoader(dataset=dataset,
                      batch_size=params["batch_size"],
                      shuffle=True,
                      num_workers=4)


def run_bengali_engine(training_data: str, epochs: int, params: Dict) -> None:
    """Trains a ResNet34 model for the BengaliAI bengali grapheme competiton.

    Arguments:
        training_data {str} -- path to train-folds.csv file
        epochs {int} -- number of epochs you want to train the classifier
        params {Dict} -- parameter dictionary
    """
    train = get_loader(train_path=training_data,
                       folds=[0, 1, 2, 3],
                       params=params)
    val = get_loader(train_path=training_data, folds=[4], params=params)
    trainer = BengaliTrainer(model_name='resnet')
    model_path = f'trained_models/{trainer}_bengali.p'
    for epoch in range(epochs):
        LOGGER.info(f'EPOCH: {epoch}')
        train_loss, train_score = trainer.train(train)
        val_loss, val_score = trainer.evaluate(val)
        LOGGER.info(f'Train loss: {train_loss}, Train score: {train_score}')
        LOGGER.info(
            f'Validation loss: {val_loss}, Validation score: {val_score}')
        trainer.scheduler.step(val_loss)
        trainer.early_stopping(val_score, trainer.model)
        if trainer.early_stopping.early_stop:
            LOGGER.info(f"Early stopping at epoch: {epoch}")
            trainer.save_model_locally(key=model_path)
            break
        trainer.save_model_locally(key=model_path)


@click.command()
@click.option('-d', '--data', type=str, default='bengali')
def runner(data: str) -> Optional:
    if data == 'bengali':
        PARAMS = {
            "image_height": 137,
            "image_width": 236,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.239, 0.225)
        }
        run_bengali_engine(
            epochs=10,
            params=PARAMS,
            training_data="inputs/bengali_grapheme/train-folds.csv")


if __name__ == "__main__":
    preds = runner()
