from typing import Dict, List, Optional, Tuple

import click
import torch
from torch.utils.data import DataLoader

import utils
from datasets import BengaliDataSetTrain
from trainers import BengaliTrainer

LOGGER = utils.get_logger(__name__)
from trainers import Trainer


class BengaliEngine:
    def __init__(self, name: str, trainer: Trainer, params: Dict, **kwds):
        super().__init__(**kwds)
        self.name = name
        self.trainer = trainer
        self.dataset = BengaliDataSetTrain
        self.params = params

    def _get_loader(self, train_path: str, folds: List[int]) -> DataLoader:
        self.dataset = self.dataset(train_path=train_path,
                                    folds=folds,
                                    image_height=self.params["image_height"],
                                    image_width=self.params["image_width"],
                                    mean=self.params["mean"],
                                    std=self.params["std"])
        return DataLoader(dataset=self.dataset,
                          batch_size=self.params["batch_size"],
                          shuffle=True,
                          num_workers=4)

    def run_engine(self) -> None:
        """Trains a ResNet34 model for the BengaliAI bengali grapheme competiton.

        Arguments:
            training_data {str} -- path to train-folds.csv file
            epochs {int} -- number of epochs you want to train the classifier
            params {Dict} -- parameter dictionary
        """
        train = self._get_loader(train_path=self.dataset.train_path,
                                 folds=self.params["train_folds"])
        val = self._get_loader(train_path=self.dataset.train_path,
                               folds=self.params["val_folds"])
        model_path = f'trained_models/{self.trainer}_bengali.p'
        for epoch in range(self.params["epochs"]):
            LOGGER.info(f'EPOCH: {epoch}')
            train_loss, train_score = self.trainer.train(train)
            val_loss, val_score = self.trainer.evaluate(val)
            LOGGER.info(
                f'Train loss: {train_loss}, Train score: {train_score}')
            LOGGER.info(
                f'Validation loss: {val_loss}, Validation score: {val_score}')
            self.trainer.scheduler.step(val_loss)
            self.trainer.early_stopping(val_score, self.trainer.model)
            if self.trainer.early_stopping.early_stop:
                LOGGER.info(f"Early stopping at epoch: {epoch}")
                self.trainer.save_model_locally(key=model_path)
                break

        self.trainer.save_model_locally(key=model_path)