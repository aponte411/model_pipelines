import os
from abc import abstractmethod
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import dispatcher
import models
import utils
from metrics import macro_recall
from utils import EarlyStopping

LOGGER = utils.get_logger(__name__)


class Trainer:
    """Base class for training/inference

    Arguements:
        model {Any} -- object from models module.
    """
    def __init__(self, model: Any, **kwds):
        super().__init__(**kwds)
        self.model = model
        self.model = None
        self.model_path = None

    @abstractmethod
    def train(self):
        pass

    def get_model_path(self):
        return self.model_path

    def get_model(self):
        return self.model

    def load_model_locally(self):
        pass

    def load_model_from_s3(self):
        pass

    def predict(self):
        pass

    def predict_and_score(self):
        pass

    def save_model_locally(self):
        pass

    def save_model_to_s3(self):
        pass


class QuoraTrainer(Trainer):
    def __init__(self, model: Any, **kwds):
        super().__init__(model, **kwds)
        self.model = model

    def load_model_locally(self, key: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = dispatcher.MODELS['randomforest']
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str):
        self.model = dispatcher.MODELS['randomforest']
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
              X_val: pd.DataFrame, y_val: pd.DataFrame):
        LOGGER.info("Building model from scratch")
        self.model.fit(X_train, y_train)

    def predict(self, X_new: pd.DataFrame) -> pd.Series:
        LOGGER.info(f'Making predictions..')
        return self.model.predict(X=X_new)

    def predict_and_score(self, X_new: pd.DataFrame,
                          y_new: pd.DataFrame) -> None:
        LOGGER.info(f'Making predictions and scoring the model...')
        preds = self.model.predict(X=X_new)
        LOGGER.info(f'ROC-AUC-SCORE: {self.model.score(y=y_new, y_hat=preds)}')
        return preds

    def save_model_locally(self, key: str):
        LOGGER.info(f"Saving model for {self.tournament} locally")
        self.model.save(key)

    def save_to_s3(self, filename: str, key: str):
        LOGGER.info(f"Saving {self.name} for {self.tournament} to s3 bucket")
        self.model.save_to_s3(filename=filename, key=key)


class BengaliTrainer(Trainer):
    def __init__(self, model: Any, **kwds):
        super().__init__(model, **kwds)
        self.model = model
        self.model.cuda()
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.3, verbose=True)

    def _loss_fn(self, preds, targets):
        pred1, pred2, pred3 = preds
        target1, target2, target3 = targets
        loss1 = self.criterion(pred1, target1)
        loss2 = self.criterion(pred2, target2)
        loss3 = self.criterion(pred3, target3)
        return (loss1 + loss2 + loss3) / 3

    def _load_to_gpu_float(self, data):
        return data.to(self.device, dtype=torch.float)

    def _load_to_gpu_long(self, data):
        return data.to(self.device, dtype=torch.long)

    def _get_image(self, data):
        return self._load_to_gpu_float(data["image"])

    def _get_targets(self, data) -> List:
        grapheme_root = self._load_to_gpu_long(data["grapheme_root"])
        vowel_diacritic = self._load_to_gpu_long(data["vowel_diacritic"])
        consonant_diacritic = self._load_to_gpu_long(
            data["consonant_diacritic"])
        return [grapheme_root, vowel_diacritic, consonant_diacritic]

    @staticmethod
    def score(preds, targets) -> float:
        final_preds = torch.cat(preds)
        final_targets = torch.cat(targets)
        return macro_recall(final_preds, final_targets)

    @staticmethod
    def concat_tensors(tensor) -> torch.tensor:
        one, two, three = tensor
        return torch.cat((one, two, three), dim=1)

    @staticmethod
    def stack_tensors(tensor) -> torch.tensor:
        one, two, three = tensor
        return torch.stack((one, two, three), dim=1)

    def train(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        final_loss = 0
        counter = 0
        final_preds, final_targets = [], []
        for batch, data in tqdm(enumerate(data_loader)):
            counter += 1
            image = self._get_image(data=data)
            targets = self._get_targets(data=data)
            self.optimizer.zero_grad()
            predictions = self.model(image)
            loss = self._loss_fn(preds=predictions, targets=targets)
            loss.backward()
            self.optimizer.step()
            final_loss += loss
            final_preds.append(self.concat_tensors(tensor=predictions))
            final_targets.append(self.stack_tensors(tensor=targets))

        macro_recall_score = self.score(preds=final_preds,
                                        targets=final_targets)
        LOGGER.info(f'loss: {final_loss/counter}')
        LOGGER.info(f'macro-recall: {macro_recall_score}')

        return final_loss / counter, macro_recall_score

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        with torch.no_grad():
            self.model.eval()
            final_loss = 0
            counter = 0
            final_preds, final_targets = [], []
            for batch, data in tqdm(enumerate(data_loader)):
                counter += 1
                image = self._get_image(data=data)
                targets = self._get_targets(data=data)
                predictions = self.model(image)
                final_loss += self._loss_fn(preds=predictions, targets=targets)
                final_preds.append(self.concat_tensors(tensor=predictions))
                final_targets.append(self.stack_tensors(tensor=targets))

            macro_recall_score = self.score(preds=final_preds,
                                            targets=final_targets)
        LOGGER.info(f'loss: {final_loss/counter}')
        LOGGER.info(f'macro-recall: {macro_recall_score}')

        return final_loss / counter, macro_recall_score

    def inference(self, data_loader):
        pass
