import os
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics, preprocessing
from tqdm import tqdm

import dispatcher
import models
import utils
from dataset import DataSet
from metrics import macro_recall

LOGGER = utils.get_logger(__name__)


class BaseTrainer:
    """Base class for handling training/inference"""
    def __init__(
        self,
        model_name: str,
        params: Dict = None,
    ):
        self.model_name = model_name
        self.params = params
        self.model = None

    def __repr__(self):
        return self.model_name

    def __str__(self):
        return self.model_name

    def get_model(self):
        return self.model

    @abstractmethod
    def load_model_locally(self):
        pass

    @abstractmethod
    def load_model_from_s3(self):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def predict_and_score(self):
        pass

    @abstractmethod
    def save_model_locally(self):
        pass

    @abstractmethod
    def save_model_to_s3(self):
        pass


class QuoraTrainer(BaseTrainer):
    def __init__(self, model_name: str, params: Dict):
        super().__init__(model_name=model_name, params=params)
        self.model_name = model_name
        self.params = params
        self.model = None

    def load_model_locally(self, key: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = models.XGBoostModel()
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str):
        self.model = models.XGBoostModel()
        self.model.load_from_s3(filename=filename, key=key)
        self.model = self.model.load(key)
        LOGGER.info(
            f"Trained model loaded from s3 bucket: {os.environ['BUCKET']}")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                    X_val: pd.DataFrame, y_val: pd.DataFrame):
        LOGGER.info("Building XGBoostModel from scratch")
        if self.params["tree_method"] == 'gpu_hist':
            LOGGER.info(f"Training XGBoost with GPU's")
        self.model = models.XGBoostModel(
            max_depth=self.params["max_depth"],
            learning_rate=self.params["learning_rate"],
            l2=self.params["l2"],
            n_estimators=self.params["n_estimators"],
            tree_method=self.params["tree_method"])
        LOGGER.info(f"Training XGBoost model..")
        eval_set = [(X_val, y_val)]
        self.model.fit(X=X_train, y=y_train, eval_set=eval_set)

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


class BengaliTrainer(BaseTrainer):
    def __init__(self, model_name: str, params: Dict = None):
        super().__init__(model_name, params)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.criterion = nn.CrossEntropyLoss()
        self.model = models.ResNet34()

    def loss_fn(self, outputs, targets) -> float:
        output1, output2, output3 = outputs
        target1, target2, target3 = targets
        loss1 = self.criterion(output1, target1)
        loss2 = self.criterion(output2, target2)
        loss3 = self.criterion(output3, target3)
        return (loss1 + loss2 + loss3) / 3

    def train(self, dataset, data_loader) -> Tuple[float, float]:
        def _load_to_gpu_float(data):
            return data.to(self.device, dtype=torch.float)

        def _load_to_gpu_long(data):
            return data.to(self.device, dtype=torch.long)

        self.model.train()
        final_loss = 0
        counter = 0
        final_outputs, final_targets = [], []
        for batch, data in tqdm(enumerate(data_loader)):
            counter += 1
            image = _load_to_gpu_float(data["image"])
            grapheme_root = _load_to_gpu_long(data["grapheme_root"])
            vowel_diacritic = _load_to_gpu_long(data["vowel_diacritic"])
            consonant_diacritic = _load_to_gpu_long(
                data["consonant_diacritic"])
            self.optimizer.zero_grad()
            outputs = self.model(image)
            targets = [grapheme_root, vowel_diacritic, consonant_diacritic]
            loss = self.loss_fn(outputs=outputs, targets=targets)
            loss.backward()
            self.optimizer.step()

            final_loss += loss

            output1, output2, output3 = outputs
            target1, target2, target3 = targets
            final_outputs.append(torch.cat((output1, output2, output3), dim=1))
            final_targets.append(
                torch.stack((target1, target2, target3), dim=1))

        final_outputs = torch.cat(final_outputs)
        final_targets = torch.cat(final_targets)
        macro_recall = macro_recall(final_outputs, final_targets)

        LOGGER.info(f'loss: {final_loss/counter')
        LOGGER.info(f'macro_recall: {macro_recall}')

        return final_loss / counter, macro_recall
