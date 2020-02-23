import os
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing

import dispatcher
import models
import utils
from dataset import DataSet

LOGGER = utils.get_logger(__name__)


class BaseTrainer:
    """Base class for handling training/inference"""
    def __init__(
        self,
        model_name: str,
        params: Dict,
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
    def __init__(self, params: Dict):
        super().__init__(params=params)
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