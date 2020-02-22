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

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
TARGET = os.environ.get("TARGET")
MODEL_PATH = os.environ.get("MODEL_PATH")
DROP = ['is_duplicate', 'kfold']


class BaseTrainer:
    """Base class for handling training/inference"""
    def __init__(
        self,
        model_name: str,
        to_drop: List,
    ):
        self.model_name = model_name
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
    """Trains, serializes, loads, and conducts inference"""
    def __init__(self, model_name='xgboost'):
        super().__init__(model_name=model_name, to_drop=to_drop)

        self.model_name = model_name
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
                    params: Dict):
        LOGGER.info("Building XGBoostModel from scratch")
        if params["tree_method"] == 'gpu_hist':
            LOGGER.info(f"Training XGBoost with GPU's")
        self.model = models.XGBoostModel(max_depth=params["max_depth"],
                                         learning_rate=params["learning_rate"],
                                         l2=params["l2"],
                                         n_estimators=params["n_estimators"],
                                         tree_method=params["tree_method"])
        LOGGER.info(f"Training XGBoost model for {self.tournament}")
        eval_set = [(self.data['validation'].x,
                     self.data['validation'].y[self.tournament])]
        self.model.fit(dfit=self.data['train'],
                       tournament=self.tournament,
                       eval_set=eval_set)

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


def main():
    trainer = QuoraTrainer(tournament='quora_question_pairs',
                           name='base_trainer')
    dataset = DataSet(path="../inputs/quora_question_pairs/train-folds.csv",
                      fold=0)
    train, val = dataset.prepare_data()
    y_train, y_val = dataset.get_targets()
    X_train, X_val = dataset.clean_data()
    trainer.train_model(X=X_train, y=y_train)
    trainer.predict_and_score(X_new=X_val, y_new=y_val)


if __name__ == "__main__":
    main()
