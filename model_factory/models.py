from datetime import datetime
from typing import Any, List, Tuple

import joblib
import numpy as np
import pandas as pd
import pretrainedmodels
import tensorflow as tf
import torch.nn as nn
import xgboost as xgb
from catboost import CatBoostRegressor
from keras import Sequential, layers, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import multi_gpu_model
from lightgbm import LGBMRegressor
from sklearn import metrics
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.linear_model import LinearRegression
from torch.nn import functional as F
from xgboost import XGBoostClassifier, XGBRegressor

import utils

LOGGER = utils.get_logger(__name__)


class XGBoostModel:
    def __init__(self,
                 max_depth: int = 7,
                 learning_rate: float = 0.001777765,
                 l2: float = 0.1111119,
                 n_estimators: int = 2019,
                 colsample_bytree: float = 0.019087,
                 tree_method: str = 'auto'):
        self.params = None
        self.model = XGBoostClassifier(max_depth=max_depth,
                                       learning_rate=learning_rate,
                                       reg_lambda=l2,
                                       n_estimators=n_estimators,
                                       n_jobs=-1,
                                       tree_method=tree_method,
                                       colsample_bytree=colsample_bytree,
                                       verbosity=3)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame, eval_set=None) -> None:
        self.model.fit(X=X, y=y, eval_set=eval_set, early_stopping_rounds=50)

    def predict(self, X: pd.DataFrame) -> Any:
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict(X)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        return pd.DataFrame(yhat)

    def score(self, y_new: pd.DataFrame, preds: pd.DataFrame):
        return metrics.roc_auc_score(y_new, preds)

    def save_to_s3(self, filename: str, key: str) -> None:
        s3 = utils.S3Client()
        s3.upload_file(filename=filename, key=key)

    def load_from_s3(self, filename: str, key: str) -> None:
        s3 = utils.S3Client()
        s3.download_file(filename=filename, key=key)

    def save(self, filename) -> None:
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        return joblib.load(filename)


class ResNet34(nn.Module):
    def __init__(self, pretrained: bool):
        super(ResNet34, self).__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__["resnet34"](
                pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)

        self.linear1 = nn.Linear(512, 168)
        self.linear2 = nn.Linear(512, 11)
        self.linear3 = nn.Linear(512, 7)

    def forward(self, x: torch.tensor) -> Tuple:
        bs, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(bs, -1)
        linear1 = self.linear1(x)
        linear2 = self.linear2(x)
        linear3 = self.linear3(x)
        return linear1, linear2, linear3
