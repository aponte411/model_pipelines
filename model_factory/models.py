import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numerox as nx
import numpy as np
import pandas as pd
import pretrainedmodels
import pytorch_lightning as pl
import tensorflow as tf
import torch
import torch.nn as nn
import transformers
from catboost import CatBoostRegressor
from keras import Sequential, layers, metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.utils import multi_gpu_model
from lightgbm import LGBMRegressor
from pytorch_transformers import GPT2LMHeadModel
from sklearn import metrics
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.linear_model import LinearRegression
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from xgboost import XGBRegressor

import utils
from datasets import BengaliDataSetTest, BengaliDataSetTrain
from metrics import macro_recall

LOGGER = utils.get_logger(__name__)


class BaseModel(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def save_to_s3(filename: str, key: str) -> None:
        """Save model to s3 bucket"""
        s3 = utils.S3Client()
        s3.upload_file(filename=filename, key=key)

    @staticmethod
    def load_from_s3(filename: str, key: str) -> None:
        """Download model from s3 bucket"""
        s3 = utils.S3Client()
        s3.download_file(filename=filename, key=key)

    @staticmethod
    def load(self, filename: str):
        """Load trained model"""
        return joblib.load(filename)

    def save(self, filename: str):
        """Serialize model"""
        joblib.dump(self, filename)


class ResNet34(nn.Module, BaseModel):
    def __init__(self, pretrained: bool):
        super().__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__["resnet34"](
                pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet34"](pretrained=None)
        self.linear1 = nn.Linear(512, 168)
        self.linear2 = nn.Linear(512, 11)
        self.linear3 = nn.Linear(512, 7)

    def forward(self, x: torch.tensor) -> Tuple[torch.Tensor]:
        batch_size = x.shape[0]
        features = self.model.features(x)
        features = F.adaptive_avg_pool2d(features, 1).reshape(batch_size, -1)
        linear1 = self.linear1(features)
        linear2 = self.linear2(features)
        linear3 = self.linear3(features)
        return linear1, linear2, linear3


class ResNet50(nn.Module, BaseModel):
    def __init__(self, pretrained: bool):
        super().__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__["resnet50"](
                pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["resnet50"](pretrained=None)
        self.linear1 = nn.Linear(2048, 168)
        self.linear2 = nn.Linear(2048, 11)
        self.linear3 = nn.Linear(2048, 7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        batch_size = x.shape[0]
        features = self.model.features(x)
        features = F.adaptive_avg_pool2d(features, 1).reshape(batch_size, -1)
        linear1 = self.linear1(features)
        linear2 = self.linear2(features)
        linear3 = self.linear3(features)
        return linear1, linear2, linear3


class SeResNext101(nn.Module, BaseModel):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if pretrained:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](
                pretrained="imagenet")
        else:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](
                pretrained=None)
        self.linear1 = nn.Linear(2048, 168)
        self.linear2 = nn.Linear(2048, 11)
        self.linear3 = nn.Linear(2048, 7)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        batch_size = x.shape[0]
        features = self.model.features(x)
        features = F.adaptive_avg_pool2d(features, 1).reshape(batch_size, -1)
        linear1 = self.linear1(features)
        linear2 = self.linear2(features)
        linear3 = self.linear3(features)
        return linear1, linear2, linear3


# WIP
class ResNet34Lightning(pl.LightningModule):
    def __init__(self, params: Dict):
        super().__init__()
        self.model = pretrainedmodels.__dict__["resnet34"](
            pretrained="imagenet")
        self.linear1 = nn.Linear(512, 168)
        self.linear2 = nn.Linear(512, 11)
        self.linear3 = nn.Linear(512, 7)
        self.train_constructor = BengaliDataSetTrain
        self.val_constructor = BengaliDataSetTrain
        self.test_constructor = BengaliDataSetTest

    def forward(self, x: torch.Tensor):
        batch_size, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch_size, -1)
        linear1 = self.linear1(x)
        linear2 = self.linear2(x)
        linear3 = self.linear3(x)
        return linear1, linear2, linear3

    def training_step(self, batch, batch_idx):
        image = self._get_image(data=batch)
        targets = self._get_targets(data=batch)
        predictions = self.forward(image)
        loss = self.cross_entropy_loss(preds=predictions, targets=targets)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        image = self._get_image(data=batch)
        targets = self._get_targets(data=batch)
        predictions = self.forward(image)
        loss = self._loss_fn(preds=predictions, targets=targets)
        return {'val_loss': loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        image = self._get_image(data=batch)
        targets = self._get_targets(data=batch)
        predictions = self.forward(image)
        return {'test_loss': F.cross_entropy(predictions, targets)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        setattr(
            self, 'training_set',
            self.train_constructor(train_path=self.params["train_path"],
                                   pickle_path=self.params["pickle_path"],
                                   folds=self.params["train_folds"],
                                   image_height=self.params["image_height"],
                                   image_width=self.params["image_width"],
                                   mean=self.params["mean"],
                                   std=self.params["std"]))
        return DataLoader(self.training_set,
                          batch_size=self.params["train_batch_size"],
                          shuffle=True,
                          num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        setattr(
            self, 'val_set',
            self.train_constructor(train_path=self.params["train_path"],
                                   pickle_path=self.params["pickle_path"],
                                   folds=self.params["val_folds"],
                                   image_height=self.params["image_height"],
                                   image_width=self.params["image_width"],
                                   mean=self.params["mean"],
                                   std=self.params["std"]))
        return DataLoader(self.val_set,
                          batch_size=self.params["train_batch_size"],
                          shuffle=True,
                          num_workers=4)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.test_path, batch_size=32)

    def cross_entropy_loss(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        output1, output2, output3 = outputs
        target1, target2, target3 = targets
        loss1 = criterion(output1, target1)
        loss2 = criterion(output2, target2)
        loss3 = criterion(output3, target3)
        return (loss1 + loss2 + loss3) / 3

    def _load_to_gpu_float(self, data):
        return data.to(self.device, dtype=torch.float)

    def _load_to_gpu_long(self, data):
        return data.to(self.device, dtype=torch.long)


class GPT2(nn.Module, BaseModel):
    def __init__(self):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')


class BERTBaseUncased(nn.Module):
    def __init__(self, bert_path: str, n_outputs: int):
        super().__init__()
        self.bert_path = bert_path
        self.bert = transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, n_outputs)

    def forward(self, ids: torch.Tensor, mask: torch.Tensor,
                token_type_ids: torch.Tensor):
        hidden_states, pooler_output = self.bert(ids,
                                                 attention_mask=mask,
                                                 token_type_ids=token_type_ids)
        bert_output = self.bert_drop(pooler_output)
        return self.out(bert_output)


class NumeraAIModel(nx.Model):
    def __init__(self,
                 max_depth: int = 7,
                 learning_rate: float = 0.001777765,
                 l2: float = 0.1111119,
                 n_estimators: int = 2019,
                 colsample_bytree: float = 0.019087,
                 tree_method: str = 'auto'):
        self.params = None
        self.model = XGBRegressor(max_depth=max_depth,
                                  learning_rate=learning_rate,
                                  reg_lambda=l2,
                                  n_estimators=n_estimators,
                                  n_jobs=-1,
                                  tree_method=tree_method,
                                  colsample_bytree=colsample_bytree,
                                  verbosity=3)

    def fit(self,
            dfit: nx.data.Data,
            tournament: str,
            eval_set=None,
            eval_metric=None) -> None:
        self.model.fit(X=dfit.x,
                       y=dfit.y[tournament],
                       eval_set=eval_set,
                       eval_metric=eval_metric,
                       early_stopping_rounds=50)

    def predict(self, dpre: nx.data.Data, tournament: str) -> nx.Prediction:
        """
        Alternative to fit_predict() 
        dpre: must be data['tournament']
        tournament: can be int or str.
        """
        prediction = nx.Prediction()
        data_predict = dpre.y_to_nan()
        try:
            LOGGER.info('Inference started...')
            yhat = self.model.predict(data_predict.x)
            LOGGER.info(
                'Inference complete...now preparing predictions for submission'
            )
        except Exception as e:
            LOGGER.error(f'Failure to make predictions with {e}')
            raise e

        try:
            prediction = prediction.merge_arrays(data_predict.ids, yhat,
                                                 self.name, tournament)
            return prediction
        except Exception as e:
            LOGGER.error(f'Failure to prepare predictions with {e}')
            raise e

    def fit_predict(self, dfit, dpre, tournament) -> Tuple:
        # fit is done separately in `.fit()`
        yhat = self.model.predict(dpre.x)
        return dpre.ids, yhat

    def save_to_s3(self, filename: str, key: str, credentials: Any) -> None:
        """Save model to s3 bucket"""
        s3 = utils.S3Client(user=credentials['user'],
                            password=credentials['password'],
                            bucket=credentials['bucket'])
        s3.upload_file(filename=filename, key=key)

    def load_from_s3(self, filename: str, key: str, credentials: Any) -> None:
        """Download model from s3 bucket"""
        s3 = utils.S3Client(user=credentials['user'],
                            password=credentials['password'],
                            bucket=credentials['bucket'])
        s3.download_file(filename=filename, key=key)

    def save(self, filename) -> None:
        """Serialize model locally"""
        joblib.dump(self, filename)

    @classmethod
    def load(cls, filename) -> Any:
        """Load trained model"""
        return joblib.load(filename)
