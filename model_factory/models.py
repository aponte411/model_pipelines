import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import pretrainedmodels
import pytorch_lightning as pl
import tensorflow as tf
import torch
import torch.nn as nn
import xgboost as xgb
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

import utils
from datasets import BengaliDataSetTest, BengaliDataSetTrain
from metrics import macro_recall

LOGGER = utils.get_logger(__name__)


class BaseModel:
    def __init__(self, **kwds):
        super().__init__(**kwds)

    def save_to_s3(self, filename: str, key: str) -> None:
        """Save model to s3 bucket"""
        s3 = utils.S3Client()
        s3.upload_file(filename=filename, key=key)

    def load_from_s3(self, filename: str, key: str) -> None:
        """Download model from s3 bucket"""
        s3 = utils.S3Client()
        s3.download_file(filename=filename, key=key)

    def save(self, filename):
        """Serialize model"""
        joblib.dump(self, filename)

    def load(self, filename):
        """Load trained model"""
        return joblib.load(filename)


class ResNet34(nn.Module, BaseModel):
    def __init__(self, pretrained: bool, **kwds):
        super().__init__(**kwds)
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
    def __init__(self, pretrained: bool, **kwds):
        super().__init__(**kwds)
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
    def __init__(self, pretrained: bool = True, **kwds):
        super().__init__(**kwds)
        if pretrained:
            self.model = pretrainedmodels.__dict__["se_resnext101_32x4d"](
                pretrained='image_net')
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
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
