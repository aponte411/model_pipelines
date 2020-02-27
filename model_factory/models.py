import os
from datetime import datetime
from typing import Any, List, Tuple

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
from sklearn import metrics
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor, StackingRegressor,
                              VotingRegressor)
from sklearn.linear_model import LinearRegression
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from xgboost import XGBoostClassifier, XGBRegressor

import utils
from metrics import macro_recall

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


# WIP
class ResNet34Lightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = pretrainedmodels.__dict__["resnet34"](
            pretrained="imagenet")
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

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_end(self, outputs):
        # OPTIONAL
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.02)

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_path, batch_size=32)

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(self.val_path, batch_size=32)

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(self.test_path, batch_size=32)

    def _loss_fn(self, outputs, targets):
        output1, output2, output3 = outputs
        target1, target2, target3 = targets
        loss1 = self.criterion(output1, target1)
        loss2 = self.criterion(output2, target2)
        loss3 = self.criterion(output3, target3)
        return (loss1 + loss2 + loss3) / 3

    def _load_to_gpu_float(self, data):
        return data.to(self.device, dtype=torch.float)

    def _load_to_gpu_long(self, data):
        return data.to(self.device, dtype=torch.long)

    def train(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        final_loss = 0
        counter = 0
        final_outputs, final_targets = [], []
        for batch, data in tqdm(enumerate(data_loader)):
            counter += 1
            image = self._load_to_gpu_float(data["image"])
            grapheme_root = self._load_to_gpu_long(data["grapheme_root"])
            vowel_diacritic = self._load_to_gpu_long(data["vowel_diacritic"])
            consonant_diacritic = self._load_to_gpu_long(
                data["consonant_diacritic"])
            self.optimizer.zero_grad()
            outputs = self.model(image)
            targets = [grapheme_root, vowel_diacritic, consonant_diacritic]
            loss = self._loss_fn(outputs=outputs, targets=targets)
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

        LOGGER.info(f'loss: {final_loss/counter}')
        LOGGER.info(f'macro-recall: {macro_recall}')

        return final_loss / counter, macro_recall

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        with torch.no_grad():
            self.model.eval()
            final_loss = 0
            counter = 0
            final_outputs, final_targets = [], []
            for batch, data in tqdm(enumerate(data_loader)):
                counter += 1
                image = self._load_to_gpu_float(data["image"])
                grapheme_root = self._load_to_gpu_long(data["grapheme_root"])
                vowel_diacritic = self._load_to_gpu_long(
                    data["vowel_diacritic"])
                consonant_diacritic = self._load_to_gpu_long(
                    data["consonant_diacritic"])

                outputs = self.model(image)
                targets = [grapheme_root, vowel_diacritic, consonant_diacritic]
                final_loss += self._loss_fn(outputs=outputs, targets=targets)

                output1, output2, output3 = outputs
                target1, target2, target3 = targets
                final_outputs.append(
                    torch.cat((output1, output2, output3), dim=1))
                final_targets.append(
                    torch.stack((target1, target2, target3), dim=1))

            final_outputs = torch.cat(final_outputs)
            final_targets = torch.cat(final_targets)
            macro_recall_score = macro_recall(final_outputs, final_targets)

        return final_loss / counter, macro_recall_score
