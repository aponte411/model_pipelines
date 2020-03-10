import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from pytorch_lightning import Trainer
from sklearn import metrics, preprocessing
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

import models
import utils
from dispatcher import MODEL_DISPATCHER
from metrics import macro_recall, spearman_correlation
from utils import EarlyStopping

warnings.filterwarnings('ignore')

LOGGER = utils.get_logger(__name__)


class BaseTrainer(ABC):
    """Base class for training/inference

    Args:
        model {Any} -- object from models module.
    """
    def __init__(self, model: Any):
        super().__init__()
        self.model = model

    @abstractmethod
    def train(self):
        """Login to train models"""
        raise NotImplementedError()

    def get_model(self):
        return self.model


class QuoraTrainer(BaseTrainer):
    def __init__(self, model: Any):
        super().__init__(model)

    def load_model_locally(self, key: str):
        LOGGER.info(f"Using saved model for {self.tournament}")
        self.model = MODEL_DISPATCHER['randomforest']
        self.model.load(key)

    def load_from_s3(self, filename: str, key: str):
        self.model = MODEL_DISPATCHER['randomforest']
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


class BengaliTrainer(BaseTrainer):
    def __init__(self, model: Any, model_name: str = None):
        super().__init__(model)
        self.model_name = model_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        self.criterion = nn.CrossEntropyLoss()
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.3, verbose=True)

    def get_model_name(self):
        return self.model_name

    def _loss_fn(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        pred1, pred2, pred3 = preds
        target1, target2, target3 = targets
        loss1 = self.criterion(pred1, target1)
        loss2 = self.criterion(pred2, target2)
        loss3 = self.criterion(pred3, target3)
        return (loss1 + loss2 + loss3) / 3

    def _load_to_gpu_float(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(self.device, dtype=torch.float)

    def _load_to_gpu_long(self, data: torch.Tensor) -> torch.Tensor:
        return data.to(self.device, dtype=torch.long)

    def _get_image(self, data: torch.Tensor) -> torch.Tensor:
        return self._load_to_gpu_float(data["image"])

    def _get_targets(self, data: torch.Tensor) -> List[torch.Tensor]:
        grapheme_root = self._load_to_gpu_long(data["grapheme_root"])
        vowel_diacritic = self._load_to_gpu_long(data["vowel_diacritic"])
        consonant_diacritic = self._load_to_gpu_long(
            data["consonant_diacritic"])
        return [grapheme_root, vowel_diacritic, consonant_diacritic]

    @staticmethod
    def score(preds: torch.Tensor, targets: torch.Tensor) -> float:
        final_preds = torch.cat(preds)
        final_targets = torch.cat(targets)
        return macro_recall(final_preds, final_targets)

    @staticmethod
    def concat_tensors(tensor: torch.Tensor) -> torch.Tensor:
        one, two, three = tensor
        return torch.cat((one, two, three), dim=1)

    @staticmethod
    def stack_tensors(tensor: torch.Tensor) -> torch.Tensor:
        one, two, three = tensor
        return torch.stack((one, two, three), dim=1)

    def save_model_locally(self, model_path: str) -> None:
        LOGGER.info(f'Saving model to {model_path}')
        torch.save(self.model.state_dict(), model_path)

    def save_model_to_s3(self, filename: str, key: str, creds: Dict) -> None:
        """
        Saves trained model to s3 bucket. Requires credentials
        dictionary. E.g.

            CREDENTIALS = {}
            CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
            CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
            CREDENTIALS['bucket'] = os.environ.get("bucket")

        Args:
            filename {str} -- Path to model directory
            key {str} -- Object name
            creds {Dict} -- Credentials dictionary containing AWS aws_access_key_id,
            aws_secret_access_key, and bucket.
        """
        LOGGER.info(f'Saving model to s3 bucket..')
        s3 = utils.S3Client(user=creds["aws_access_key_id"],
                            password=creds["aws_secret_access_key"],
                            bucket=creds["bucket"])
        s3.upload_file(filename=filename, key=key)

    def load_model_locally(self, model_path: str) -> None:
        LOGGER.info(f'Loading model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))

    def load_model_from_s3(self, filename: str, key: str, creds: Dict) -> None:
        """
        Loads trained model from s3 bucket. Requires credentials
        dictionary. E.g.

            CREDENTIALS = {}
            CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
            CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
            CREDENTIALS['bucket'] = os.environ.get("bucket")

        Args:
            filename {str} -- Path to model directory
            key {str} -- Object name
            creds {Dict} -- Credentials dictionary containing AWS aws_access_key_id,
            aws_secret_access_key, and bucket.
        """
        LOGGER.info(f'Loading model from s3 bucket..')
        s3 = utils.S3Client(user=creds["aws_access_key_id"],
                            password=creds["aws_secret_access_key"],
                            bucket=creds["bucket"])
        s3.download_file(filename=filename, key=key)

    def train(self, data_loader: DataLoader) -> Tuple[float, float]:
        # self.model.to(self.device)
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
        LOGGER.info(f'Training Loss: {final_loss/counter}')
        LOGGER.info(f'Training Macro-Recall: {macro_recall_score}')

        return final_loss / counter, macro_recall_score

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        with torch.no_grad():
            self.model.to(self.device)
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
        LOGGER.info(f'Validation Loss: {final_loss/counter}')
        LOGGER.info(f'Validation Macro-Recall: {macro_recall_score}')

        return final_loss / counter, macro_recall_score

    def inference(self, data_loader):
        pass


# WIP
class BengaliLightningTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def train_model(self, model: Any):
        self.fit(model)


class GoogleQATrainer(BaseTrainer):
    """
    Trainer to handle training, inference, scoring, 
    and saving/loading weights.

    Args:
        model {Any} -- trainable model.
        model_name {str} -- name of model
    """
    def __init__(self, model: Any, model_name: str = None):
        super().__init__(model)
        self.model_name = model_name
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.optimizer = transformers.AdamW(self.model.parameters(), lr=1e-4)
        self.criterion = nn.BCEWithLogitsLoss()
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.3, verbose=True)

    def get_model_name(self) -> str:
        return self.model_name

    def _load_to_gpu_float(self, data) -> torch.Tensor:
        return data.to(self.device, dtype=torch.float)

    def _load_to_gpu_long(self, data) -> torch.Tensor:
        return data.to(self.device, dtype=torch.long)

    def _loss_fn(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> float:
        return self.criterion(predictions, targets)

    def _get_features(self, data) -> Tuple[torch.Tensor]:
        ids = self._load_to_gpu_long(data['ids'])
        token_type_ids = self._load_to_gpu_long(data['token_type_ids'])
        mask = self._load_to_gpu_long(data['attention_mask'])
        return ids, mask, token_type_ids

    def _get_targets(self, data) -> torch.Tensor:
        return self._load_to_gpu_float(data['targets'])

    @staticmethod
    def score(preds: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
        final_preds = torch.cat(preds)
        final_targets = torch.cat(targets)
        return spearman_correlation(final_preds, final_targets)

    @staticmethod
    def stack_tensors(tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(tensor, dim=1)

    def save_model_locally(self, model_path: str) -> None:
        LOGGER.info(f'Saving model to {model_path}')
        torch.save(self.model.state_dict(), model_path)

    def save_model_to_s3(self, filename: str, key: str, creds: Dict) -> None:
        """
        Saves trained model to s3 bucket. Requires credentials
        dictionary. E.g.

            CREDENTIALS = {}
            CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
            CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
            CREDENTIALS['bucket'] = os.environ.get("bucket")

        Args:
            filename {str} -- Path to model directory
            key {str} -- Object name
            creds {Dict} -- Credentials dictionary containing AWS aws_access_key_id,
            aws_secret_access_key, and bucket.
        """
        LOGGER.info(f'Saving model to s3 bucket..')
        s3 = utils.S3Client(user=creds["aws_access_key_id"],
                            password=creds["aws_secret_access_key"],
                            bucket=creds["bucket"])
        s3.upload_file(filename=filename, key=key)

    def load_model_locally(self, model_path: str) -> None:
        LOGGER.info(f'Loading model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))

    def load_model_from_s3(self, filename: str, key: str, creds: Dict) -> None:
        """
        Loads trained model from s3 bucket. Requires credentials
        dictionary. E.g.

            CREDENTIALS = {}
            CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
            CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
            CREDENTIALS['bucket'] = os.environ.get("bucket")

        Args:
            filename {str} -- Path to model directory
            key {str} -- Object name
            creds {Dict} -- Credentials dictionary containing AWS aws_access_key_id,
            aws_secret_access_key, and bucket.
        """
        LOGGER.info(f'Loading model from s3 bucket..')
        s3 = utils.S3Client(user=creds["aws_access_key_id"],
                            password=creds["aws_secret_access_key"],
                            bucket=creds["bucket"])
        s3.download_file(filename=filename, key=key)

    def train(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.to(self.device)
        self.model.train()
        final_loss = 0
        counter = 0
        final_preds, final_targets = [], []
        for batch, data in tqdm(enumerate(data_loader)):
            counter += 1
            ids, mask, token_type_ids = self._get_features(data=data)
            targets = self._get_targets(data=data)
            self.optimizer.zero_grad()
            predictions = self.model(ids=ids,
                                     mask=mask,
                                     token_type_ids=token_type_ids)
            loss = self._loss_fn(preds=predictions, targets=targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            final_loss += loss
            final_preds.append(self.stack_tensors(tensor=predictions))
            final_targets.append(self.stack_tensors(tensor=targets))

        spearman_correlation = self.score(preds=final_preds,
                                          targets=final_targets)
        LOGGER.info(f'Training Loss: {final_loss/counter}')
        LOGGER.info(
            f'Training Spearman Correlation Coefficient: {spearman_correlation}'
        )

        return final_loss / counter, spearman_correlation

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            final_loss = 0
            counter = 0
            final_preds, final_targets = [], []
            for batch, data in tqdm(enumerate(data_loader)):
                counter += 1
                ids, mask, token_type_ids = self._get_features(data=data)
                targets = self._get_targets(data=data)
                predictions = self.model(ids=ids,
                                         mask=mask,
                                         token_type_ids=token_type_ids)
                loss = self._loss_fn(preds=predictions, targets=targets)
                final_loss += self._loss_fn(preds=predictions, targets=targets)
                final_preds.append(self.concat_tensors(tensor=predictions))
                final_targets.append(self.stack_tensors(tensor=targets))

            spearman_correlation = self.score(preds=final_preds,
                                              targets=final_targets)
        LOGGER.info(f'Validation Loss: {final_loss/counter}')
        LOGGER.info(
            f'Validation Spearman Correlation Coefficient: {spearman_correlation}'
        )

        return final_loss / counter, spearman_correlation


class IMDBTrainer(BaseTrainer):
    """
    Trainer to handle training, inference, scoring, 
    and saving/loading weights.

    Args:
        model {Any} -- trainable model.
        model_name {str} -- name of model
    """
    def __init__(self,
                 model: Any,
                 model_name: str = None,
                 num_training_steps: int = 10000 / 8 * 12):
        super().__init__(model)
        self.model_name = model_name
        self.num_training_steps = num_training_steps
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.BCEWithLogitsLoss()
        self.early_stopping = EarlyStopping(patience=5, verbose=True)
        self.setup_optimizer_and_scheduler

    @property
    def setup_optimizer_and_scheduler(self):
        def _filter_params(parameters: List,
                           filters: List[str],
                           exclude: bool = True) -> List[str]:
            if exclude:
                return [
                    parameter for name, parameter in parameters
                    if not any(param in name for param in filters)
                ]
            else:
                return [
                    parameter for name, parameter in parameters
                    if any(param in name for param in filters)
                ]

        model_params = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [{
            'params':
            _filter_params(params=model_params, filter=no_decay, exclude=True)
        }, {
            'params':
            _filter_params(params=model_params, filter=no_decay, exclude=False)
        }]
        self.optimizer = transformers.AdamW(optimizer_params, lr=1e-4)
        self.scheduler = transformers.get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_training_steps)

    def get_model_name(self) -> str:
        return self.model_name

    def _load_to_gpu_float(self, data) -> torch.Tensor:
        return data.to(self.device, dtype=torch.float)

    def _load_to_gpu_long(self, data) -> torch.Tensor:
        return data.to(self.device, dtype=torch.long)

    def _loss_fn(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> float:
        return self.criterion(predictions, targets)

    def _get_features(self, data) -> Tuple[torch.Tensor]:
        ids = self._load_to_gpu_long(data['ids'])
        token_type_ids = self._load_to_gpu_long(data['token_type_ids'])
        mask = self._load_to_gpu_long(data['attention_mask'])
        return ids, mask, token_type_ids

    def _get_targets(self, data) -> torch.Tensor:
        return self._load_to_gpu_float(data['targets'])

    @staticmethod
    def score(preds: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
        final_preds = torch.cat(preds)
        final_targets = torch.cat(targets)
        return spearman_correlation(final_preds, final_targets)

    @staticmethod
    def stack_tensors(tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(tensor, dim=1)

    def save_model_locally(self, model_path: str) -> None:
        LOGGER.info(f'Saving model to {model_path}')
        torch.save(self.model.state_dict(), model_path)

    def save_model_to_s3(self, filename: str, key: str, creds: Dict) -> None:
        """
        Saves trained model to s3 bucket. Requires credentials
        dictionary. E.g.

            CREDENTIALS = {}
            CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
            CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
            CREDENTIALS['bucket'] = os.environ.get("bucket")

        Args:
            filename {str} -- Path to model directory
            key {str} -- Object name
            creds {Dict} -- Credentials dictionary containing AWS aws_access_key_id,
            aws_secret_access_key, and bucket.
        """
        LOGGER.info(f'Saving model to s3 bucket..')
        s3 = utils.S3Client(user=creds["aws_access_key_id"],
                            password=creds["aws_secret_access_key"],
                            bucket=creds["bucket"])
        s3.upload_file(filename=filename, key=key)

    def load_model_locally(self, model_path: str) -> None:
        LOGGER.info(f'Loading model from {model_path}')
        self.model.load_state_dict(torch.load(model_path))

    def load_model_from_s3(self, filename: str, key: str, creds: Dict) -> None:
        """
        Loads trained model from s3 bucket. Requires credentials
        dictionary. E.g.

            CREDENTIALS = {}
            CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
            CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
            CREDENTIALS['bucket'] = os.environ.get("bucket")

        Args:
            filename {str} -- Path to model directory
            key {str} -- Object name
            creds {Dict} -- Credentials dictionary containing AWS aws_access_key_id,
            aws_secret_access_key, and bucket.
        """
        LOGGER.info(f'Loading model from s3 bucket..')
        s3 = utils.S3Client(user=creds["aws_access_key_id"],
                            password=creds["aws_secret_access_key"],
                            bucket=creds["bucket"])
        s3.download_file(filename=filename, key=key)

    def train(self, data_loader: DataLoader) -> Tuple[float, float]:
        self.model.to(self.device)
        self.model.train()
        final_loss = 0
        counter = 0
        final_preds, final_targets = [], []
        for batch, data in tqdm(enumerate(data_loader)):
            counter += 1
            ids, mask, token_type_ids = self._get_features(data=data)
            targets = self._get_targets(data=data)
            self.optimizer.zero_grad()
            predictions = self.model(ids=ids,
                                     mask=mask,
                                     token_type_ids=token_type_ids)
            loss = self._loss_fn(preds=predictions, targets=targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            final_loss += loss
            final_preds.append(
                self.stack_tensors(tensor=torch.sigmoid(predictions)))
            final_targets.append(self.stack_tensors(tensor=targets))

        spearman_correlation = self.score(preds=final_preds,
                                          targets=final_targets)
        LOGGER.info(f'Training Loss: {final_loss/counter}')
        LOGGER.info(
            f'Training Spearman Correlation Coefficient: {spearman_correlation}'
        )

        return final_loss / counter, spearman_correlation

    def evaluate(self, data_loader: DataLoader) -> Tuple[float, float]:
        with torch.no_grad():
            self.model.to(self.device)
            self.model.eval()
            final_loss = 0
            counter = 0
            final_preds, final_targets = [], []
            for batch, data in tqdm(enumerate(data_loader)):
                counter += 1
                ids, mask, token_type_ids = self._get_features(data=data)
                targets = self._get_targets(data=data)
                predictions = self.model(ids=ids,
                                         mask=mask,
                                         token_type_ids=token_type_ids)
                loss = self._loss_fn(preds=predictions, targets=targets)
                final_loss += self._loss_fn(preds=predictions, targets=targets)
                final_preds.append(self.stack_tensors(tensor=predictions))
                final_targets.append(self.stack_tensors(tensor=targets))

            spearman_correlation = self.score(preds=final_preds,
                                              targets=final_targets)
        LOGGER.info(f'Validation Loss: {final_loss/counter}')
        LOGGER.info(
            f'Validation Spearman Correlation Coefficient: {spearman_correlation}'
        )

        return final_loss / counter, spearman_correlation
