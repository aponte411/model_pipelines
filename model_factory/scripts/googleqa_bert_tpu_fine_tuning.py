from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
import yaml
from torch.utils.data import DataLoader, DistributedSampler

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp
import utils
from datasets import GoogleQADataSetTest, GoogleQADataSetTrain
from dispatcher import MODEL_DISPATCHER
from engines import Engine
from metrics import spearman_correlation
from trainers import BaseTrainer

CREDENTIALS = {}
CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
CREDENTIALS['bucket'] = os.environ.get("bucket")


class GoogleQATPUTrainer(BaseTrainer):
    """
    Trainer to handle training, inference, scoring, 
    and saving/loading weights using GCP TPU node.

    Args:
        model {Any} -- trainable model.
        model_name {str} -- name of model
    """
    def __init__(self, model: Any, model_name: str = None, **kwds):
        super().__init__(model, **kwds)
        self.model_name = model_name
        self.device = xm.xla_device()
        self.optimizer = transformers.AdamW(self.model.parameters(),
                                            lr=1e-4 * xm.xrt_world_size())
        self.criterion = nn.BCEWithLogitsLoss()
        self.early_stopping = utils.EarlyStopping(patience=5, verbose=True)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.3, verbose=True)

    def get_model_name(self) -> str:
        return self.model_name

    def _load_to_tpu_float(self, data) -> torch.Tensor:
        return data.to(self.device, dtype=torch.float)

    def _load_to_tpu_long(self, data) -> torch.Tensor:
        return data.to(self.device, dtype=torch.long)

    def _loss_fn(self, predictions: torch.Tensor,
                 targets: torch.Tensor) -> float:
        return self.criterion(predictions, targets)

    def _get_features(self, data) -> Tuple[torch.Tensor]:
        ids = self._load_to_tpu_long(data['ids'])
        token_type_ids = self._load_to_tpu_long(data['token_type_ids'])
        mask = self._load_to_tpu_long(data['attention_mask'])
        return ids, mask, token_type_ids

    def _get_targets(self, data) -> torch.Tensor:
        return self._load_to_tpu_float(data['targets'])

    @staticmethod
    def score(preds: List[torch.Tensor], targets: List[torch.Tensor]) -> float:
        final_preds = torch.cat(preds)
        final_targets = torch.cat(targets)
        return spearman_correlation(final_preds, final_targets)

    @staticmethod
    def stack_tensors(tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(tensor, dim=1)

    def save_model_locally(self, model_path: str) -> None:
        xm.master_print(f'Saving model to {model_path}')
        xm.save(self.model.state_dict(), model_path)

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
        xm.master_print(f'Saving model to s3 bucket..')
        s3 = utils.S3Client(user=creds["aws_access_key_id"],
                            password=creds["aws_secret_access_key"],
                            bucket=creds["bucket"])
        s3.upload_file(filename=filename, key=key)

    def load_model_locally(self, model_path: str) -> None:
        xm.master_print(f'Loading model from {model_path}')
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
        xm.master_print(f'Loading model from s3 bucket..')
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
            xm.optimizer_step(optimizer)
            xm.mark_step()
            final_loss += loss
            final_preds.append(self.stack_tensors(tensor=predictions))
            final_targets.append(self.stack_tensors(tensor=targets))

        spearman_correlation = self.score(preds=final_preds,
                                          targets=final_targets)
        xm.master_print(f'Training Loss: {final_loss/counter}')
        xm.master_print(
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
        xm.master_print(f'Validation Loss: {final_loss/counter}')
        xm.master_print(
            f'Validation Spearman Correlation Coefficient: {spearman_correlation}'
        )

        return final_loss / counter, spearman_correlation


class GoogleQATPUEngine(Engine):
    def __init__(self, trainer: trainers.BaseTrainer, config_file: str,
                 **kwds):
        super().__init__(trainer, **kwds)
        self.params: Dict = self.get_params(config_file)
        self.train_contructor = GoogleQADataSetTrain
        self.val_contructor = GoogleQADataSetTrain
        self.test_contructor = GoogleQADataSetTest
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True)

    def _get_training_loader(self, folds: List[int], name: str) -> DataLoader:
        if name == "val":
            batch_size = self.params["training_params"].get("test_batch_size")
        else:
            batch_size = self.params["training_params"].get("train_batch_size")
        constructor = getattr(self, f'{name}_constructor')
        setattr(
            self, f'{name}_set',
            constructor(
                data_folder=self.params["data_params"].get("train_path"),
                folds=folds,
                tokenizer=self.tokenzier,
                max_len=self.params["data_params"].get("max_len")))
        dataset = getattr(self, f'{name}_set')
        sampler = DistributedSampler(dataset,
                                     num_replicas=xm.xrt_world_size(),
                                     rank=xm.get_ordinal())
        return DataLoader(dataset=getattr(self, f'{name}_set'),
                          batch_size=batch_size,
                          sampler=sampler)

    @staticmethod
    def get_params(config_file: str) -> Dict:
        with open(config_file, 'rb') as f:
            return yaml.load(f)

    def run_training_engine(self, save_to_s3: bool = False, creds: Dict = {}):
        xm.master_print(
            f'Training the model using folds: {self.params["training_params"].get("train_folds")}'
        )
        xm.master_print(
            f'Validating the model using folds {self.params["training_params"].get("val_folds")[0]}'
        )
        train = self._get_training_loader(
            folds=self.params["training_params"].get("train_folds"),
            name="train")
        train_parallel_loader = pl.ParallelLoader(train, [self.trainer.device])
        val = self._get_training_loader(
            folds=self.params["training_params"].get("val_folds"), name="val")
        val_parallel_loader = pl.ParallelLoader(val, [self.trainer.device])
        self.model_name = f'{self.trainer.get_model_name()}_googleqa'
        model_with_val_fold = f'{self.model_name}_fold{self.params["training_params"].get("val_folds")}.pth'
        self.model_state_path = f'{self.params["model_params"].get("model_dir")}/{model_with_val_fold}'
        best_score = -1
        for epoch in range(1,
                           self.params["training_params"].get("epochs") + 1):
            xm.master_print(f'EPOCH: {epoch}')
            train_loss, train_score = self.trainer.train(
                train_parallel_loader.per_device_loader(self.trainer.device))
            val_loss, val_score = self.trainer.evaluate(
                val_parallel_loader.per_device_loader(self.trainer.device))
            if val_score > best_score:
                best_score = val_score
                self.trainer.save_model_locally(
                    model_path=self.model_state_path)
                if save_to_s3:
                    self.trainer.save_model_to_s3(
                        filename=self.model_state_path,
                        key=model_with_val_fold,
                        creds=creds)
            xm.master_print(
                f'Training loss: {train_loss:.3f}, Training score: {train_score:.3f}'
            )
            xm.master_print(
                f'Validation loss: {val_loss:.3f}, Validation score: {val_score:.3f}'
            )
            self.trainer.scheduler.step(val_loss)
            self.trainer.early_stopping(val_score, self.trainer.model)
            if self.trainer.early_stopping.early_stop:
                xm.master_print(f'Early stopping at epoch: {epoch}')
                break

    def run_inference_engine(self):
        pass


def run_googleqa_engine(index):
    model = MODEL_DISPATCHER['bert-googleqa']
    trainer = GoogleQATPUTrainer(model=model, model_name='bert_tpu')
    engine = GoogleQATPUEngine(trainer=trainer,
                               config_file='configs/bert_tpu_fine_tuning.yml')
    engine.run_training_engine(save_to_s3=True, creds=CREDENTIALS)


if __name__ == "__main__":
    xmp.spawn(run_googleqa_engine, nprocs=8)
