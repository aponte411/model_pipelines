import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import utils
from datasets import BengaliDataSetTest, BengaliDataSetTrain
from trainers import BaseTrainer, BengaliTrainer

LOGGER = utils.get_logger(__name__)


# requires CUDA to be enabled for OSX
class BengaliEngine:
    """
    The BengaliEngine will combine trainers, datasets, and models
    into a single object that contains functionality to train models
    and conduct inference.

    Args:
        trainer {Trainer} - Trainer object that handles training and
        serialization.
        params {Dict} - Parameter dictionary containing engine arguments
        such as the number of epochs, paths to data, and preprocessing
        parameters, e.g.
            - train_path {str}: "inputs/train-folds.csv",
            - test_path {str}: "inputs",
            - pickle_path {str}: "inputs/pickled_images",
            - model_dir {str}: "trained_models",
            - train_folds {List[int]}: [0],
            - val_folds {List[int]}: [4],
            - train_batch_size {int}: 64,
            - test_batch_size {int}: 32,
            - epochs {int}: 3,
            - test_loops {int}: 5,
            - image_height {int}: 137,
            - image_width {int}: 236,
            - mean {Tuple[float]}: (0.485, 0.456, 0.406),
            - std {Tuple[float]}: (0.229, 0.239, 0.225)
    """
    def __init__(self, trainer: BaseTrainer, params: Dict, **kwds):
        super().__init__(**kwds)
        self.trainer = trainer
        self.training_constructor = BengaliDataSetTrain
        self.val_constructor = BengaliDataSetTrain
        self.test_constructor = BengaliDataSetTest
        self.params = params
        self.model_name = None
        self.model_state_path = None

    def _get_training_loader(self, folds: List[int], name: str) -> DataLoader:
        if name == "val":
            batch_size = self.params["test_batch_size"]
        else:
            batch_size = self.params["train_batch_size"]
        constructor = getattr(self, f'{name}_constructor')
        setattr(
            self, f'{name}_set',
            constructor(train_path=self.params["train_path"],
                        pickle_path=self.params["pickle_path"],
                        folds=folds,
                        image_height=self.params["image_height"],
                        image_width=self.params["image_width"],
                        mean=self.params["mean"],
                        std=self.params["std"]))
        return DataLoader(dataset=getattr(self, f'{name}_set'),
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

    def _get_all_testing_loaders(self) -> List[DataLoader]:
        def _get_loader(df: pd.DataFrame) -> DataLoader:
            test_set = self.test_constructor(
                df=df,
                image_height=self.params["image_height"],
                image_width=self.params["image_width"],
                mean=self.params["mean"],
                std=self.params["std"])
            return DataLoader(dataset=test_set,
                              batch_size=self.params["test_batch_size"],
                              shuffle=False,
                              num_workers=4)

        loaders = []
        for idx in range(4):
            df = pd.read_parquet(
                f"{self.params['test_path']}/test_image_data_{idx}.parquet")
            loaders.append(_get_loader(df=df))
        return loaders

    def run_training_engine(self) -> None:
        """
        Trains a ResNet34 model for the BengaliAI bengali grapheme competition.
        """
        LOGGER.info(
            f'Training the model using folds: {self.params["train_folds"]}')
        LOGGER.info(
            f'Validating the model using folds {self.params["val_folds"]}')
        LOGGER.info(f'Using {torch.cuda.device_count()} GPUs')
        if torch.cuda.device_count() > 1:
            self.trainer.model = nn.DataParallel(self.trainer.model)
        train = self._get_training_loader(folds=self.params["train_folds"],
                                          name='training')
        val = self._get_training_loader(folds=self.params["val_folds"],
                                        name='val')
        self.model_name = f"{self.trainer.get_model_name()}_bengali"
        self.model_state_path = f"{self.params['model_dir']}/{self.model_name}_fold{self.params['val_folds'][0]}.pth"
        best_score = -1
        for epoch in range(1, self.params["epochs"] + 1):
            LOGGER.info(f'EPOCH: {epoch}')
            train_loss, train_score = self.trainer.train(train)
            val_loss, val_score = self.trainer.evaluate(val)
            if val_score > best_score:
                best_score = val_score
                self.trainer.save_model_locally(
                    model_path=self.model_state_path)
            LOGGER.info(
                f'Training loss: {train_loss}, Training score: {train_score}')
            LOGGER.info(
                f'Validation loss: {val_loss}, Validation score: {val_score}')
            self.trainer.scheduler.step(val_loss)
            self.trainer.early_stopping(val_score, self.trainer.model)
            if self.trainer.early_stopping.early_stop:
                LOGGER.info(f"Early stopping at epoch: {epoch}")
                break

    def run_inference_engine(self,
                             model_dir: str,
                             to_csv: False,
                             output_dir: str = None) -> pd.DataFrame:
        """Conducts inference using the test set.

        Returns:
            submission_df {pd.DataFrame} -- A predictions dataframe ready for submission 
            to the public leaderboard.
        """
        def _conduct_inference() -> defaultdict:
            predictions = defaultdict(list)
            testing_loaders = self._get_all_testing_loaders()
            for loader in testing_loaders:
                for batch, data in enumerate(loader):
                    image = self.trainer._load_to_gpu_float(data["image"])
                    grapheme, vowel, consonant = self.trainer.model(image)
                    for idx, img_id in enumerate(data["image_id"]):
                        predictions["grapheme"].append(
                            grapheme[idx].cpu().detach().numpy())
                        predictions["vowel"].append(
                            vowel[idx].cpu().detach().numpy())
                        predictions["consonant"].append(
                            consonant[idx].cpu().detach().numpy())
                        predictions["image_id"].append(img_id)

            return predictions

        def _get_maximum_probs(preds: defaultdict) -> Dict:
            return {
                "final_grapheme":
                np.argmax(np.mean(preds["grapheme"], axis=0), axis=1),
                "final_vowel":
                np.argmax(np.mean(preds["vowel"], axis=0), axis=1),
                "final_consonant":
                np.argmax(np.mean(preds["consonant"], axis=0), axis=1),
                "image_ids":
                preds["image_id"]
            }

        def _create_submission_df(pred_dict: Dict) -> pd.DataFrame:
            predictions = []
            for idx, image_id in enumerate(pred_dict["image_ids"]):
                predictions.append((f"{image_id}_grapheme_root",
                                    pred_dict["final_grapheme"][idx]))
                predictions.append((f"{image_id}_vowel_diacritic",
                                    pred_dict["final_vowel"][idx]))
                predictions.append((f"{image_id}_consonant_diacritic",
                                    pred_dict["final_consonant"][idx]))

            return pd.DataFrame(predictions, columns=["row_id", "target"])

        final_predictions = defaultdict(list)
        for idx in range(1, self.params["test_loops"]):
            LOGGER.info(f'Conducting inference for fold {idx}')
            model_state_path = f'{model_dir}/restnet34_bengali_fold{idx}.pth'
            self.trainer.load_model_locally(model_path=model_state_path)
            self.trainer.model.to(self.trainer.device)
            self.trainer.model.eval()
            predictions = _conduct_inference()
            final_predictions["grapheme"].append(predictions["grapheme"])
            final_predictions["vowel"].append(predictions["vowel"])
            final_predictions["consonant"].append(predictions["consonant"])
            if idx == 1:
                final_predictions["image_id"].extend(predictions["image_id"])

        pred_dictionary = _get_maximum_probs(preds=final_predictions)
        submission_df = _create_submission_df(pred_dict=pred_dictionary)
        if to_csv:
            timestamp = datetime.datetime.today().strftime(
                "%B -%d,- %Y -%H:%M").replace(" ", "").replace(",", "")
            output_path = f"{output_dir}/submission_{timestamp}"
            LOGGER.info(f'Saving submission dataframe to {output_path}')
            submission_df.to_csv(output_path, index=False)

        return submission_df
