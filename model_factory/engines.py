from typing import Dict, List, Optional, Tuple
import click
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import utils
from datasets import BengaliDataSetTest, BengaliDataSetTrain
from trainers import BengaliTrainer, Trainer

LOGGER = utils.get_logger(__name__)

# requires CUDA to be enabled for OSX
class BengaliEngine:
    """
    The BengaliEngine will combine trainers, datasets, and models
    into a single object that contains functionality to train models
    and conduct inference.

    Arguments:
        name {str} - name of engine
        trainer {Trainer} - Trainer object that handles training
        params {Dict} - parameter dictionary
    """
    def __init__(self, name: str, trainer: Trainer, params: Dict, **kwds):
        super().__init__(**kwds)
        self.name = name
        self.trainer = trainer
        self.training_constructor = BengaliDataSetTrain
        self.val_constructor = BengaliDataSetTrain
        self.test_constructor = BengaliDataSetTest
        self.params = params
        self.model_name = None
        self.model_state_path = None
        self.model_path = None

    def _get_training_loader(self, folds: List[int], name: str) -> DataLoader:
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
                          batch_size=self.params["train_batch_size"],
                          shuffle=True,
                          num_workers=4)

    def _get_all_testing_loaders(self):
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

    def run_training_engine(self,
                            load: bool = False,
                            save: bool = True,
                            model_dir: str = "trained_models") -> None:
        """
        Trains a ResNet34 model for the BengaliAI bengali grapheme competition.
        """
        train = self._get_training_loader(folds=self.params["train_folds"],
                                          name='training')
        val = self._get_training_loader(folds=self.params["val_folds"],
                                        name='val')
        self.model_name = f"{self.trainer.get_model_name()}_bengali"
        self.model_state_path = f"{model_dir}/{self.model_name}_fold{self.params['train_folds']}.pth"
        self.model_path = f'{model_dir}/{self.model_name}.p'
        best_score = -1
        for epoch in range(self.params["epochs"]):
            LOGGER.info(f'EPOCH: {epoch}')
            train_loss, train_score = self.trainer.train(train)
            val_loss, val_score = self.trainer.evaluate(val)
            if val_score > best_score:
                best_score = val_score
                torch.save(self.trainer.model.state_dict(), model_state_path)
            LOGGER.info(
                f'Train loss: {train_loss}, Train score: {train_score}')
            LOGGER.info(
                f'Validation loss: {val_loss}, Validation score: {val_score}')
            self.trainer.scheduler.step(val_loss)
            self.trainer.early_stopping(val_score, self.trainer.model)
            if self.trainer.early_stopping.early_stop:
                LOGGER.info(f"Early stopping at epoch: {epoch}")
                self.trainer.save_model_locally(key=model_path)
                self.trainer.save_to_s3(filename=model_path,
                                        key=self.model_name)
                break

        self.trainer.save_model_locally(key=model_path)
        self.trainer.save_to_s3(filename=model_path, key=model_name)

    def run_inference_engine(self) -> pd.DataFrame:
        """Conducts inference using the test set.

        Returns:
            pd.DataFrame -- Predictions ready for submission to the leaderboard
        """
        def _inference() -> Dict:
            predictions = {}
            testing_loaders = self._get_all_testing_loaders()
            for loader in testing_loaders:
                for batch, data in enumerate(loader):
                    image = data["image"]
                    img_id = data["image_id"]
                    image = image.to(self.trainer.device, dtype=torch.float)
                    grapheme, vowel, consonant = self.trainer.model(image)
                    for idx, image_id in enumerate(img_id):
                        predictions["grapheme"] = grapheme[idx].cpu().detach(
                        ).numpy()
                        predictions["vowel"] = vowel[idx].cpu().detach().numpy(
                        )
                        predictions["consonant"] = consonant[idx].cpu().detach(
                        ).numpy()
                        predictions["image_id"] = image_id

            return predictions

        def _get_final_preds(preds: Dict) -> Dict:
            return {
                "final_grapheme":
                np.argmax(np.mean(final_predictions["grapheme"], axis=0),
                          axis=1),
                "final_vowel":
                np.argmax(np.mean(final_predictions["vowel"], axis=0), axis=1),
                "final_consonant":
                np.argmax(np.mean(final_predictions["consonant"], axis=0),
                          axis=1)
            }

        def _create_submission_df(pred_dict: Dict) -> pd.DataFrame:
            predictions = []
            for idx, image_id in enumerate(pred_dict["image_id"]):
                predictions.append((f"{image_id}_grapheme_root",
                                    pred_dict["final_grapheme"][idx]))
                predictions.append((f"{image_id}_vowel_diacritic",
                                    pred_dict["final_vowel"][idx]))
                predictions.append((f"{image_id}_consonant_diacritic",
                                    pred_dict["final_consonant"][idx]))

            return pd.DataFrame(predictions, columns=["row_id", "target"])

        final_predictions = {}
        for idx in range(self.params["test_loops"]):
            self.trainer.model.load_state_dict(
                torch.load(self.model_state_path))
            self.trainer.model.to(self.trainer.device)
            self.trainer.model.eval()
            predictions = _inference()
            final_predictions["grapheme"] = predictions["grapheme"]
            final_predictions["vowel"] = predictions["vowel"]
            final_predictions["consonant"] = predictions["consonant"]
            if idx == 0:
                final_predictions["image_id"] = predictions["image_id"]

        pred_dictionary = _get_final_preds(preds=final_predictions)
        return _create_submission_df(preds=pred_dictionary)
