import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn import model_selection

from datasets import BengaliDataSetTrain
from utils import get_logger

LOGGER = get_logger(__name__)


class CrossValidator:
    def __init__(self, input_path: str, output_path: str, target: Any):
        self.input_path = input_path
        self.output_path = output_path
        self.target = target
        self.fold_mapping = {
            0: [1, 2, 3, 4],
            1: [0, 2, 3, 4],
            2: [0, 1, 3, 4],
            3: [0, 1, 2, 4],
            4: [0, 1, 2, 3],
        }

    def _load_train_for_cv(self, input_path: str) -> pd.DataFrame:
        train = pd.read_csv(input_path)
        train['kfold'] = -1
        return train

    def split_data(self, fold: int,
                   input_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        LOGGER.info(f'Loading training data from: {input_path}')
        LOGGER.info(f'Fold: {fold}')
        df = pd.read_csv(input_path)
        train = df.loc[df.kfold.isin(
            self.fold_mapping.get(fold))].reset_index(drop=True)
        valid = df.loc[df.kfold == fold]
        del df

        return train, valid

    def get_targets(self, train: pd.DataFrame,
                    valid: pd.DataFrame) -> Tuple[np.array, np.array]:
        y_train = train[self.target].values
        y_val = valid[self.target].values
        return y_train, y_val

    @staticmethod
    def clean_data(train: pd.DataFrame, valid: pd.DataFrame,
                   to_drop: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train = train.drop(to_drop, axis=1).reset_index(drop=True)
        valid = valid.drop(to_drop, axis=1).reset_index(drop=True)
        valid = valid[train.columns]
        LOGGER.info(f'Train: {train.shape}')
        LOGGER.info(f'Val: {valid.shape}')
        return train, valid


class QuoraCrossValidator(CrossValidator):
    def __init__(self, input_path: str, output_path: str, target: str):
        super().__init__(input_path, output_path, target)

    def apply_stratified_kfold(self) -> None:
        train = self._load_train_for_cv(input_path=self.input_path)
        kf = model_selection.StratifiedKFold(n_splits=5,
                                             shuffle=True,
                                             random_state=123)
        for fold, (train_idx, val_idx) in enumerate(
                kf.split(X=train, y=train[self.target].values)):
            LOGGER.info(f'Train index: {len(train_idx)}, Val index: {val_idx}')
            train.loc[val_idx, 'kfold'] = fold

        LOGGER.info(f'Saving train folds to disk at {self.output_path}')
        train.to_csv(self.output_path, index=False)


class BengaliCrossValidator(CrossValidator):
    def __init__(self, input_path: str, output_path: str, target: str):
        super().__init__(input_path, output_path, target)

    def _split_data(self,
                    train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = train.image_id.values
        y = train[self.target].values
        return X, y

    def apply_multilabel_stratified_kfold(self) -> None:
        train = self._load_train_for_cv(input_path=self.input_path)
        X, y = self._split_data(train=train)
        mskf = MultilabelStratifiedKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
            LOGGER.info(f'Train index: {len(train_idx)}, Val index: {val_idx}')
            train.loc[val_idx, 'kfold'] = fold

        LOGGER.info(f'Saving train folds to disk at {self.output_path}')
        train.to_csv(self.output_path, index=False)
