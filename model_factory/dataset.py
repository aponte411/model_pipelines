from typing import Any, List, Tuple

import pandas as pd
from sklearn import model_selection
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

import utils

LOGGER = utils.get_logger(__name__)


class DataSet:
    def __init__(
        self,
        path: str,
        target: Any,
    ):
        self.path = path
        self.target = target
        self.feats = None
        self.fold_mapping = {
            0: [1, 2, 3, 4],
            1: [0, 2, 3, 4],
            2: [0, 1, 3, 4],
            3: [0, 1, 2, 4],
            4: [0, 1, 2, 3],
        }
        self.train = None
        self.valid = None
        self.y_train = None
        self.y_val = None

    def _load_train_for_cv(self, input_path: str) -> pd.DataFrame:
        train = pd.read_csv(input_path)
        train['kfold'] = -1
        return train

    def prepare_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        LOGGER.info(f'Loading training data from: {self.path}')
        LOGGER.info(f'Fold: {fold}')
        df = pd.read_csv(self.path)
        self.train = df.loc[df.kfold.isin(
            self.fold_mapping.get(fold))].reset_index(drop=True)
        self.valid = df.loc[df.kfold == fold]
        del df

        return self.train, self.valid

    def get_targets(self) -> Tuple:
        self.y_train = self.train[self.target].values
        self.y_val = self.valid[self.target].values
        return self.y_train, self.y_val

    def clean_data(self,
                   to_drop: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.train = self.train.drop(to_drop, axis=1).reset_index(drop=True)
        self.valid = self.valid.drop(to_drop, axis=1).reset_index(drop=True)
        self.valid = self.valid[self.train.columns]
        LOGGER.info(f'Train: {self.train.shape}')
        LOGGER.info(f'Val: {self.valid.shape}')
        return self.train, self.valid


class QuoraDataSet(DataSet):
    def __init__(self,
                 path="inputs/quora_question_pairs/train-folds.csv",
                 target="is_duplicate"):
        super().__init__(path=path, target=target)
        self.path = path
        self.target = target

    def apply_stratified_kfold(self, input: str, output: str) -> None:
        train = self._load_train_for_cv(input_path=input)
        kf = model_selection.StratifiedKFold(n_splits=5,
                                             shuffle=True,
                                             random_state=123)
        for fold, (train_idx, val_idx) in enumerate(
                kf.split(X=train, y=train[self.target].values)):
            LOGGER.info(f'Train index: {len(train_idx)}, Val index: {val_idx}')
            train.loc[val_idx, 'kfold'] = fold

        LOGGER.info(f'Saving train folds to disk at {output}')
        train.to_csv(output, index=False)
        self.train = train
        self.feats = [
            col for col in train.columns
            if col != self.target and col != 'kfold'
        ]


class BengaliDataSet(DataSet):
    def __init__(self,
                 path: str = "inputs/bengali_grapheme/train-folds.csv",
                 target: List[str] = [
                     "grapheme_root", "vowel_diacritic", "consonant_diacritic"
                 ]):
        super().__init__(path=path, target=target)

    def _split_data(self,
                    train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = train.image_id.values
        y = train[self.target].values
        return X, y

    def apply_multilabel_stratified_kfold(self, input: str,
                                          output: str) -> None:
        train = self._load_train_for_cv(input_path=input)
        X, y = self._split_data(train=train)
        mskf = MultilabelStratifiedKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
            LOGGER.info(f'Train index: {len(train_idx)}, Val index: {val_idx}')
            train.loc[val_idx, 'kfold'] = fold

        LOGGER.info(f'Saving train folds to disk at {output}')
        train.to_csv(output, index=False)
        self.train = train
        self.feats = [
            col for col in train.columns
            if col not in self.target and col != 'kfold'
        ]
