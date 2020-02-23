from typing import Any, List, Tuple

import pandas as pd
from sklearn import model_selection

import utils

LOGGER = utils.get_logger(__name__)


class DataSet:
    def __init__(
        self,
        path: str,
        target: str,
    ):
        self.path = path
        self.target = target
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
        df = pd.read_csv(self.path, index_col=0)
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
        train.reset_index(drop=True)
        train.to_csv(output, index_label='Unnamed: 0.1')
        self.train = train


class BengaliDataSet(DataSet):
    def __init__(self, path: str, fold: int):
        super().__init__(path=path, fold=fold)
