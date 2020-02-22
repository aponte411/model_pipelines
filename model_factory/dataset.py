import pandas as pd
from sklearn import model_selection

import utils

LOGGER = utils.get_logger(__name__)


class DataSet:
    def __init__(self, path: str, fold: int):
        self.path = path
        self.fold = fold
        self.to_drop = to_drop
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

    def load_training_data(self, path: str) -> pd.DataFrame:
        train = pd.read_csv(path)
        train['kfold'] = -1
        return train

    def prepare_data(self) -> Tuple:
        LOGGER.info(f'Loading training data from: {self.path}')
        LOGGER.info(f'Fold: {self.fold}')
        df = pd.read_csv(self.path, index_col=0)
        self.train = df.loc[df.kfold.isin(self.fold_mapping.get(
            self.fold))].reset_index(drop=True)
        self.valid = df.loc[df.kfold == self.fold]
        del df

        return self.train, self.valid

    def get_targets(self) -> Tuple:
        self.y_train = self.train[self.target].values
        self.y_val = self.valid[self.target].values
        return self.y_train, self.y_val

    def clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.train = self.train.drop(self.to_drop,
                                     axis=1).reset_index(drop=True)
        self.valid = self.valid.drop(to_drop, axis=1).reset_index(drop=True)
        self.valid = self.valid[self.train.columns]
        LOGGER.info(f'Train: {self.train.shape}')
        LOGGER.info(f'Val: {self.valid.shape}')
        return self.train, self.valid


class QuoraDataSet(DataSet):
    def __init__(self, path: str, fold: int):
        super().__init__(path=path, fold=fold)

    def apply_stratified_kfold(self, train: pd.DataFrame, path: str,
                               target: str) -> None:

        kf = model_selection.StratifiedKFold(n_splits=5,
                                             shuffle=True,
                                             random_state=123)
        for fold, (train_idx, val_idx) in enumerate(
                kf.split(X=train, y=train[target].values)):
            LOGGER.info(f'Train index: {len(train_idx)}, Val index: {val_idx}')
            train.loc[val_idx, 'kfold'] = fold

        LOGGER.info(f'Saving train folds to disk at {path}')
        train.reset_index(drop=True)
        train.to_csv(path)