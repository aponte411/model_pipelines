import pandas as pd
from sklearn import preprocessing
import os

from utils import get_logger

LOGGER = get_logger(__name__)

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = os.environ.get("FOLD")

FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}


def prepare_data(TRAINING_DATA: str, FOLD: str, FOLD_MAPPING: str) -> Tuple:

    df = pd.read_csv(TRAINING_DATA)
    train_df = df.loc[df.kfold.isin(FOLD_MAPPING.get(FOLD))]
    valid_df = df.loc[df.kfold == FOLD]
    del df

    return train_df, valid_df


def get_targets(train: pd.DataFrame, val: pd.DataFrame) -> Tuple:

    y_train = train.target.values
    y_val = val.target.values

    return y_train, y_val


def clean_data(train: pd.DataFrame, val: pd.DataFrame) -> Tuple:

    train.drop(['id', 'target', 'kfold'], axis=1, inplace=True)
    val.drop(['id', 'target', 'kfold'], axis=1, inplace=True)
    val = val[train.columns]

    return train, val


def main():

    train, val = prepare_data(TRAINING_DATA, FOLD, FOLD_MAPPING)
    y_train, y_val = get_targets(train, val)
    X_train, X_val = clean_data(train, val)

    LOGGER.info(f'Train shape: {X_train.shape}')
    LOGGER.info(f'Val shape: {X_val.shape}')


if __name__ == "__main__":
    main()
