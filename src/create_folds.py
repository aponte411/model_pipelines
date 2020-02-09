import pandas as pd
from sklearn import model_selection
from typing import Tuple
from pathlib import Path
import os

from utils import get_logger

LOGGER = get_logger(__name__)

INPUT = os.environ.get("INPUT")
OUTPUT = os.environ.get("OUTPUT")


def load_training_data(path: str) -> pd.DataFrame:

    train = pd.read_csv(path)
    train['kfold'] = -1

    return train


def apply_stratified_kfold(train: pd.DataFrame, path: str) -> None:

    kf = model_selection.StratifiedKFold(n_splits=5,
                                         shuffle=True,
                                         random_state=123)
    for fold, (train_idx,
               val_idx) in enumerate(kf.split(X=train, y=train.target.values)):
        LOGGER.info(f'Train index: {len(train_idx)}, Val index: {val_idx}')
        train.loc[val_idx, 'kfold'] = fold

    LOGGER.info(f'Saving train folds to disk at {path}')
    train.to_csv(path)


def main():

    train = load_training_data(path=INPUT)
    apply_stratified_kfold(train=train, path=OUTPUT)


if __name__ == "__main__":
    main()
