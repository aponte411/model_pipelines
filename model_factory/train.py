import os
from typing import Any, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing

import dispatcher
import utils

LOGGER = utils.get_logger(__name__)

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
TARGET = os.environ.get("TARGET")
MODEL_PATH = os.environ.get("MODEL_PATH")
DROP = ['label', 'kfold']
FOLD_MAPPING = {
    0: [1, 2, 3, 4],
    1: [0, 2, 3, 4],
    2: [0, 1, 3, 4],
    3: [0, 1, 2, 4],
    4: [0, 1, 2, 3],
}


def prepare_data(TRAINING_DATA: str, FOLD: str, FOLD_MAPPING: str) -> Tuple:

    LOGGER.info(f'Loading training data from: {TRAINING_DATA}')
    LOGGER.info(f'Fold: {FOLD}')
    df = pd.read_csv(TRAINING_DATA, index_col=False)
    train = df.loc[df.kfold.isin(
        FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid = df.loc[df.kfold == FOLD]
    del df

    return train, valid


def get_targets(train: pd.DataFrame, val: pd.DataFrame, target: str) -> Tuple:

    y_train = train[target].values
    y_val = val[target].values

    return y_train, y_val


def train_model(X: np.array, y: np.array) -> Any:
    try:
        LOGGER.info(f'Training {MODEL}..')
        model = dispatcher.MODELS[MODEL]
        model.fit(X, y)
        mlflow.sklearn.save_model(model, MODEL_PATH)
        LOGGER.info(f'Training complete!')
        return model
    except Exception as e:
        LOGGER.error(f'Failed to train {MODEL} with {e}')
        raise e


def make_predictions_and_score(model: Any, X_new: pd.DataFrame,
                               y_new: pd.DataFrame) -> None:

    LOGGER.info(f'Making predictions and scoring the model...')
    preds = model.predict_proba(X_new)[:, 1]
    LOGGER.info(f'ROC-AUC-SCORE: {metrics.roc_auc_score(y_new, preds)}')


def main():

    train, val = prepare_data(TRAINING_DATA=TRAINING_DATA,
                              FOLD=FOLD,
                              FOLD_MAPPING=FOLD_MAPPING)
    y_train, y_val = get_targets(train=train, val=val, target=TARGET)
    X_train, X_val = utils.clean_data(train=train, val=val, to_drop=DROP)
    model = train_model(X=X_train, y=y_train)
    make_predictions_and_score(model=model, X_new=X_val, y_new=y_val)


if __name__ == "__main__":
    main()
