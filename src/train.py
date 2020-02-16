import os
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing

from . import dispatcher, utils

LOGGER = utils.get_logger(__name__)

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
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
    df = pd.read_csv(TRAINING_DATA)
    train = df.loc[df.kfold.isin(
        FOLD_MAPPING.get(FOLD))].reset_index(drop=True)
    valid = df.loc[df.kfold == FOLD]
    del df

    return train, valid


def get_targets(train: pd.DataFrame, val: pd.DataFrame) -> Tuple:

    y_train = train.target.values
    y_val = val.target.values

    return y_train, y_val


def clean_data(train: pd.DataFrame, val: pd.DataFrame) -> Tuple:

    train.drop(['id', 'target', 'kfold'], axis=1, inplace=True)
    val.drop(['id', 'target', 'kfold'], axis=1, inplace=True)
    val = val[train.columns]
    LOGGER.info(f'Train: {train.shape}')
    LOGGER.info(f'Val: {val.shape}')

    return train, val


def label_encode_all_data(train: pd.DataFrame, val: pd.DataFrame) -> Tuple:

    label_encoders = {}
    for col in train.columns:
        LOGGER.info('Preprocessing column {col}')
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train[col].values)
        train[col] = lbl.transform(train[col].values)
        val[col] = lbl.transform(val[col].values)
        label_encoders[col] = lbl

    joblib.dump(label_encoders, f'models/{MODEL}_{FOLD}_label_encoders.pkl')

    return train, valid


def train_model(X_train: np.array, y_train: np.array) -> Any:
    try:
        LOGGER.info(f'Training {MODEL}..')
        model = dispatcher.MODELS[MODEL]
        model.fit(X_train, y_train)
        joblib.dump(model, f'models/{MODEL}_{FOLD}_trained.pkl')
        LOGGER.info(f'Training complete!')
        return model
    except Exception as e:
        LOGGER.error(f'Failed to train {MODEL} with {e}')
        raise e


def make_predictions_and_score(model: Any, X_val: np.array,
                               y_val: np.array) -> None:

    LOGGER.info(f'Making predictions and scoring the model...')
    preds = model.predict_proba(X_val)[:, 1]
    LOGGER.info(f'ROC AUC SCORE: {metrics.roc_auc_score(y_val, preds)}')


def main():

    train, val = prepare_data(TRAINING_DATA, FOLD, FOLD_MAPPING)
    y_train, y_val = get_targets(train, val)
    X_train, X_val = clean_data(train, val)
    X_train, X_val = label_encode_all_data(X_train, X_val)
    model = train_model(X_train, y_train)
    make_predictions_and_score(model, X_val, y_val)


if __name__ == "__main__":
    main()
