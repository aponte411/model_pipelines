import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
import os
from typing import Tuple, Any
import joblib

from . import utils
from . import dispatcher

LOGGER = utils.get_logger(__name__)

# TRAINING_DATA = r"\Users\apont\KAGGLE_COMPETITIONS\ml-project-template\inputs\categorical_challenge\train_folds.csv"
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


def label_encode_all_data(train: pd.DataFrame, valid: pd.DataFrame) -> Tuple:

    for col in train.columns:
        LOGGER.info('Preprocessing data..')
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train[col])
        train[col] = lbl.transform(train[col])
        valid[col] = lbl.transform(valid[col])

    LOGGER.info(f'Train shape: {train.shape}')
    LOGGER.info(f'Val shape: {valid.shape}')

    return train, valid


def train_model(X_train, y_train) -> Any:

    LOGGER.info(f'Training {MODEL}..')
    model = dispatcher.MODELS[MODEL]
    model.fit(X_train, y_train)
    joblib.dump(model, f'{MODEL}_trained')
    LOGGER.info(f'Training complete!')

    return model


def make_predictions_and_score(model: Any, X_val: np.array,
                               y_val: np.array) -> None:

    LOGGER.info(f'Making predictions and scoring the model...')
    preds = model.predict_proba(X_val)[:, 1]
    LOGGER.info(f'ROC_AUC_SCORE: {metrics.roc_auc_score(y_val, preds)}')


def main():

    train, val = prepare_data(TRAINING_DATA, FOLD, FOLD_MAPPING)
    y_train, y_val = get_targets(train, val)
    X_train, X_val = clean_data(train, val)
    X_train, X_val = label_encode_all_data(X_train, X_val)
    model = train_model(X_train, y_train)
    make_predictions_and_score(model, X_val, y_val)


if __name__ == "__main__":
    main()
