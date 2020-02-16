import os
from typing import Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics, preprocessing

from . import dispatcher, utils

LOGGER = utils.get_logger(__name__)


def label_encode_all_features(train: pd.DataFrame, val: pd.DataFrame,
                              MODEL: str, FOLD: int) -> Tuple:

    label_encoders = {}
    for col in train.columns:
        LOGGER.info(f'Preprocessing column {col}')
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train[col].values.tolist() + val[col].values.tolist())
        train[col] = lbl.transform(train[col].values)
        val[col] = lbl.transform(val[col].values)
        label_encoders[col] = lbl

    joblib.dump(label_encoders, f'models/{MODEL}_{FOLD}_label_encoders.pkl')

    return train, val