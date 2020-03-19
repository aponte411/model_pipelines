import os
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

import joblib
import nltk
import numpy as np
import pandas as pd
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn import metrics, preprocessing

import dispatcher
import utils

nltk.download('punkt')

LOGGER = utils.get_logger(__name__)


def label_encode_all_features(train: pd.DataFrame, val: pd.DataFrame) -> Tuple:
    for col in train.columns:
        LOGGER.info(f'Preprocessing column {col}')
        lbl = preprocessing.LabelEncoder()
        lbl.fit(train[col].values.tolist() + val[col].values.tolist())
        train[col] = lbl.transform(train[col].values)
        val[col] = lbl.transform(val[col].values)

    return train, val


class FeatureGenerator(ABC):
    def __init__(self, name: str):
        super().__init__()
        self.name = name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    @abstractmethod
    def create_features(self) -> Tuple:
        pass

    def save_features(self, X: pd.DataFrame) -> None:
        X.to_csv(f"X_{self.name}.csv", index=False)


class QuoraFeatureGenerator(FeatureGenerator):
    def __init__(self, name: str):
        super().__init__(name)

    def create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        def _preprocess(X: pd.DataFrame):
            X.apply(self.remove_spaces, axis=1)
            X.apply(self.tokenize_words, axis=1)
            X.apply(self.stemm_words, axis=1)

        return _preprocess(X=X)

    def remove_spaces(self, text: str) -> str:
        processed_string = text.strip().split()
        return " ".join(processed_string)

    def tokenize_words(self, text: str) -> List[str]:
        return word_tokenize(text)

    def stemm_words(self, text: str) -> str:
        stemmer = SnowballStemmer('english')
        return stemmer.stem(text)

    def create_frequency_feats(self, df: pd.DataFrame, feat: str,
                               stat: str) -> pd.DataFrame:
        return df.groupby(feat)[feat].transform(stat)

    def create_length_feats(self, df: pd.DataFrame, feat: str) -> pd.DataFrame:
        return df[feat].str.len()

    def create_n_words_feats(self, df: pd.DataFrame,
                             feat: str) -> pd.DataFrame:
        return df[feat].apply(lambda x: len(x.split()))

    def create_quora_total_words_feats(self, df: pd.DataFrame) -> pd.DataFrame:
        def _normalize(row: pd.Series) -> float:
            word1 = set(
                map(lambda x: x.lower().strip(), row['question1'].split()))
            word2 = set(
                map(lambda x: x.lower().strip(), row['question2'].split()))
            return 1.0 * (len(word1) + len(word2))

        return df.apply(_normalize, axis=1)

    def create_quora_common_words_feats(self,
                                        df: pd.DataFrame) -> pd.DataFrame:
        def _normalize(row: pd.Series) -> float:
            word1 = set(
                map(lambda x: x.lower().strip(), row['question1'].split()))
            word2 = set(
                map(lambda x: x.lower().strip(), row['question2'].split()))
            return 1.0 * len(word1 & word2)

        return df.apply(_normalize, axis=1)

    def create_quora_shared_words_feats(self,
                                        df: pd.DataFrame) -> pd.DataFrame:
        def _normalize(row: pd.Series) -> float:
            word1 = set(
                map(lambda x: x.lower().strip(), row['question1'].split()))
            word2 = set(
                map(lambda x: x.lower().strip(), row['question2'].split()))
            return 1.0 * len(word1 & word2) / (len(word1) + len(word2))

        return df.apply(_normalize, axis=1)
