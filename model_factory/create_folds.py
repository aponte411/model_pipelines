import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn import model_selection

from datasets import QuoraDataSet, BengaliDataSet
from utils import get_logger

LOGGER = get_logger(__name__)


def create_quora_folds() -> None:
    dataset = QuoraDataSet()
    dataset.apply_stratified_kfold(
        input="inputs/quora_question_pairs/train.csv",
        output="inputs/quora_question_pairs/train-folds.csv")
    LOGGER.info(dataset.train.shape)
    LOGGER.info(f'Target: {dataset.target}')
    LOGGER.info(f'Features: {dataset.feats}')
    LOGGER.info(dataset.train.head())


def create_bengali_folds() -> None:
    dataset = BengaliDataSet()
    dataset.apply_multilabel_stratified_kfold(
        input="inputs/bengali_grapheme/train.csv",
        output="inputs/bengali_grapheme/train-folds.csv")
    LOGGER.info(dataset.train.shape)
    LOGGER.info(f'Target: {dataset.target}')
    LOGGER.info(f'Features: {dataset.feats}')
    LOGGER.info(dataset.train.head())


def main():
    create_bengali_folds()


if __name__ == "__main__":
    main()
