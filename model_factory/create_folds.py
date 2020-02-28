import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn import model_selection

from datasets import QuoraDataSet, BengaliDataSetTrain
from utils import get_logger

LOGGER = get_logger(__name__)


def create_quora_folds(
    input_path: str = "inputs/quora_question_pairs/train.csv",
    output_path: str = "inputs/quora_question_pairs/train-folds.csv"
) -> None:
    dataset = QuoraDataSet()
    dataset.apply_stratified_kfold(input=input_path, output=output_path)
    LOGGER.info(dataset.train.shape)
    LOGGER.info(f'Target: {dataset.target}')
    LOGGER.info(f'Features: {dataset.feats}')
    LOGGER.info(dataset.train.head())


def create_bengali_folds(
        input_path: str = "inputs/bengali_grapheme/train.csv",
        output_path: str = "inputs/bengali_grapheme/train-folds.csv") -> None:
    dataset = BengaliDataSetTrain(train_path=input_path)
    dataset.apply_multilabel_stratified_kfold(input=input_path,
                                              output=output_path)
    LOGGER.info(dataset.train.shape)
    LOGGER.info(f'Target: {dataset.target}')
    LOGGER.info(f'Features: {dataset.feats}')
    LOGGER.info(dataset.train.head())