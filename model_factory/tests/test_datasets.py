import pandas as pd
import numpy as np
import pytest
import torch

import datasets


@pytest.fixture
def bengali_dataset():
    return datasets.BengaliDataSetTrain(
        train_path="inputs/bengali_grapheme/train-folds.csv",
        folds=[0],
        image_height=137,
        image_width=236,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.239, 0.225))


def test_data_info(bengali_dataset):
    EXPECTED_NUM_TRAIN_IMGS = 40168
    EXPECTED_NUM_GRPHME_RTS = 168
    EXPECTED_NUM_VOWELS = 11
    EXPECTED_NUM_CONS = 7
    EXPECTED_DTYPE = torch.tensor
    assert len(bengali_dataset) == EXPECTED_NUM_TRAIN_IMGS
    assert pd.Series(
        bengali_dataset.grapheme_root).nunique() == EXPECTED_NUM_GRPHME_RTS
    assert pd.Series(
        bengali_dataset.vowel_diacritic).nunique() == EXPECTED_NUM_VOWELS
    assert pd.Series(
        bengali_dataset.consonant_diacritic).nunique() == EXPECTED_NUM_CONS
    assert isinstance(bengali_dataset[np.random.randint(100)]['grapheme_root'],
                      torch.Tensor)
