import glob
import os
from typing import Any, Dict, List, Tuple

import albumentations
import joblib
import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image
from sklearn import model_selection
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import cross_validators
import utils

LOGGER = utils.get_logger(__name__)


class BengaliDataSetTrain(Dataset):
    """
    Dataset for training/validation.

    Args:
        train_path {str} - path to train-folds.csv
        pickle_path {str} - path to pickled images
        folds {List[int]} - key for dataset fold
        image_height {int} - height of images
        image_width {int} - width of images
        mean {Tuple[float]} - mean for image augmentation
        std {Tuple[float]} - variance for image augmentation
    """
    def __init__(self,
                 train_path: str,
                 pickle_path: str = None,
                 folds: List[int] = [0],
                 image_height: int = None,
                 image_width: int = None,
                 mean: Tuple[float] = None,
                 std: Tuple[float] = None):
        self.train_path = train_path
        self.folds = folds
        self.image_height = image_height
        self.image_width = image_width
        self.mean = mean
        self.std = std
        self.pickle_path = pickle_path
        self._create_attributes()
        self._create_augmentations()

    def _create_attributes(self) -> None:
        def _load_df() -> pd.DataFrame:
            df = pd.read_csv(self.train_path)
            df = df.drop('grapheme', axis=1)
            return df.loc[df.kfold.isin(self.folds)].reset_index(drop=True)

        df = _load_df()
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values

    def _create_augmentations(self) -> None:
        if len(self.folds) > 1:
            self.aug = albumentations.Compose([
                albumentations.Resize(self.image_height,
                                      self.image_width,
                                      always_apply=True),
                albumentations.Normalize(self.mean,
                                         self.std,
                                         always_apply=True)
            ])
        else:
            self.aug = albumentations.Compose([
                albumentations.Resize(self.image_height,
                                      self.image_width,
                                      always_apply=True),
                albumentations.ShiftScaleRotate(shift_limit=0.0625,
                                                scale_limit=0.1,
                                                rotate_limit=5,
                                                p=0.9),
                albumentations.Normalize(self.mean,
                                         self.std,
                                         always_apply=True)
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, item: int) -> Dict:
        def _prepare_image() -> Image:
            image = joblib.load(
                f"{self.pickle_path}/{self.image_ids[item]}.pkl")
            image = image.reshape(self.image_height,
                                  self.image_width).astype(float)
            return Image.fromarray(image).convert("RGB")

        def _augment_image(image) -> np.array:
            image = self.aug(image=np.array(image))["image"]
            return np.transpose(image, (2, 0, 1)).astype(np.float32)

        def _return_image_dict(image) -> Dict:
            return {
                "image":
                torch.tensor(image, dtype=torch.float),
                "grapheme_root":
                torch.tensor(self.grapheme_root[item], dtype=torch.long),
                "vowel_diacritic":
                torch.tensor(self.vowel_diacritic[item], dtype=torch.long),
                "consonant_diacritic":
                torch.tensor(self.consonant_diacritic[item], dtype=torch.long),
            }

        image = _prepare_image()
        image = _augment_image(image=image)
        return _return_image_dict(image=image)


class BengaliDataSetTest(Dataset):
    """
    Dataset for inference.

    Args:
        df {pd.DataFrame} - parquet dataframe with test images
        image_height {int} - height of images
        image_width {int} - width of images
        mean {Tuple[float]} - mean for image augmentation
        std {Tuple[float]} - variance for image augmentation
    """
    def __init__(self,
                 df: pd.DataFrame,
                 image_height: int = None,
                 image_width: int = None,
                 mean: float = None,
                 std: float = None):
        self.df = df
        self.image_height = image_height
        self.image_width = image_width
        self.mean = mean
        self.std = std
        self._create_attributes()
        self._create_augmentations()

    def _create_attributes(self) -> None:
        self.image_id = self.df.image_id.values
        self.image_arr = self.df.iloc[:, 1:].values

    def _create_augmentations(self) -> None:
        self.aug = albumentations.Compose([
            albumentations.Resize(self.image_height,
                                  self.image_width,
                                  always_apply=True),
            albumentations.Normalize(self.mean, self.std, always_apply=True)
        ])

    def __len__(self):
        return len(self.image_id)

    def __getitem__(self, item: int) -> Dict:
        def _prepare_image() -> Image:
            image = self.image_arr[item, :]
            image = image.reshape(self.image_height,
                                  self.image_width).astype(float)
            return Image.fromarray(image).convert("RGB")

        def _augment_image(image) -> np.array:
            image = self.aug(image=np.array(image))["image"]
            return np.transpose(image, (2, 0, 1)).astype(np.float32)

        def _get_image_id() -> int:
            return self.image_id[item]

        def _return_image_dict(image, image_id) -> Dict:
            return {
                "image": torch.tensor(image, dtype=torch.float),
                "image_id": image_id
            }

        image = _prepare_image()
        augmented_image = _augment_image(image=image)
        image_id = _get_image_id()
        return _return_image_dict(image=augmented_image, image_id=image_id)


class GoogleQADataSetTrain(Dataset):
    """
    Requires you to download kaggle dataset using:
        kaggle competitions download -c google-quest-challenge
        unzip google-quest-challenge.zip

    Args:
        Dataset {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    def __init__(self, data_folder: str, folds: List[int], tokenizer: Any,
                 max_len: int):
        super().__init__()
        self.data_folder = data_folder
        self.folds = folds
        self.tokenizer = tokenizer
        self.max_len = max_len
        self._create_attributes()

    def _get_targets(self):
        return cross_validators.GoogleQACrossValidator.get_targets(
            f'{self.data_folder}/sample_submission.csv')

    def _get_features(self):
        df = pd.read_csv(f'{self.data_folder}/train-folds.csv')
        return df.loc[df.kfold.isin(self.folds)].reset_index(drop=True)

    def _create_attributes(self) -> None:
        test_columns = self._get_targets()
        df = self._get_features()
        self.question_title = df.question_title.values
        self.question_body = df.question_body.values
        self.answer = df.answer.values
        self.targets = df[test_columns].values

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, item):
        def _stringify(array: np.array) -> str:
            return str(array)

        def _encode_strings(string1: str, string2: str, string3: str) -> Tuple:
            inputs = self.tokenizer.encode_plus(string1 + string2,
                                                string3,
                                                add_special_tokens=True,
                                                max_len=self.max_len)
            ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            mask = inputs['attention_mask']
            return ids, token_type_ids, mask

        def _add_padding(array: np.array, len: int) -> np.array:
            return array + ([0] * len)

        question_title = _stringify(array=self.question_title[item])
        question_body = _stringify(array=self.question_body[item])
        answer = _stringify(array=self.answer[item])
        ids, token_type_ids, mask = _encode_strings(string1=question_title,
                                                    string2=question_title,
                                                    string3=answer)

        padding = self.max_len - len(ids)
        ids = _add_padding(ids, padding)
        token_type_ids = _add_padding(token_type_ids, padding)
        mask = _add_padding(mask, padding)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "targets": torch.tensor(self.targets[item, :], dtype=torch.float)
        }


class GoogleQADataSetTest(Dataset):
    def __init__(self, question_title, question_body, answer, tokenizer,
                 max_len):
        super().__init__()
        self.question_title = question_title
        self.question_body = question_body
        self.answer = answer
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.answer)

    def __getitem__(self, item):
        def _stringify(array: np.array) -> str:
            return str(array)

        def _encode_strings(string1: str, string2: str, string3: str) -> Tuple:
            inputs = self.tokenizer.encode_plus(string1 + string2,
                                                string3,
                                                add_special_tokens=True,
                                                max_len=self.max_len)
            ids = inputs['input_ids']
            token_type_ids = inputs['token_type_ids']
            mask = inputs['attention_mask']
            return ids, token_type_ids, mask

        def _add_padding(array: np.array, len: int) -> np.array:
            return array + ([0] * len)

        question_title = _stringify(self.question_title[item])
        question_body = _stringify(self.question_body[item])
        answer = _stringify(self.answer[item])
        ids, token_type_ids, mask = _encode_strings(question_title +
                                                    question_title,
                                                    answer,
                                                    add_special_tokens=True,
                                                    max_len=self.max_len)

        padding = self.max_len - len(ids)
        ids = _add_padding(ids, padding)
        token_type_ids = _add_padding(token_type_ids, padding)
        mask = _add_padding(mask, padding)

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long)
        }