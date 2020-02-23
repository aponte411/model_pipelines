import glob
from typing import Any, List, Tuple

import albumentations
import joblib
import numpy as np
import pandas as pd
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from PIL import Image
from sklearn import model_selection
from tqdm import tqdm

import utils

LOGGER = utils.get_logger(__name__)


class DataSet:
    def __init__(
        self,
        path: str,
        target: Any,
    ):
        self.path = path
        self.target = target
        self.feats = None
        self.fold_mapping = {
            0: [1, 2, 3, 4],
            1: [0, 2, 3, 4],
            2: [0, 1, 3, 4],
            3: [0, 1, 2, 4],
            4: [0, 1, 2, 3],
        }
        self.train = None
        self.valid = None
        self.y_train = None
        self.y_val = None

    def _load_train_for_cv(self, input_path: str) -> pd.DataFrame:
        train = pd.read_csv(input_path)
        train['kfold'] = -1
        return train

    def prepare_data(self, fold: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        LOGGER.info(f'Loading training data from: {self.path}')
        LOGGER.info(f'Fold: {fold}')
        df = pd.read_csv(self.path)
        self.train = df.loc[df.kfold.isin(
            self.fold_mapping.get(fold))].reset_index(drop=True)
        self.valid = df.loc[df.kfold == fold]
        del df

        return self.train, self.valid

    def get_targets(self) -> Tuple:
        self.y_train = self.train[self.target].values
        self.y_val = self.valid[self.target].values
        return self.y_train, self.y_val

    def clean_data(self,
                   to_drop: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self.train = self.train.drop(to_drop, axis=1).reset_index(drop=True)
        self.valid = self.valid.drop(to_drop, axis=1).reset_index(drop=True)
        self.valid = self.valid[self.train.columns]
        LOGGER.info(f'Train: {self.train.shape}')
        LOGGER.info(f'Val: {self.valid.shape}')
        return self.train, self.valid


class QuoraDataSet(DataSet):
    def __init__(self,
                 path="inputs/quora_question_pairs/train-folds.csv",
                 target="is_duplicate"):
        super().__init__(path=path, target=target)
        self.path = path
        self.target = target

    def apply_stratified_kfold(self, input: str, output: str) -> None:
        train = self._load_train_for_cv(input_path=input)
        kf = model_selection.StratifiedKFold(n_splits=5,
                                             shuffle=True,
                                             random_state=123)
        for fold, (train_idx, val_idx) in enumerate(
                kf.split(X=train, y=train[self.target].values)):
            LOGGER.info(f'Train index: {len(train_idx)}, Val index: {val_idx}')
            train.loc[val_idx, 'kfold'] = fold

        LOGGER.info(f'Saving train folds to disk at {output}')
        train.to_csv(output, index=False)
        self.train = train
        self.feats = [
            col for col in train.columns
            if col != self.target and col != 'kfold'
        ]


class BengaliDataSet(DataSet):
    def __init__(self,
                 path: str = "inputs/bengali_grapheme/train-folds.csv",
                 target: List[str] = [
                     "grapheme_root", "vowel_diacritic", "consonant_diacritic"
                 ],
                 folds: int = None,
                 image_height: int = None,
                 image_width: int = None,
                 mean: float = None,
                 std: float = None):
        super().__init__(path=path, target=target)
        self.folds = folds
        self.image_height = image_height
        self.image_width = image_width
        self.mean = mean
        self.std = std
        self._create_attributes()
        self._create_augmentations()

    def _create_attributes(self) -> None:
        df = pd.read_csv(self.path)
        df = df.drop('grapheme', axis=1)
        df = df.loc[df.kfold.isin(self.folds)].reset_index(drop=True)
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values

    def _create_augmentations(self) -> None:
        if len(self.folds) > 1:
            self.aug = albumentations.compose([
                albumentations.Resize(self.image_height,
                                      self.image_width,
                                      always_apply=True),
                albumentations.Normalize(self.mean,
                                         self.std,
                                         always_apply=True)
            ])
        else:
            self.aug = albumentations.compose([
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
                f"inputs/bengali_grapheme/pickled_images/{self.image_ids[item]}.p"
            )
            image = image.reshape(137, 236).astype(float)
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

    def _split_data(self,
                    train: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = train.image_id.values
        y = train[self.target].values
        return X, y

    def apply_multilabel_stratified_kfold(self, input: str,
                                          output: str) -> None:
        train = self._load_train_for_cv(input_path=input)
        X, y = self._split_data(train=train)
        mskf = MultilabelStratifiedKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
            LOGGER.info(f'Train index: {len(train_idx)}, Val index: {val_idx}')
            train.loc[val_idx, 'kfold'] = fold

        LOGGER.info(f'Saving train folds to disk at {output}')
        train.to_csv(output, index=False)
        self.train = train
        self.feats = [
            col for col in train.columns
            if col not in self.target and col != 'kfold'
        ]

    def pickle_images(self,
                      input: str = "inputs/bengali_grapheme/train_*.parquet"):
        for file in glob.glob(input):
            df = pd.read_parquet(file)
            image_ids = df.image_id.values
            image_array = df.drop('image_id', axis=1).values
            for idx, image_id in tqdm(enumerate(image_ids),
                                      total=len(image_ids)):
                joblib.dump(
                    image_array[idx, :],
                    f"inputs/bengali_grapheme/pickled_images/{image_id}.p")
