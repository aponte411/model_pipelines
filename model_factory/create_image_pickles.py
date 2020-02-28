from datasets import BengaliDataSetTrain

import utils

LOGGER = utils.get_logger(__name__)


def pickle_bengali_images(
    train_path: str = "inputs/bengali_grapheme/train-folds.csv",
    parquet_path: str = "inputs/bengali_grapheme/train_*.parquet"):
    dataset = BengaliDataSetTrain(train_path=train_path)
    dataset.pickle_images(input=parquet_path)


if __name__ == "__main__":
    pickle_bengali_images()