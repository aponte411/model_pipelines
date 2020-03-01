from datasets import BengaliDataSetTrain

import utils

LOGGER = utils.get_logger(__name__)


def main():
    utils.pickle_images(train_path="inputs/bengali_grapheme/train-folds.csv",
                        parquet_path="inputs/bengali_grapheme/train_*.parquet")


if __name__ == "__main__":
    main()