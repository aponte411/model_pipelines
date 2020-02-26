from datasets import BengaliDataSetTrain

import utils

LOGGER = utils.get_logger(__name__)


def main():
    dataset = BengaliDataSetTrain()
    dataset.pickle_images()


if __name__ == "__main__":
    main()