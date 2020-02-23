from datasets import BengaliDataSet

import utils

LOGGER = utils.get_logger(__name__)


def main():
    dataset = BengaliDataSet()
    dataset.pickle_images()


if __name__ == "__main__":
    main()