from datasets import BengaliDataSetTrain
import click
import utils

LOGGER = utils.get_logger(__name__)


@click.command()
@click.option('-in',
              '--input',
              type=str,
              default="inputs/bengali_grapheme/train-folds.csv")
@click.option('-ou',
              '--output',
              type=str,
              default="inputs/bengali_grapheme/train_*.parquet")
def main(input: str, output: str):
    utils.pickle_images(train_path=input, parquet_path=output)


if __name__ == "__main__":
    main()