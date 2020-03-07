import click
import utils

LOGGER = utils.get_logger(__name__)


@click.command()
@click.option('-in',
              '--input',
              type=str,
              default="inputs/bengali_grapheme/train_*.parquet")
@click.option('-ou',
              '--output',
              type=str,
              default="inputs/bengali_grapheme/image_pickles")
def main(input: str, output: str):
    utils.pickle_images(input=input, output_dir=output)


if __name__ == "__main__":
    main()