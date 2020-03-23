import argparse
import types

import utils
from engines import NumerAIEngine

LOGGER = utils.get_logger(__name__)


def parse_args() -> types.SimpleNamespace:
    parser = argparse.ArgumentParser(
        description='Make predictions for NumeraAI tournament', )
    parser.add_argument('--training-config',
                        default='deployments/api/training_config.yml')
    parser.add_argument('--competition', default='numerai')
    parser.add_argument('--submit', default=False)
    return parser.parse_args()


def main(args: types.SimpleNamespace):
    engine = NumerAIEngine(args=args)
    predictions = engine.run_inference_engine()
    for tournament, prediction in predictions.items():
        LOGGER.info(prediction.df.shape)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
