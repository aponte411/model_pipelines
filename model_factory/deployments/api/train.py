import argparse
import types

from engines import NumerAIEngine


def parse_args() -> types.SimpleNamespace:
    parser = argparse.ArgumentParser(
        description='Train model for NumerAI tournament', )
    parser.add_argument('--training-config',
                        default='deployments/api/training_config.yml')
    parser.add_argument('--competition', default='numerai')
    return parser.parse_args()


def main(args: types.SimpleNamespace):
    engine = NumerAIEngine(args=args)
    engine.run_training_engine()


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
