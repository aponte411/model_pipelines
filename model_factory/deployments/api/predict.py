from model_factory.engines import NumerAIEngine


def parse_args() -> types.SimpleNamespace:
    parser = argparse.ArgumentParser(
        description='Make predictions for NumeraAI tournament', )
    parser.add_argument('--training-config',
                        default='deployments/api/training_config.yml')
    parser.add_argument('--competition', default='numerai')
    parser.add_argument('--submit-to-numerai', default=True)
    return parser.parse_args()


def main(args):
    engine = NumerAIEngine(args=args)
    engine.run_inference_engine()


if __name__ == "__main__":
    args = parse_args()
    main(args=args)