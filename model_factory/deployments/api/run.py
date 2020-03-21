from model_factory.engines import NumerAIEngine


def parse_args():
    pass


def main(args):
    engine = NumerAIEngine(args=args)
    engine.run_inference_engine()


if __name__ == "__main__":
    args = parse_args()
    main(args=args)