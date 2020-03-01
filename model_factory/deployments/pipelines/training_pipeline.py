import utils
from engines import BengaliEngine
from models import ResNet34
from trainers import BengaliTrainer

LOGGER = utils.get_logger(__name__)


def runner(data: str) -> Optional:
    PARAMS = {
        "image_height": 137,
        "image_width": 236,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.239, 0.225),
        "train_folds": [0, 1, 2, 3],
        "val_folds": [4]
    }
    model = ResNet34(pretrained=True)
    trainer = BengaliTrainer(model=model)
    bengali = BengaliEngine(name='bengali-engine',
                            trainer=trainer,
                            params=PARAMS)
    LOGGER.info(f"Training model..")
    bengali.run_engine(load=False, save=True)
    LOGGER.info(f"Saving to s3 bucket")


if __name__ == "__main__":
    runner()
