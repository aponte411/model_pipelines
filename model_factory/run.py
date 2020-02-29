import click

from engines import BengaliEngine
from trainers import BengaliTrainer
from models import ResNet34


@click.command()
@click.option('-d', '--data', type=str, default='bengali')
def runner(data: str) -> Optional:
    if data == 'bengali':
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
        bengali.run_engine()


if __name__ == "__main__":
    preds = runner()
