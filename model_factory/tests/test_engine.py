import pytest

import engines
import models
import trainers
import datasets


@pytest.fixture
def params():
    return {
        "image_height": 137,
        "image_width": 236,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.239, 0.225),
        "train_folds": [0],
        "val_folds": [4]
    }


def test_attributes(params):
    model = models.ResNet34(pretrained=False)
    trainer = trainers.BengaliTrainer(model=model)
    bengali = engines.BengaliEngine(name='bengali-engine',
                                    trainer=trainer,
                                    params=params)
    assert isinstance(bengali.trainer, trainers.Trainer)
    assert issubclass(bengali.training_set, datasets.BengaliDataSetTrain)
    assert issubclass(bengali.val_set, datasets.BengaliDataSetTrain)
    assert issubclass(bengali.test_set, datasets.BengaliDataSetTest)