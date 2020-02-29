import pytest

import trainers
import utils


@pytest.fixture
def params():
    return {
        "image_height": 10,
        "image_width": 10,
        "mean": (0.2, 0.2, 0.2),
        "std": (0.1, 0.1, 0.1)
    }


def test_attributes(params):
    test_trainer = trainers.BengaliTrainer(model_name='resnet34',
                                           params=params)
    assert hasattr(test_trainer, "train")
    assert hasattr(test_trainer, "_loss_fn")
    assert hasattr(test_trainer, "criterion")
    assert hasattr(test_trainer, "optimizer")
    assert hasattr(test_trainer, "scheduler")
    assert hasattr(test_trainer, "early_stopping")
    assert hasattr(test_trainer, "model")
    assert hasattr(test_trainer, "load_model_locally")
    assert hasattr(test_trainer, "load_model_from_s3")
    assert hasattr(test_trainer, "save_model_locally")
    assert hasattr(test_trainer, "save_model_to_s3")
