import trainers
import utils
from models import ResNet34
from trainers import Trainer


def test_attributes():
    model = ResNet34(pretrained=False)
    test_trainer = trainers.BengaliTrainer(model=model)
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
    assert isinstance(test_trainer, Trainer)
