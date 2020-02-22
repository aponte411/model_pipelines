import utils

from model_factory.trainers import QuoraTrainer
from model_factory.dataset import DataSet

LOGGER = utils.get_logger(__name__)


def train_model(model_name: str, path: str, fold: int) -> Any:
    trainer = QuoraTrainer()
    dataset = DataSet(path=path, fold=fold)
    train, val = dataset.prepare_data()
    y_train, y_val = dataset.get_targets()
    X_train, X_val = dataset.clean_data()
    LOGGER.info(f"Training {model_name}..")
    trainer.train_model(X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val)
    LOGGER.info(f"Saving to s3 bucket")
    trainer.save_to_s3(filename=model_name, key=model_name)
