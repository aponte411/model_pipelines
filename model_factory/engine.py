from typing import Dict, List, Optional, Tuple

import click
import torch
from torch.utils.data import DataLoader

from datasets import BengaliDataSetTrain, QuoraDataSet
from trainers import BengaliTrainer, QuoraTrainer

LOGGER = utils.get_logger(__name__)


def run_quora_engine(params: Dict, data_path: str, fold: int):
    def preprocess_data(path: str, fold: int) -> Tuple[pd.DataFrame]:
        dataset = QuoraDataSet(path=path, target='is_duplicate')
        train, val = dataset.prepare_data(fold=fold)
        y_train, y_val = dataset.get_targets()
        X_train, X_val = dataset.clean_data(to_drop=['is_duplicate', 'kfold'])
        return X_train, X_val, y_train, y_val

    trainer = QuoraTrainer(params=params)
    X_train, X_val, y_train, y_val = preprocess_data(path=data_path, fold=fold)
    trainer.train_model(X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val)
    return trainer.predict_and_score(X_new=X_val, y_new=y_val)


def run_bengali_engine(training_data: str, epochs: int, params: Dict) -> None:
    def _get_loader(train_path: str, folds: List[int],
                    params: Dict) -> DataLoader:
        dataset = BengaliDataSetTrain(train_path=train_path,
                                      folds=folds,
                                      image_height=params["image_height"],
                                      image_width=params["image_width"],
                                      mean=params["mean"],
                                      std=params["std"])
        return DataLoader(dataset=dataset,
                          batch_size=params["batch_size"],
                          shuffle=True,
                          num_workers=4)

    train = _get_loader(train_path=training_data,
                        folds=[0, 1, 2, 3],
                        params=params)
    val = _get_loader(train_path=training_data, folds=[4], params=params)
    trainer = BengaliTrainer(model_name='resnet')
    for epoch in range(epochs):
        train_loss, train_score = trainer.train(train)
        val_loss, val_score = trainer.evaluate(val)
        trainer.scheduler.step(val_loss)
        trainer.early_stopping(val_score, trainer.model)
        if trainer.early_stopping.early_stop:
            LOGGER.info("Early stopping")
            break


@click.command()
@click.option('-c', '--competition', type=str, default='bengali')
@click.option('-f', '--fold', type=int, default=0)
def runner(competition: str, fold: int) -> Optional:
    if competition == 'quora':
        XGBOOST_PARAMS = {
            "max_depth": 7,
            "learning_rate": 0.000123,
            "l2": 0.02,
            "n_estimators": 3000,
            "tree_method": "gpu_hist"
        }
        return run_quora_engine(params=XGBOOST_PARAMS, fold=fold)
    if competition == 'bengali':
        PARAMS = {
            "image_height": 137,
            "image_width": 236,
            "mean": (0.485, 0.456, 0.406),
            "std": (0.229, 0.239, 0.225)
        }
        run_bengali_engine(
            epochs=10,
            params=PARAMS,
            training_data="inputs/bengali_grapheme/train-folds.csv")


if __name__ == "__main__":
    preds = runner()
