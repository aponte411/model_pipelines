import os
from typing import Any, Dict, List, Tuple

import click
import numpy as np

import dataset
import models
import numerapi
import numerox as nx
import trainers
import utils

LOGGER = utils.get_logger(__name__)


def train_and_predict_quora_model(load_model: bool, save_model: bool) -> None:

    train, val = dataset.DataSet().prepare_data()
    y_train, y_val = dataset.DataSet().get_targets()
    X_train, X_val = dataset.DataSet().clean_data()
    trainer = trainers.QuoraTrainer()
    saved_model_name = f'{trainer}_prediction_model.p'
    if load_model:
        trainer.load_from_s3(filename=saved_model_name, key=saved_model_name)
        predictions = trainer.predict(X=X_val)
        return predictions
    else:
        trainer.train_model(X=X_train, y=y_train, X_val=X_val, y_val=X_val)
        if save_model:
            trainer.save_model_locally(key=saved_model_name)
            trainer.save_to_s3(filename=saved_model_name, key=saved_model_name)
        predictions = trainer.predict_and_score(X_new=X_val, y_new=y_val)
        return predictions


@click.command()
@click.option('-m', '--competition', type=str, default='quora')
@click.option('-lm', '--load-model', type=bool, default=True)
@click.option('-sm', '--save-model', type=bool, default=False)
def main(model: str, load_model: bool, save_model: bool,
         submit: bool) -> pd.DataFrame:

    if model == 'quora':
        return train_and_predict_quora_model(load_model=load_model,
                                             save_model=save_model)


if __name__ == "__main__":
    predictions = main()
    LOGGER.info(predictions)
