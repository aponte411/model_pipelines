import os
from typing import Any, Dict, List, Tuple

import click
import numpy as np

import models
import numerapi
import numerox as nx
import trainers
import utils

LOGGER = utils.get_logger(__name__)

TRAINING_DATA = os.environ.get("TRAINING_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")
TARGET = os.environ.get("TARGET")
MODEL_PATH = os.environ.get("MODEL_PATH")
DROP = ['is_duplicate', 'kfold']


def train_and_predict_xgboost_model(load_model: bool, save_model: bool,
                                    params: Dict) -> None:
    """Train/load model and conduct inference"""

    trainer = trainers.QuoraTrainer()
    train, val = trainer.prepare_data(TRAINING_DATA=TRAINING_DATA,
                                    FOLD=FOLD)
    y_train, y_val = trainer.get_targets(train=train, val=val, target=TARGET)
    X_train, X_val = trainer.clean_data(train=train, val=val)
    trainer.make_predictions_and_score(X_new=X_val, y_new=y_val)
    saved_model_name = f'xgboost_prediction_model_{trainer}'
    if load_model:
            trainer.load_from_s3(filename=saved_model_name,
                                 key=saved_model_name)
            trainer.train_model(X=X_train, y=y_train)
            predictions = trainer.predict(X=X_val)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions
        else:
            trainer.train_model(params=params)
            if save_model:
                trainer.save_model_locally(key=saved_model_name)
                trainer.save_to_s3(filename=saved_model_name,
                                   key=saved_model_name)
            predictions = trainer.make_predictions_and_prepare_submission(
                tournament=tournament_name, submit=submit_to_numerai)
            utils.evaluate_predictions(predictions=predictions,
                                       trainer=trainer,
                                       tournament=tournament_name)
            return predictions


@click.command()
@click.option('-m', '--model', type=str, default='xgboost')
@click.option('-lm', '--load-model', type=bool, default=True)
@click.option('-sm', '--save-model', type=bool, default=False)
@click.option('-s', '--submit', type=bool, default=False)
def main(model: str, load_model: bool, save_model: bool,
         submit: bool) -> nx.Prediction:

    if model == 'xgboost':
        XGBOOST_PARAMS = {
            "max_depth": 7,
            "learning_rate": 0.000123,
            "l2": 0.02,
            "n_estimators": 3000,
            "tree_method": "gpu_hist"
        }
        return train_and_predict_xgboost_model(load_model=load_model,
                                               save_model=save_model,
                                               submit_to_numerai=submit,
                                               params=XGBOOST_PARAMS)


if __name__ == "__main__":
    predictions = main()
    LOGGER.info(predictions.shape)
    LOGGER.info(predictions)
