import click
from typing import List, Tuple, Dict, Optional

from datasets import QuoraDataSet, BengaliDataSetTrain
from trainers import QuoraTrainer, BengaliTrainer


def run_quora_model(params: Dict, data_path: str, fold: int):
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


def run_bengali_model(fold: int):
    dataset = BengaliDataSetTrain(folds=[0, 1],
                                  image_height=137,
                                  image_width=236,
                                  mean=(0.485, 0.456, 0.406),
                                  std=(0.229, 0.224, 0.225))
    trainer = BengaliTrainer(model_name='resnet')
    print(len(dataset))
    # WIP


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
        return run_quora_model(params=XGBOOST_PARAMS, fold=fold)
    if competition == 'bengali':
        run_bengali_model(fold=fold)


if __name__ == "__main__":
    preds = runner()
