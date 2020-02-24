import click.core

from datasets import QuoraDataSet
from trainers import QuoraTrainer


def preprocess_data(path: str, fold: int) -> Tuple[pd.DataFrame]:
    dataset = QuoraDataSet(path=path, target='is_duplicate')
    train, val = dataset.prepare_data(fold=fold)
    y_train, y_val = dataset.get_targets()
    X_train, X_val = dataset.clean_data(to_drop=['is_duplicate', 'kfold'])
    return X_train, X_val, y_train, y_val


@click.command()
@click.option('-d',
              '--data-path',
              type=str,
              default="inputs/quora_question_pairs/train-folds.csv")
@click.option('-f', '--fold', type=int, default=0)
def runner(data_path: str, fold: int) -> pd.DataFrame:
    XGBOOST_PARAMS = {
        "max_depth": 7,
        "learning_rate": 0.000123,
        "l2": 0.02,
        "n_estimators": 3000,
        "tree_method": "gpu_hist"
    }
    trainer = QuoraTrainer(params=XGBOOST_PARAMS)
    X_train, X_val, y_train, y_val = preprocess_data(path=data_path, fold=fold)
    trainer.train_model(X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val)
    preds = trainer.predict_and_score(X_new=X_val, y_new=y_val)
    return preds


if __name__ == "__main__":
    preds = runner()
