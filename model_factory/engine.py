import click.core

from model_factory.dataset import QuoraDataSet
from model_factory.trainers import QuoraTrainer


def preprocess_data(path: str, fold: int) -> Tuple[pd.DataFrame]:
    dataset = QuoraDataSet(path=path, fold=fold)
    train, val = dataset.prepare_data()
    y_train, y_val = dataset.get_targets()
    X_train, X_val = dataset.clean_data()
    return X_train, X_val, y_train, y_val


@click.command()
@click.option('-d',
              '--data-path',
              type=str,
              default="inputs/quora_question_pairs/train-folds.csv")
@click.option('-f', '--fold', type=int, default=0)
def runner(data_path: str, fold: int) -> pd.DataFrame:
    trainer = QuoraTrainer(tournament='quora_question_pairs',
                           name='base_trainer')
    X_train, X_val, y_train, y_val = preprocess_data(path=data_path, fold=fold)
    trainer.train_model(X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val)
    preds = trainer.predict_and_score(X_new=X_val, y_new=y_val)
    return preds


if __name__ == "__main__":
    preds = runner()
