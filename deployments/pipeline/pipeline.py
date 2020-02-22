import os
from datetime import datetime
from typing import Any, Tuple

import mlflow.sklearn
import numpy as np
import pandas as pd
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account
from sklearn.ensemble import RandomForestClassifier

import utils

LOGGER = utils.get_logger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH")
PROJECT_ID = os.environ.get("PROJECT_ID")
CREDS = os.environ.get("CREDS")
TABLE_ID = "pipeline_runs.user_scores"


def get_data() -> pd.DataFrame:
    return pd.read_csv(
        "https://raw.githubusercontent.com/bgweber/Twitch/master/Recommendations/games-expand.csv"
    )


def create_users(df: pd.DataFrame) -> pd.DataFrame:

    df['User_ID'] = df.index
    df['New_User'] = np.floor(np.random.randint(0, 10, df.shape[0]) / 9)

    return df


def split_data(df: pd.DataFrame) -> Tuple:

    train = df.loc[df['New_User'] == 0]
    x_train = train.iloc[:, 0:10]
    y_train = train['label']

    test = df.loc[df['New_User'] == 1]
    x_test = test.iloc[:, 0:10]

    return x_train, y_train, x_test, test


def train_model(X: pd.DataFrame, y: pd.DataFrame, model_path: str) -> Any:

    model = RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2)
    model.fit(X, y)
    mlflow.sklearn.save_model(model, model_path)

    return model


def predict(model: Any, X_test: pd.DataFrame) -> np.array:
    return model.predict_proba(X_test)[:, 1]


def format_preds(test: pd.DataFrame, preds: np.array) -> pd.DataFrame:

    results = pd.DataFrame({"User_ID": test["User_ID"], "Pred": preds})
    results["time"] = str(datetime.now())

    return results


def upload_to_gbq(results: pd.DataFrame, project_id: str, table_id: str,
                  file_name: str) -> None:
    def _setup_creds(file: str) -> Any:
        return service_account.Credentials.from_service_account_file(file)

    creds = _setup_creds(file=file_name)
    pandas_gbq.to_gbq(results,
                      table_id,
                      project_id=project_id,
                      if_exists='replace',
                      credentials=creds)


def query_preds(table_id: str) -> pd.DataFrame:

    client = bigquery.Client()
    sql = f"select * from {table_id}"

    return client.query(sql).to_dataframe()


def run():

    df = get_data()
    df = create_users(df=df)
    x_train, y_train, x_test, test = split_data(df=df)
    model = train_model(X=x_train, y=y_train, model_path=MODEL_PATH)
    preds = predict(model=model, X_test=x_test)
    results = format_preds(test=test, preds=preds)
    upload_to_gbq(results=results,
                  project_id=PROJECT_ID,
                  table_id=TABLE_ID,
                  file_name='pipeline-creds.json')
    LOGGER.info(
        f'Upload to Google Big Query @ table {TABLE_ID} under project {PROJECT_ID}'
    )

    return results


def evaluate():
    query_results = query_preds(table_id=TABLE_ID)
    LOGGER.info(query_results.head())


if __name__ == "__main__":
    results = run()
    evaluate()
