import os
from typing import Any, Dict

import flask
import mlflow
import mlflow.sklearn
import pandas as pd

MODEL = os.environ.get("MODEL")
MODEL_PATH = os.environ.get("MODEL_PATH")

app = flask.Flask(__name__)


def load_model(model_path: str) -> Any:
    return mlflow.sklearn.load_model(model_path)


def create_new_row(params: Dict) -> Dict:
    if "G1" in params.keys():
        return {
            "G1": params.get("G1"),
            "G2": params.get("G2"),
            "G3": params.get("G3"),
            "G4": params.get("G4"),
            "G5": params.get("G5"),
            "G6": params.get("G6"),
            "G7": params.get("G7"),
            "G8": params.get("G8"),
            "G9": params.get("G9"),
            "G10": params.get("G10")
        }
    else:
        return {"G1": 0}


def format_new_row(new_row: Dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(new_row, orient='index').tranpose()


def make_prediction(model: Any, X: pd.DataFrame) -> Dict:

    data = {}
    data["response"] = str(model.predict_proba(X)[0, 1])
    data["success"] = True

    return data


def jsonify_response(data: Dict) -> Any:
    return flask.jsonify(data)


@app.route("/", methods=["GET", "POST"])
def predict():

    model = load_model(model_path=MODEL_PATH)
    new_row = create_new_row(params=flask.request.args)
    X_new = format_new_row(new_row=new_row)
    response_data = make_prediction(model=model, X=X_new)

    return jsonify_response(data=response_data)


if __name__ == "__main__":
    app.run(host='0.0.0.0')