import os
from typing import Any, Dict

import dash
import dash_core_components as dcc
import dash_html_components as html
import mlflow.sklearn
import pandas as pd
from dash.dependencies import Input, Output

MODEL_PATH = os.environ.get("MODEL_PATH")

application = dash.Dash(__name__)

application.layout = html.Div(children=[
    html.H1(children='Model UI'),
    html.P([html.Label('Game 1'),
            dcc.Input(value='1', type='text', id='g1')]),
    html.Div(
        [html.Label('Game 2'),
         dcc.Input(value='0', type='text', id='g2')]),
    html.
    P([html.Label('Prediction'),
       dcc.Input(value='0', type='text', id='pred')]),
])


def create_new_row(game1: Any, game2: Any) -> Dict:
    return {
        "G1": float(game1),
        "G2": float(game2),
        "G3": 0,
        "G4": 0,
        "G5": 0,
        "G6": 0,
        "G7": 0,
        "G8": 0,
        "G9": 0,
        "G10": 0,
        "G11": 0
    }


def format_new_row(new_row: Dict) -> pd.DataFrame:
    return pd.DataFrame.from_dict(new_row, orient='index').transpose()


def load_model(model_path: str) -> Any:
    return mlflow.sklearn.load_model(model_path)


def make_prediction(model: Any, X: pd.DataFrame) -> str:
    return str(model.predict_proba(X)[0, 1])


@application.callback(Output(component_id='pred', component_property='value'),
                      [
                          Input(component_id='g1', component_property='value'),
                          Input(component_id='g2', component_property='value')
                      ])
def update_prediction(game1, game2):

    new_row = create_new_row(game1=game1, game2=game2)
    X_new = format_new_row(new_row=new_row)
    model = load_model(model_path=MODEL_PATH)

    return make_prediction(model=model, X=X_new)


if __name__ == "__main__":
    application.run_server(host='0.0.0.0')
