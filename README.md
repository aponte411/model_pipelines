# ML-PROJECT-TEMPLATE

Repository to preprocess data, train models, and make inference.

# Setup

1. Create a virtual environment using conda, virtualenv, virtualenvwrapper, etc. then pip install the requirements: `pip install -r requirements.txt`

# Training WIP

1. Define your model in src/dispatcher:
    - `MODELS = {
        "randomforest":
        ensemble.RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2),
        "extratrees":
        ensemble.ExtraTreesClassifier(n_estimators=300, n_jobs=-1, verbose=2)
    }`

2. Setup a bash script (e.g. `training-script.sh`) to define your environmental variables:
    - `export AWS_ACCESS_KEY_ID=<aws-access-key-id>`
    - `export AWS_SECRET_ACCESS_KEY=<aws-secret-access-key>`
    - `export BUCKET=<bucket>`
    - `export TRAINING_DATA=<path-to-train>`
    - `export FOLD=<fold-number>`
    - `export MODEL=<model-name>`
    - `python src/train.py`

3. Run the training script: `sh training-script.sh`


# Deploy WebApp WIP

1. `cd webapp`

2. `sh deploy-app.sh <fold> <model-name>`