# ML-PROJECT-TEMPLATE

Repository to preprocess data, train models, and make inference.

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
    - `export FOLD=0`
    - `export MODEL=<model-name>`
    - `python src/train.py`

3. Run the training script: `sh training-script.sh`
