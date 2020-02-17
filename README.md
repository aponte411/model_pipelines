# ML-PROJECT-TEMPLATE

Repository to preprocess data, train models, and make inference. I set up it so that it can generalize to using AWS, GCP, and any dataset. 

# Setup

1. Create a virtual environment using conda, virtualenv, virtualenvwrapper, etc. then pip install the requirements: `pip install -r requirements.txt`

2. `mkdir inputs`

3. Download a dataset and store it in `inputs`

4. `mkdir models`

5. Train a model and set MODEL_PATH - e.g. `export MODEL_PATH=<path>`

6. Set up connections to AWS or GCP: 

    - AWS: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration
    - GCP: https://cloud.google.com/sdk/gcloud/reference/auth/login

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


# Deploy Web Application WIP

1. `sh bash_scripts/run-app.sh`

# Run pipeline

1. Setup pipeline-creds.json containg all of your GCP credentials info.

2. `export PROJECT_ID=<project-id> && export IMAGE_NAME=<image-name>`

3. `sh set-up-creds.sh && sh push-to-gcr.io`