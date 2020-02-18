# ML-PROJECT-TEMPLATE

Repository to preprocess data, train models, and make inference. I set up it so that it can generalize to using AWS, GCP, and any dataset. 

Using inspiration from https://github.com/bgweber and https://github.com/abhishekkrthakur.

# Setup

1. Create a virtual environment using conda, virtualenv, virtualenvwrapper, etc. then pip install the requirements: `pip install -r requirements.txt`

2. `mkdir inputs`

3. Download a dataset and store it in `inputs`

4. `mkdir trained_models`

5. Train a model and set MODEL_PATH - e.g. `export MODEL_PATH=<path>`

6. Set up connections to AWS or GCP:

    - AWS: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration
    - GCP: https://cloud.google.com/sdk/gcloud/reference/auth/login

# Training WIP

1. Define your model in model_factory/dispatcher:
    - `MODELS = {
        "randomforest":
        ensemble.RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=2),
        "extratrees":
        ensemble.ExtraTreesClassifier(n_estimators=300, n_jobs=-1, verbose=2)
    }`

2. Setup a bash script (e.g. `training-script.sh`) to define your environmental variables:
    - `export TRAINING_DATA=<path-to-train>`
    - `export FOLD=<fold-number>`
    - `export MODEL=<model-name>`
    - `python model_factory/train.py`

3. Run the training script: `sh training-script.sh`


# Deploy Web Application Locally WIP (EC2 coming soon)

1. `cd deployments/webapp`

2. `sh bash_scripts/run-app.sh`

# Run pipeline

1. `cd deployments/pipeline`

2. Setup pipeline-creds.json containg all of your GCP credentials info.

3. `export PROJECT_ID=<project-id>`

4. `export IMAGE_NAME=<image-name>`

5. `export CREDS=<path-to-creds-file>`

5. `sh set-up-creds.sh && sh push-to-gcr.io`