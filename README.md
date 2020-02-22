# ML-PROJECT-TEMPLATE

Repository to preprocess data, train models, and make inference. I set up it so that it can generalize to using AWS, GCP, and any dataset. 

Using inspiration from https://github.com/bgweber and https://github.com/abhishekkrthakur.

# Setup

1. Create a virtual environment using conda, virtualenv, virtualenvwrapper, etc. then pip install the requirements. I recommend using virtualenvwrapper as it makes it super easy to switch between virtual environments https://virtualenvwrapper.readthedocs.io/en/latest/:

    - `mkvirtualenv new-project`
    - `pip install -r requirements.txt`

2. Download a dataset and store it in `inputs` for example:`
    - `mkdir inputs` && `cd inputs`
    - `mkdir quora_question_pairs` && `cd quora_question_pairs`
    - `kaggle competitions download -c quora-question-pairs`

3. Train a model and set MODEL_PATH - e.g. `export MODEL_PATH=trained_models/<model-name>`:

    - `mkdir trained_models`

4. Set up connections to AWS or GCP. This step is a little more involved so checkout the documentation:

    - AWS: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration
    - GCP: https://cloud.google.com/sdk/gcloud/reference/auth/login

# Training Models WIP

1. Define your model in `model_factory/dispatcher`:
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

2. Setup pipeline-creds.json containing all of your GCP credentials info.

3. Setup environmental variables:

    - `export PROJECT_ID=<project-id>`
    - `export IMAGE_NAME=<image-name>`
    - `export CREDS=<path-to-creds-file>`

4. Run the setup scripts:

    - `sh set-up-creds.sh && sh push-to-gcr.io`
