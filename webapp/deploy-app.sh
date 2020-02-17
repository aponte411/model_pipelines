export FOLD=$1
export MODEL=$2
export MODEL_PATH=~/KAGGLE_COMPETITIONS/kaggle-template/ml-project-template/models/${MODEL}_${FOLD}_trained

gunicorn --bind 0.0.0.0 wsgi:application
