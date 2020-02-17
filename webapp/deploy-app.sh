export FOLD=0
export MODEL=randomforest
export MODEL_PATH=models/${MODEL}_${FOLD}_trained

gunicorn --bind 0.0.0.0 wsgi:application
