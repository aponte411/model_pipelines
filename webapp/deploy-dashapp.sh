export FOLD=0
export MODEL=randomforest
export MODEL_PATH=models/${MODEL}_${FOLD}_trained

python webapp/dash_app.py
