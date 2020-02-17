export FOLD=0
export MODEL=$1
export MODEL_PATH=models/${MODEL}_${FOLD}_trained

python -m webapp.app
