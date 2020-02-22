export TRAINING_DATA=inputs/quora_question_pairs/train-folds.csv
export FOLD=0
export MODEL=randomforest
export BUCKET=${BUCKET}
export DATA=quora
export MODEL_PATH=models/${MODEL}_${FOLD}_${DATA}_trained
export TARGET=is_duplicate
python model_factory/trainers.py
