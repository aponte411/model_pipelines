export TRAINING_DATA=inputs/games/games-train-folds.csv
export FOLD=0
export MODEL=randomforest
export MODEL_PATH=models/${MODEL}_${FOLD}_trained
export TARGET=label

python model_factory/train
