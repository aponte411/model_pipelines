export INPUT=inputs/games/games-expand.csv
export OUTPUT=inputs/games/games-train-folds.csv
export TARGET=label

python model_factory/create_folds.py
