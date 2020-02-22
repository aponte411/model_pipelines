export INPUT=inputs/quora_question_pairs/train.csv
export OUTPUT=inputs/quora_question_pairs/train-folds.csv
export TARGET=is_duplicate
python model_factory/create_folds.py
