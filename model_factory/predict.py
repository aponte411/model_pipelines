import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import click
import numpy as np
import pandas as pd

import datasets
import engines
import models
import trainers
import utils
from dispatcher import MODEL_DISPATCHER

LOGGER = utils.get_logger(__name__)

CREDENTIALS = {}
CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
CREDENTIALS['bucket'] = os.environ.get("bucket")


def parse_args():
    parser = argparse.ArgumentParser(description='Conduct Inference', )
    parser.add_argument('--model-name', default='resnet34')
    parser.add_argument('--train-path',
                        default='inputs/bengali_grapheme/train-folds.csv')
    parser.add_argument('--test-path', default='inputs/bengali_grapheme')
    parser.add_argument('--pickle-path', default="inputs/pickled_images")
    parser.add_argument('--model-dir', default='trained_models')
    parser.add_argument('--submission-dir', default='inputs/bengali_grapheme')
    parser.add_argument('--train-folds', default=[0])
    parser.add_argument('--val-folds', default=[4])
    parser.add_argument('--train-batch-size', default=64)
    parser.add_argument('--test-batch-size', default=32)
    parser.add_argument('--epochs', default=3)
    parser.add_argument('--test-loops', default=5)
    return parser.parse_args()


def main(args) -> Optional:
    ENGINE_PARAMS = {
        "train_path": args.train_path,
        "test_path": args.test_path,
        "pickle_path": args.pickle_path,
        "model_dir": args.model_dir,
        "submission_dir": args.submission_dir,
        "train_folds": args.train_path,
        "val_folds": args.val_folds,
        "train_batch_size": args.train_path,
        "test_batch_size": args.test_path,
        "epochs": args.epochs,
        "test_loops": args.test_loops,
        "image_height": 137,
        "image_width": 236,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.239, 0.225)
    }
    model = MODEL_DISPATCHER.get(args.model_name)
    trainer = trainers.BengaliTrainer(model=model, model_name=args.model_name)
    bengali = engines.BengaliEngine(trainer=trainer, params=ENGINE_PARAMS)
    submission = bengali.run_inference_engine(
        model_name=args.model_name,
        model_dir=ENGINE_PARAMS['model_dir'],
        to_csv=True,
        output_dir=ENGINE_PARAMS['submission_dir'],
        load_from_s3=True,
        creds=CREDENTIALS)
    LOGGER.info(submission)


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
