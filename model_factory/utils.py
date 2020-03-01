import glob
import logging
import os
from typing import Any, List, Tuple

import boto3
import joblib
import numpy as np
import pandas as pd
import torch
from boto3.s3 import transfer
from tqdm import tqdm

from configs import __config__

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s = %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S")


def get_logger(name, level=logging.INFO) -> Any:
    """Returns logger object with given name"""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger


LOGGER = get_logger(__name__)


class S3Client:
    """
    Sets up a client to download/upload
    s3 files.
    : aws access key id: login
    : aws secret access key: password
    : bucket: path to file
    """
    def __init__(self,
                 user=__config__["aws_access_key_id"],
                 password=__config__["aws_secret_access_key"],
                 bucket=__config__["s3_bucket_name"]):
        self.bucket = bucket
        self.client = boto3.client('s3',
                                   aws_access_key_id=user,
                                   aws_secret_access_key=password)

    def upload_file(self, filename: str, key: str) -> None:

        s3t = transfer.S3Transfer(self.client)
        s3t.upload_file(filename, self.bucket, key)
        LOGGER.info('File successfully uploaded!')

    def download_file(self, filename: str, key: str) -> None:

        s3t = transfer.S3Transfer(self.client)
        s3t.download_file(self.bucket, key, filename)
        LOGGER.info('File successfully downloaded!')


# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            LOGGER.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}'
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            LOGGER.info(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def pickle_images(input: str, output_dir: str):
    for file_name in glob.glob(input):
        df = pd.read_parquet(file_name)
        image_ids = df.image_id.values
        image_array = df.drop('image_id', axis=1).values
        for idx, image_id in tqdm(enumerate(image_ids), total=len(image_ids)):
            joblib.dump(image_array[idx, :], f"{output_dir}/{image_id}.p")
