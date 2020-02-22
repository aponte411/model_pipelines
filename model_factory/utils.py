import logging
import numpy as np
import pandas as pd
from typing import Tuple, Any, List
import os
import boto3
from boto3.s3 import transfer

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

