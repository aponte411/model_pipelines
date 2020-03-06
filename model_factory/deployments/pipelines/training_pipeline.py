import os

import utils
from engines import BengaliEngine
from models import ResNet34
from trainers import BengaliTrainer

LOGGER = utils.get_logger(__name__)

CREDENTIALS = {}
CREDENTIALS['aws_access_key_id'] = os.environ.get("aws_access_key_id")
CREDENTIALS['aws_secret_access_key'] = os.environ.get("aws_secret_access_key")
CREDENTIALS['bucket'] = os.environ.get("bucket")

ENGINE_PARAMS = {
        "train_path": "inputs/train-folds.csv",
        "test_path": "inputs",
        "pickle_path": "inputs/pickled_images",
        "model_dir": "trained_models",
        "submission_dir": "inputs"
        "train_folds": [0],
        "val_folds": [4],
        "train_batch_size": 64,
        "test_batch_size": 32,
        "epochs": 3,
        "test_loops": 5,
        "image_height": 137,
        "image_width": 236,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.239, 0.225)
    }


def main(credentials: Dict, engine_params: Dict) -> Optional:
    model = ResNet34(pretrained=False)
    trainer = BengaliTrainer(model=model, model_name='resnet34')
    bengali = BengaliEngine(trainer=trainer, params=engine_params)
    submission = bengali.run_inference_engine(model_dir=engine_params['model_dir'],
                                              to_csv=True,
                                              output_dir=engine_params['submission_dir'],
                                              load_from_s3=True,
                                              creds=credentials)

if __name__ == "__main__":
    main(credentials=CREDENTIALS, bengine_params=ENGINE_PARAMS)
