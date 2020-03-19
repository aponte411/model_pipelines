import argparse
import types
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras import Model, layers, optimizers
from tensorflow.keras.models import Model, load_model

import utils
LOGGER = utils.get_logger(__name__)


def parse_args() -> types.SimpleNamespace:
    parser = argparse.ArgumentParser(description='Create Entity Embeddings', )
    parser.add_argument('--train-path',
                        default='inputs/categorical_challenge/train.csv')
    parser.add_argument('--test-path',
                        default='inputs/categorical_challenge/test.csv')
    parser.add_argument('--target', default='target')
    return parser.parse_args()


def combine_train_and_test(args: types.SimpleNamespace) -> pd.DataFrame:
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    test[args.target] = -1
    combined = pd.concat([train, test]).reset_index(drop=True)
    LOGGER.info(f'Combined DataFrame shape: {combined.shape}')
    return combined


def prepare_train_test_data(df: pd.DataFrame) -> Dict:
    def _get_feature_names(df: pd.DataFrame) -> List[str]:
        return [
            feature for feature in df.columns
            if feature not in ['target', 'id']
        ]

    def _split_combined_into_train_test(combined_df: pd.DataFrame) -> Tuple:
        train = combined_df.loc[combined_df.target != -1].reset_index(
            drop=True)
        test = combined_df.loc[combined_df.target == -1].reset_index(drop=True)
        return train, test

    def _label_encode(df: pd.DataFrame, feature_names: List[str]) -> Tuple:
        for feat in feature_names:
            LOGGER.info(f'Preprocessing Feature {feat}')
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[feat].values.tolist())
            df[feat] = lbl.transform(df[feat].values)
        return df

    feature_names = _get_feature_names(df=df)
    encoded_data = _label_encode(df=df, feature_names=feature_names)
    train, test = _split_combined_into_train_test(combined_df=encoded_data)
    return {
        'X_train': train,
        'X_val': test,
        'y_train': train.target,
        'y_valid': test.target,
        'feature_names': feature_names
    }


def create_model(df: pd.DataFrame, features: List[str]) -> Model:
    inputs, outputs = [], []
    for feature in features:
        num_unique_vals = df[feature].nunique()
        embedding_dim = min((num_unique_vals // 2), 50)
        input_layer = layers.Input(shape=(1, ))
        output = layers.Embedding(num_unique_vals + 1,
                                  embedding_dim,
                                  name=feature)(input_layer)
        output_layer = layers.Reshape(target_shape=(embedding_dim, ))(output)
        inputs.append(input_layer)
        outputs.append(output_layer)

    x = layers.Concatenate()(outputs)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    y = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=y)


def listify_features(df: pd.DataFrame, features: List[str]) -> List[np.array]:
    return [df.loc[:, feature].values for feature in features]


def main(args: types.SimpleNamespace):
    combined_data = combine_train_and_test(args=args)
    data_dictionary = prepare_train_test_data(df=combined_data)
    feature_lists = listify_features(df=data_dictionary['X_train'],
                                     features=data_dictionary['feature_names'])
    model = create_model(df=data_dictionary['X_train'],
                         features=data_dictionary['feature_names'])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(feature_lists, data_dictionary['y_train'])


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
