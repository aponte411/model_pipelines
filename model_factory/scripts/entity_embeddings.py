import argparse
import types
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import preprocessing
from tensorflow.keras import Model, callbacks, layers, optimizers
from tensorflow.keras import utils as keras_utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import multi_gpu_model

import metrics
import utils

LOGGER = utils.get_logger(__name__)


def parse_args() -> types.SimpleNamespace:
    parser = argparse.ArgumentParser(description='Create Entity Embeddings', )
    parser.add_argument('--train-path',
                        default='inputs/categorical_challenge/train.csv')
    parser.add_argument('--test-path',
                        default='inputs/categorical_challenge/test.csv')
    parser.add_argument('--target', default='target')
    parser.add_argument('--model-path', default='trained_models')
    parser.add_argument('--n-gpus', default=4)
    return parser.parse_args()


def combine_train_and_test(args: types.SimpleNamespace) -> pd.DataFrame:
    train = pd.read_csv(args.train_path)
    test = pd.read_csv(args.test_path)
    test[args.target] = -1
    combined = pd.concat([train, test]).reset_index(drop=True)
    LOGGER.info(f'Combined DataFrame shape: {combined.shape}')
    return combined


def prepare_data_dictionary(df: pd.DataFrame) -> Dict:
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
            encoder = preprocessing.LabelEncoder()
            df[feat] = encoder.fit_transform(
                df[feat].fillna("-1").astype(str).values)
        return df

    feature_names = _get_feature_names(df=df)
    encoded_data = _label_encode(df=df, feature_names=feature_names)
    train, test = _split_combined_into_train_test(combined_df=encoded_data)
    return {
        'combined': encoded_data,
        'X_train': train,
        'X_valid': test,
        'y_train': train.target,
        'y_valid': test.target,
        'feature_names': feature_names
    }


def create_model(df: pd.DataFrame, features: List[str]) -> Model:
    inputs, outputs = [], []
    for feature in features:
        num_unique_vals = int(df[feature].nunique())
        embedding_dim = int(min(np.ceil(num_unique_vals / 2), 50))
        input_layer = layers.Input(shape=(1, ))
        output = layers.Embedding(num_unique_vals + 1,
                                  embedding_dim,
                                  name=feature)(input_layer)
        output_layer = layers.Reshape(target_shape=(embedding_dim, ))(output)
        inputs.append(input_layer)
        outputs.append(output_layer)

    x = layers.Concatenate()(outputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    y = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs=inputs, outputs=y)


def listify_features(df: pd.DataFrame, features: List[str]) -> List[np.array]:
    return [
        df.loc[:, features].values[:, k]
        for k in range(df.loc[:, features].values.shape[1])
    ]


def main(args: types.SimpleNamespace):
    combined_data = combine_train_and_test(args=args)
    data_dictionary = prepare_data_dictionary(df=combined_data)
    train_feature_lists = listify_features(
        df=data_dictionary['X_train'],
        features=data_dictionary['feature_names'])
    val_feature_lists = listify_features(
        df=data_dictionary['X_valid'],
        features=data_dictionary['feature_names'])
    model = create_model(df=data_dictionary['combined'],
                         features=data_dictionary['feature_names'])
    model = multi_gpu_model(model, gpus=args.n_gpus, cpu_relocation=True)
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=[metrics.keras_auc])
    early_stopping = callbacks.EarlyStopping(monitor='val_auc',
                                             min_delta=0.001,
                                             patience=5,
                                             verbose=1,
                                             mode='max',
                                             baseline=None,
                                             restore_best_weights=True)
    reduce_lr = callbacks.ReduceLROnPlateau(monitor='val_auc',
                                            factor=0.5,
                                            patience=3,
                                            min_lr=1e-6,
                                            mode='max',
                                            verbose=1)
    model.fit(train_feature_lists,
              data_dictionary['y_train'],
              validation_data=(val_feature_lists, data_dictionary['y_valid']),
              verbose=1,
              batch_size=1024,
              callbacks=[early_stopping, reduce_lr],
              epochs=100)
    model.save(f'{args.model_path}/keras_entity_embedding.h5')


if __name__ == "__main__":
    args = parse_args()
    main(args=args)
