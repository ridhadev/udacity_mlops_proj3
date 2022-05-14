import os
import shutil
import pickle

import pandas as pd
import pytest
from starter.ml.data import process_data
from starter.train_model import CAT_FEATURES


@pytest.fixture
def test_data():
    """Loads the sample test data for clean census"""
    data_filepath = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            '..', '..',
            'data',
            'test_census_clean.csv')
    )

    return pd.read_csv(data_filepath)


@pytest.fixture
def tmp_folder():
    """Creates a temporary test folder"""
    test_folder_path = "__TMP__"
    os.makedirs(test_folder_path, exist_ok=True)
    yield test_folder_path
    shutil.rmtree(test_folder_path)


def test_cat_features_in_data(test_data):
    """Test all categorical features are in the dataset ase expected"""
    assert set(CAT_FEATURES).issubset(set(test_data.columns))


def test_process_data_with_train(test_data, tmp_folder):
    """Test the process data is working in training mode"""

    X_train, y_train, encoder, lb = process_data(
        test_data, categorical_features=CAT_FEATURES, label="salary", training=True)

    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0
    assert encoder is not None
    assert lb is not None

    pickle.dump(
        encoder,
        open(
            os.path.join(
                tmp_folder,
                "encoder.pkl"),
            'wb'))

    assert os.path.exists(os.path.join(tmp_folder, "encoder.pkl"))


def test_process_data_no_train(test_data):
    """Test the process data is working in inference mode"""
    X_train, y_train, input_encoder, input_lb = process_data(
        test_data, categorical_features=CAT_FEATURES, label="salary", training=True)

    X_train, y_train, output_encoder, output_lb = process_data(
        test_data, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=input_encoder, lb=input_lb
    )

    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0
    assert output_encoder is input_encoder
    assert output_lb is input_lb
