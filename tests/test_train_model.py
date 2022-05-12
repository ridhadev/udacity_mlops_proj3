import os
import shutil
import pytest
from starter.train_model import save_model
from sklearn.preprocessing._encoders import OneHotEncoder
from sklearn.preprocessing._label import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier

from starter.train_model import CAT_FEATURES


@pytest.fixture
def tmp_folder():
    """Creates a temporary test folder"""
    test_folder_path = "__TMP__"
    os.makedirs(test_folder_path, exist_ok=True)
    yield test_folder_path
    shutil.rmtree(test_folder_path)


def test_categorical_features():
    """Test all categorical features are those excpected and have not changed"""
    assert sorted(CAT_FEATURES) == sorted(
        ['education',
         'marital-status',
         'native-country',
         'occupation',
         'race',
         'relationship',
         'sex',
         'workclass']
    )


def test_save_models(tmp_folder):
    """Test the models are concretely saved"""
    fake_encoder = OneHotEncoder()
    fake_lb = LabelBinarizer()
    fake_model = RandomForestClassifier()

    save_model(tmp_folder, fake_model, fake_encoder, fake_lb)

    assert os.path.exists(os.path.join(tmp_folder, "encoder.pkl"))
    assert os.path.exists(os.path.join(tmp_folder, "label_binarizer.pkl"))
    assert os.path.exists(os.path.join(tmp_folder, "model.pkl"))
