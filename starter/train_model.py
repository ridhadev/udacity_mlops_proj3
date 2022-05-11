# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import os
from starter.ml.data import process_data
from starter.ml.model import train_model, inference, compute_model_metrics, compute_model_metrics_by_slice
import pickle
import logging

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

def save_model(models_folder: str, model, encoder, lb)-> None:
    """
    Save the input model and feature transformers into a models_folder.
    Create the models_folder if it does not exist.

    Parameters
    ----------
    models_folder
    model
    encoder
    lb

    Returns
    -------

    """
    os.makedirs(models_folder, exist_ok=True)

    pickle.dump(
        encoder,
        open(
            os.path.join(
                models_folder,
                "encoder.pkl"),
            'wb'))
    pickle.dump(
        lb,
        open(
            os.path.join(
                models_folder,
                "label_binarizer.pkl"),
            'wb'))
    pickle.dump(
        model,
        open(
            os.path.join(
                models_folder,
                "model.pkl"),
            'wb'))

def train_and_save_model(data_filepath: str, models_folder: str) -> None:
    """
    Train the predictive model using input data located at data_filepath
    and save the trained model under models_folder.

    Parameters
    ----------
    data_filepath : Training data
    models_folder : Output folder path

    Returns
    -------
        None
    """
    logging.info(f"Reading file '{data_filepath}'...")
    data = pd.read_csv(data_filepath)

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20)

    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary", training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary", training=False, encoder=encoder, lb=lb)

    # Train and save a model.
    trained_model = train_model(X_train, y_train)

    save_model(models_folder, trained_model, encoder, lb)

    y_predict = inference(trained_model, X_test)
    overall_metrics = compute_model_metrics(y_test, y_predict)

    print(f"Overall metrics : {overall_metrics}")

    metrics_by_slice_df = compute_model_metrics_by_slice(
        data, trained_model, encoder, lb, CAT_FEATURES)
    metrics_by_slice_df.to_csv("metrics_by_slice.csv")


if __name__ == "__main__":
    # Add code to load in the data.
    data_filepath = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'data',
            'census_clean.csv'))

    if not os.path.exists(data_filepath):
        raise FileNotFoundError(
            f"Failed to find data file in : '{data_filepath}'")

    models_folder = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'models')
    )
    train_and_save_model(data_filepath, models_folder)
