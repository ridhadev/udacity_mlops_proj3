# Script to train machine learning model.
import os
import pandas as pd
import pickle
import logging
from datetime import datetime
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, compute_model_metrics_by_slice


MODEL_CARD_TEMPLATE = '''
## Model Card
__MODEL_CARD_DESC__

## Model Details
__MODEL_DETAILS_DESC__

## Intended Use
__INTENDED_USE_DESC__

## Training Data
__TRAIN_DATA_DESC__

## Evaluation Data
__EVAL_DATA_DESC__

## Metrics
__METRICS_DESC__

## Ethical Considerations
__ETHICAL_DESC__

## Caveats and Recommendations
__RECO_DESC__
    '''

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


def save_model(models_folder: str, model, encoder, lb) -> None:
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


def train_and_save_model(data_filepath: str, models_folder: str) -> tuple:
    """
    Train the predictive model using input data located at data_filepath
    and save the trained model under models_folder.

    The method evaluate the model on categorical's model slices
    see: compute_model_metrics_by_slice

    Parameters
    ----------
    data_filepath : Training data
    models_folder : Output folder path

    Returns
    -------
        Model overall metrics as a tuple, respectively: precision, recall and f1-score.
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

    metrics_by_slice_df = compute_model_metrics_by_slice(
        data, trained_model, encoder, lb, CAT_FEATURES)
    metrics_by_slice_df.to_csv("metrics_by_slice.csv")

    return overall_metrics


def create_model_card(precision, recall, f1):
    """
    Generate model card documenting the generated model.
    """

    mod_card_desc = f"This card was generated on _{datetime.now():%Y-%m-%d %H:%M}_ \
    to summarize last model performance for the project __Census Income Data Set__"

    mod_details_desc = """
The predictive model is based on three part :
    - A **label binarizer** to convert the target income column into a binary data
    - An **encoder** to convert categorical features into digital ones
    - A **random forest** classifier to predict the income category ( > or < 50k)
    """

    mod_use_desc = """
Predict whether income exceeds $50K/yr based on census data.
"""
    train_data_desc = """
The used dataset and relative details can be found [here](https://archive.ics.uci.edu/ml/datasets/census+income)
    """
    eval_data_desc = """
The dataset was split randomly into 80% data for training and 20% for model evaluation.
    """

    metrics_desc = f"The current metrics of the model is as following:\n\n\
    - Precision : {precision:.3f}\n\
    - Recall    : {recall:.3f}\n\
    - F1 Score  : {f1:.3f}\n\
    "

    ethical_desc = """
    Even though this data is completely anonymous and open to the public,
    it contains sensitive data relating to people's private lives such
    as their income levels, origins...

    Users of this data should pay particular attention to the biases
    that may exist or be introduced in order not to favor or disadvantage
    one category of the population over another.
    """

    reco_desc = """
    The use and application of all or part of this model and these results
    are the entire responsibility of the user.
"""

    card_md = MODEL_CARD_TEMPLATE.replace("__MODEL_CARD_DESC__", mod_card_desc)
    card_md = card_md.replace("__MODEL_DETAILS_DESC__", mod_details_desc)
    card_md = card_md.replace("__INTENDED_USE_DESC__", mod_use_desc)
    card_md = card_md.replace("__TRAIN_DATA_DESC__", train_data_desc)
    card_md = card_md.replace("__EVAL_DATA_DESC__", eval_data_desc)

    card_md = card_md.replace("__METRICS_DESC__", metrics_desc)
    card_md = card_md.replace("__ETHICAL_DESC__", ethical_desc)
    card_md = card_md.replace("__RECO_DESC__", reco_desc)

    return card_md


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

    metrics = train_and_save_model(data_filepath, models_folder)

    model_card_content = create_model_card(*metrics)
    with open(os.path.join(models_folder, "model_card.md"), "w") as md:
        md.write(model_card_content)
