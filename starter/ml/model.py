from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from ml.data import process_data

# Optional: implement hyperparameter tuning.


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(min_samples_split=30)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def compute_model_metrics_by_slice(
        data: pd.DataFrame,
        model,
        encoder,
        lb,
        slices_features):
    """
    Evaluates the performance of the input model on all categorical slices features of the input data.

    Parameters
    ----------
    data: Input Data
    model: Sklearn model
    encoder: sklearn.preprocessing.OneHotEncoder
    lb: sklearn.preprocessing.LabelBinarizer
    slices_features

    Returns
    -------
    Data Frame with the category name, slice name and corresponding metrics at each row.
    The category and slice names are set as index.
    """
    slice_results = []
    for slice_cat in slices_features:

        for slice_val in set(data[slice_cat].values):
            slice_data = data[data[slice_cat] == slice_val].copy()

            xslice, yslice, _, _ = process_data(
                slice_data, categorical_features=slices_features, label="salary", training=False, encoder=encoder, lb=lb)
            y_predict = inference(model, xslice)

            results = compute_model_metrics(yslice, y_predict)
            results_dict = dict(zip(("precision", "recall", "fbeta"), results))
            results_dict["slice_category"] = slice_cat
            results_dict["slice_value"] = slice_val
            results_dict["count"] = slice_data.shape[0]
            slice_results.append(results_dict)

    return pd.DataFrame(slice_results).set_index(["slice_category", "slice_value"]).sort_values(
        ["slice_category", "fbeta", "slice_value"])
