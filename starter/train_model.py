# Script to train machine learning model.
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas
import os
from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics

import logging

# Add code to load in the data.
data_filepath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'data', 'census_clean.csv'))

if not os.path.exists(data_filepath):
    raise FileNotFoundError(f"Failed to find data file in : '{data_filepath}'")

logging.info(f"Reading file '{data_filepath}'...")
data = pd.read_csv(data_filepath)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Proces the test data with the process_data function.
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb
)

# Train and save a model.
trained_model = train_model(X_train, y_train)


y_predict = inference(trained_model, X_test)

print(compute_model_metrics(y_test, y_predict))
