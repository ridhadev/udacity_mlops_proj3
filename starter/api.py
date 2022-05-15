"""Implements RESTful API using FastAPI
"""
import os
import pickle
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
import pandas as pd
from ml.data import process_data
from train_model import CAT_FEATURES


class IncomeData(BaseModel):
    age: int
    workclass: str
    fnlwgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    _example_ = """
    IncomeData(
        age=30,
        workclass="State-gov",
        fnlwgt=141297,
        education="Bachelors",
        education_num=13,
        marital_status="Married-civ-spouse",
        occupation="Prof-specialty",
        relationship="Husband",
        race="Asian-Pac-Islander",
        sex="Male",
        capital_gain=0,
        capital_loss=0,
        hours_per_week=40,
        native_country="India"
       )"""


def load_models(model_folder_path: str):
    with open(os.path.join(model_folder_path, "model.pkl"), "rb") as f:
        estimator = pickle.load(f)

    with open(os.path.join(model_folder_path, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(model_folder_path, "label_binarizer.pkl"), "rb") as f:
        lb = pickle.load(f)

    return estimator, encoder, lb


app = FastAPI()


@app.get("/")
async def hello_world():
    return {"message": "Welcome to Census Income FastAPI app."}


@app.post('/prediction/')
def predict(data: IncomeData):

    models_dir = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            '..',
            'models')
    )

    model, encoder, lb = load_models(models_dir)

    json_dict = jsonable_encoder(data)
    input_data = pd.DataFrame.from_dict([json_dict], orient='columns')
    input_data = input_data.rename(columns=lambda x: x.replace('_', '-'))

    x, _, _, _ = process_data(input_data, categorical_features=CAT_FEATURES,
                              label=None, training=False, encoder=encoder, lb=lb)

    y_pred = model.predict(x)

    y_pred_label = lb.inverse_transform(y_pred)[0]

    return {"predict": y_pred_label}
