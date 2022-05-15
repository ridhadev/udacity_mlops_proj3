# Put the code for your API here.
"""Implements RESTful API using FastAPI
"""
import sys
import os
starter_root = os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            'starter'
        )
    )
print(starter_root)
sys.path.insert(0, starter_root)
print(sys.path)
import pickle

from train_model import CAT_FEATURES
from ml.data import process_data
import pandas as pd
from pydantic import BaseModel, Field
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI




class IncomeData(BaseModel):
    age: int = Field(example="37")
    workclass: str = Field(
        example="Private, Self-emp-not-inc, Self-emp-inc, Federal-gov,\
         Local-gov, State-gov, Without-pay, Never-worked")
    fnlwgt: int = Field(description="continuous", example="141297")
    education: str = Field(
        example="Bachelors, Some-college, 11th, HS-grad, Prof-school,\
        Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool")
    education_num: int = Field(description="continuous", example="13")
    marital_status: str = Field(
        example="Married-civ-spouse, Divorced, Never-married, Separated,\
        Widowed, Married-spouse-absent, Married-AF-spouse")
    occupation: str = Field(
        example="Tech-support, Craft-repair, Other-service, Sales, Exec-managerial,\
         Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical,\
         Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces")
    relationship: str = Field(
        example="Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried")
    race: str = Field(
        example="White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black")
    sex: str = Field(example="Male, Female")
    capital_gain: int = Field(description="continuous", example="0")
    capital_loss: int = Field(description="continuous", example="0")
    hours_per_week: int = Field(description="continuous", example="40")
    native_country: str = Field(example="United-States, Cambodia, England, Puerto-Rico,\
     Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China,\
     Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico,\
     Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti,\
      Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia,\
      El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.")


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
