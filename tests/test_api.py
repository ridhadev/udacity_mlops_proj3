from api import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_hello_world():
    r = client.get('/')
    assert r.status_code == 200
    assert r.content == b"{\"message\":\"Welcome to Census Income FastAPI app.\"}"


def test_negative_predict():
    """Tests post prediction case below 50K"""
    income_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    }
    # json argument expects a dict
    r = client.post("/prediction/", json=income_data)
    assert r.status_code == 200
    assert r.content == b"{\"predict\":\"<=50K\"}"


def test_positive_predict():
    """Tests post prediction case above 50K"""

    income_data = {
        "age": 30,
        "workclass": "State-gov",
        "fnlwgt": 141297,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "India"
    }
    r = client.post("/prediction/", json=income_data)
    assert r.status_code == 200
    assert r.content == b"{\"predict\":\">50K\"}"
