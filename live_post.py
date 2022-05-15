import json
import requests

URL = "https://udacity-mlops-proj3.herokuapp.com/prediction"

POST_SAMPLE_DATA = {
    "age":27,
    "workclass":"Private",
    "fnlwgt":36440,
    "education":"Bachelors",
    "education_num":13,
    "marital_status":"Never-married",
    "occupation":"Sales",
    "relationship":"Not-in-family",
    "race":"White",
    "sex":"Female",
    "capital_gain":0,
    "capital_loss":0,
    "hours_per_week":40,
    "native_country":"United-States"
}

if __name__== "__main__":
    r_post = requests.post(URL, data=json.dumps(POST_SAMPLE_DATA))
    if r_post.status_code != 200:
        print(f"Post command failed with exit code {r_post.status_code}")
    else:
        print(f"Post command status : {r_post.status_code}")
        print(f"Post command result : {r_post.json()}")
