# pytest integration tests

import requests

#ENDPOINT = "http://127.0.0.1:8000"
ENDPOINT = "http://localhost:8000"


def test_can_call_endpoint():
    response = requests.get(ENDPOINT)
    assert response.status_code == 200

def test_can_get_features():
    response = requests.get(ENDPOINT + '/features')
    assert response.status_code == 200

def test_can_get_metadata():
    response = requests.get(ENDPOINT + '/model_metadata')
    assert response.status_code == 200

def test_can_make_prediction():
    payload = {
    "Gender":"Male",
    "Country":"United States",
    "Occupation":"Housewife",
    "self_employed": "No",
    "family_history": "No",
    "Days_Indoors": "More than 2 months",
    "Growing_Stress": "No",
    "Changes_Habits": "Yes",
    "Mental_Health_History": "Yes",
    "Mood_Swings": "Medium",
    "Coping_Struggles": "No",
    "Work_Interest": "Maybe",
    "Social_Weakness": "Maybe",
    "care_options": "No"
    }

    response = requests.post(ENDPOINT + '/predict', json=payload)
    assert response.status_code == 200

    # python -m pytest -v
    # -s for print