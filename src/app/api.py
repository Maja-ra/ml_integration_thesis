from fastapi import FastAPI, Request
from pydantic import BaseModel, create_model, PrivateAttr
import onnxruntime as rt
from box import ConfigBox
from ruamel.yaml import YAML
from uuid import UUID, uuid4
import sys
import os
# getting the name of the directory where the this file is present.
current = os.path.dirname(os.path.realpath(__file__))
# Getting the parent directory name
parent = os.path.dirname(current)
# adding the parent directory to the sys.path.
sys.path.append(parent)
import pandas as pd

from logger.logger import logging
from exception.exception import customexception


yaml = YAML(typ="safe")
params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
num_features = params.data.num_features
features_used = params.data.features_used
categorical_features = list(params.data.categorical_columns)
num_categorical_features = len(categorical_features)
numerical_features = [col for col in features_used if col not in categorical_features]
num_numerical_features = num_features - num_categorical_features

test_data = ["Male","United States","Housewife","No","No","More than 2 months","No","Yes","Yes","Medium","No","Maybe","Maybe","No"]

model_file = "models/model.onnx"

title = "MLIntegrationApp"

description = """
Gives access to ML-Model prediction for masters thesis. 🚀

## Info

Model type is: 
"""


##############################

#session = InferenceSession(model_file, providers=["CPUExecutionProvider"])
try:
    logging.info("starting API APP and inference session")
    session = rt.InferenceSession(model_file, providers=rt.get_available_providers())
except Exception as e:
    logging.error(e)
    raise customexception(e,sys)  

def make_inference(input):
    try:
    
        input_name = session.get_inputs()[0].name # "feature_input"
        output_name = session.get_outputs()[0].name # "output_label"

        #pred_onx = session.run(None, {input_name: X_test[0:1].astype(np.float32)})[0] # without preprocessing
        pred = session.run(None, {input_name: [input]})[0]

        return pred
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)  
    
def savePredInfo(filepath, df):
    pass
    
# pred_onx = make_inference(test_data)
# print(pred_onx)
class DataModel(BaseModel):
    _id = UUID # = PrivateAttr(default_factory=uuid4)
    user: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._id = uuid4()
    Gender: str
    Country: str
    Occupation: str
    self_employed: str
    family_history: str
    Days_Indoors: str
    Growing_Stress: str
    Changes_Habits: str
    Mental_Health_History: str
    Mood_Swings: str
    Coping_Struggles: str
    Work_Interest: str
    Social_Weakness: str
    care_options: str


# DynamicDataModel = create_model(

# )

#################################################

try:
    app = FastAPI(
        title = title,
        description = description,
        version = "0.0.1"
    )
except Exception as e:
    logging.error(e)
    raise customexception(e,sys)  


@app.post("/test/")
def create_foo(data: DataModel):
    print(data._id)
    return data


@app.get('/')
def root():
    return {'message': 'Welcome to the ML API. Documentation at /docs.'}


@app.post("/predict/")
async def create_upload_file(input_data: DataModel, request: Request):
    url = str(request.url)
    logging.info(f"prediction request received at {url} type = {str(request.scope["method"])}")
    try:
        # format unput for model
        input_list = []
        input_data_dict = input_data.dict()
        for feature in list(features_used):
            input_list.append(input_data_dict[feature])

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)  

    result = make_inference(input_list)

    result_dict = {
        "prediction": result.tolist()           #result is np.ndarray -> to list to make iterable
    }

    log_dict = {
        "prediction": result.tolist(),
        "input": input_data_dict       
    }
    try: # save prediction info
        print(input_data_dict)
        log_df = pd.DataFrame(input_data_dict, index=[0])
        log_df["prediction"] = result.tolist()[0]
        log_df["id"] = str(input_data._id)
        print(log_df.head())
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)  

    logging.info("Prediction: %s", log_dict)

    return result_dict

# spelling mistake in string is no problem
# additional unrquired field also not
