from fastapi import FastAPI, Request
from pydantic import BaseModel, create_model, PrivateAttr
import onnxruntime as rt
# from box import ConfigBox                       # docker problem mit import
from ruamel.yaml import YAML
from uuid import UUID, uuid4
from typing import Optional
import sys
import pandas as pd
from ..src.logger.logger import logging                 # ohne docker ohne punkt
from ..src.exception.exception import customexception
from tracely import init_tracing
from tracely import trace_event
from tracely import create_trace_event
from dotenv import load_dotenv
import os

load_dotenv("./environments/prod.env")

EVIDENTLY_API_KEY = os.getenv("EVIDENTLY_API_KEY")
EVIDENTLY_PROJECT_ID = os.getenv("EVIDENTLY_PROJECT_ID")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL")
TRACING_EXPORT_NAME = os.getenv("TRACING_EXPORT_NAME")

yaml = YAML(typ="safe")

with open("./params.yaml") as f :                 #with open("params.yaml") as f : (ohne Docker)
    params = yaml.load(f)    

num_features = params["data"]["num_features"]
features_used = params["data"]["features_used"]
categorical_features = list(params["data"]["categorical_columns"])
num_categorical_features = len(categorical_features)
numerical_features = [col for col in features_used if col not in categorical_features]
num_numerical_features = num_features - num_categorical_features 
y_column = params["data"]["y_column"]

model_file = "./models/production/model.onnx"              # ohne docker:  /models/model.onnx

title = "MLIntegrationApp"

description = """
Gives access to ML-Model prediction for masters thesis. 🚀

## Info

Model type is: 
"""

############################## initialize session and tracing

try:
    logging.info("starting API APP and inference session")
    session = rt.InferenceSession(model_file, providers=rt.get_available_providers())               # load model and run session
except Exception as e:
    logging.error(e)
    raise customexception(e,sys)  

ml_sess_id = 1

try:
# Initialize tracing
    init_tracing(
        address= EVIDENTLY_URL,              # Trace Collector Address
        api_key= EVIDENTLY_API_KEY,                                         # API Key from Evidently Cloud
        project_id= EVIDENTLY_PROJECT_ID,  # Project ID from Evidently Cloud
        export_name=TRACING_EXPORT_NAME,
    )
except Exception as e:
    logging.error(e)
    raise customexception(e,sys)  

#################################

def make_inference(input):
    try:
    
        input_name = session.get_inputs()[0].name 
        output_name = session.get_outputs()[0].name 

        #pred_onx = session.run(None, {input_name: X_test[0:1].astype(np.float32)})[0] # without preprocessing
        pred = session.run(None, {input_name: [input]})[0]

        return pred
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)  
    
def savePredInfo(filepath, df):
    pass
    

class DataModel(BaseModel):
    _id = UUID # = PrivateAttr(default_factory=uuid4)
    user: Optional[str] = None #str | None = None

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
    method = str(request.scope["method"])
    logging.info(f"prediction request received at {url} type = {method}")
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
        "prediction": result.tolist()[0]           #result is np.ndarray -> to list to make iterable
    }

    log_dict = {
        "prediction": result.tolist()[0],
        "input": input_data_dict       
    }

    trace_id = str(uuid4())

    with create_trace_event("prediction", session_id=trace_id) as event:
        for k in input_data_dict:
            event.set_attribute(k, input_data_dict[k])
        #event.set_attribute("input", pd.DataFrame(input_data_dict, index=[0]))
        event.set_attribute("input_data_id", str(input_data._id))
        event.set_attribute("prediction", result.tolist()[0])
        event.set_attribute("prediction_label", y_column)
        # event.set_attribute("prediction", "0")
        event.set_attribute("ml_session_id", ml_sess_id)


    logging.info("Prediction: %s", log_dict)

    return result_dict

# spelling mistake in string is no problem
# additional unrequired field also not
