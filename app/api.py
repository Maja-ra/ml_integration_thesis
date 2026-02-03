from uuid import UUID, uuid4
import sys
import os
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel #, create_model, PrivateAttr
import onnxruntime as rt
# from box import ConfigBox                       # docker problem mit import
from ruamel.yaml import YAML
from tracely import init_tracing
#from tracely import trace_event
from tracely import create_trace_event
import pymysql.cursors
from ..src.logger.logger import logging                 # ohne docker ohne punkt
from ..src.exception.exception import customexception

load_dotenv("./environments/prod.env")

EVIDENTLY_API_KEY = os.getenv("EVIDENTLY_API_KEY")
EVIDENTLY_PROJECT_ID = os.getenv("EVIDENTLY_PROJECT_ID")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL")
TRACING_EXPORT_NAME = os.getenv("TRACING_EXPORT_NAME")
SQL_HOST = os.getenv("SQL_HOST")
SQL_USER = os.getenv("SQL_USER")
SQL_PASSWORD = os.getenv("SQL_PASSWORD")
SQL_DATABASE = os.getenv("SQL_DATABASE")
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
learning_rate = params["train"]["learning_rate"]
max_depth = params["train"]["max_depth"]
n_estimators = params["train"]["n_estimators"]
model_type = params["train"]["model_type"]
MODEL_FILE = "./models/production/model_prod.onnx"              # ohne docker:  /models/model.onnx
TITLE = "MLIntegrationApp"
DESCRIPTION = """
Gives access to ML-Model prediction for masters thesis. 🚀

## Info

Model type is:  Gradient Boost
"""

############################## initialize session and tracing

try:
    logging.info("starting API APP and inference session")
    session = rt.InferenceSession(MODEL_FILE, providers=rt.get_available_providers())               # load model and run session
except Exception as e:
    logging.error(e)
    raise customexception(e,sys)  

ML_SESS_ID = 1

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
        # output_name = session.get_outputs()[0].name 

        #pred_onx = session.run(None, {input_name: X_test[0:1].astype(np.float32)})[0] # without preprocessing
        pred = session.run(None, {input_name: [input]})[0]

        return pred
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)  
    
    

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

class ResponseModel(BaseModel):
    prediction: int

class ModelData(BaseModel):
    learning_rate: float
    max_depth: int
    n_estimators: int
    model_type: str

# DynamicDataModel = create_model(

# )

class HealthCheck(BaseModel):

    status: str = "OK"

#################################################

try:
    app = FastAPI(
        title = TITLE,
        description = DESCRIPTION,
        version = "0.0.1"
    )
except Exception as e:
    logging.error(e)
    raise customexception(e,sys)  

origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/test/")
def create_foo(data: DataModel):
    print(data._id)
    return data


@app.get('/')
def root():
    return {'message': 'Welcome to the ML API. Documentation at /docs.'}

@app.get('/features')
async def get_features_options():

    try:

        # Connect to the database
        connection = pymysql.connect(host='host.docker.internal',
                             user=SQL_USER,
                             password=SQL_PASSWORD,
                             database=SQL_DATABASE,
                             cursorclass=pymysql.cursors.DictCursor)

        options = {}
        with connection:
            with connection.cursor() as cursor:
                for feature in features_used:
                    table_name = feature + "_info"
                    sql = f"SELECT * FROM {table_name}"
                    cursor.execute(sql)
                    result = cursor.fetchall()
                    r_list = [row[feature] for row in result]
                    options[feature] = r_list

        # for feature in features_used:
        #     table_name = feature + "_info"
        #     mydb.query(f"SELECT * FROM {table_name}")
        #     r=mydb.store_result()
        #     r = r.fetch_row(maxrows=0)
        #     r_list = [value[0].decode() for value in r]
        #     options[feature] = r_list

        #print(options)

        logging.info("Get features options")

        return options
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)  
    
@app.get('/model_metadata', 
    response_model=ModelData
)
async def get_model_info():
    modelData = ModelData(
    learning_rate = learning_rate,
    max_depth = max_depth,
    n_estimators = n_estimators,
    model_type = model_type
    )

    return modelData

@app.get(
    "/health",
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
async def health():
    # HealthCheck: Returns a JSON response with the health status
    logging.info("Perform health check")
    return HealthCheck(status="OK")

# @app.get('/health-extended')
# async def health_extended():
#     return 200



@app.post("/predict/",
    summary="Perform prediction",
    response_description="Return the result of the prediciton (0 or 1)",
    response_model=ResponseModel,
)
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

    log_dict = {
        "prediction": result.tolist()[0],
        "input": input_data_dict       
    }

    response = ResponseModel(prediction = result.tolist()[0])                   #result is np.ndarray -> to list to make iterable

    trace_id = str(uuid4())

    with create_trace_event("prediction", session_id=trace_id) as event:
        for k in input_data_dict:
            event.set_attribute(k, input_data_dict[k])
        #event.set_attribute("input", pd.DataFrame(input_data_dict, index=[0]))
        event.set_attribute("input_data_id", str(input_data._id))
        event.set_attribute("prediction", result.tolist()[0])
        event.set_attribute("prediction_label", y_column)
        # event.set_attribute("prediction", "0")
        event.set_attribute("ml_session_id", ML_SESS_ID)


    logging.info("Prediction: %s", log_dict)

    return response

# spelling mistake in string is no problem
# additional unrequired field also not
