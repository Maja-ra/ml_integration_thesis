from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as rt
from box import ConfigBox
from ruamel.yaml import YAML


yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))

test_data = ["Male","United States","Housewife","No","No","More than 2 months","No","Yes","Yes","Medium","No","Maybe","Maybe","No"]


#app = FastAPI()
model_file = "models/model.onnx"

#session = InferenceSession(model_file, providers=["CPUExecutionProvider"])
session = rt.InferenceSession(model_file, providers=rt.get_available_providers())

def make_inference(input):
    input_name = session.get_inputs()[0].name # "feature_input"
    output_name = session.get_outputs()[0].name # "output_label"

    #pred_onx = session.run(None, {input_name: X_test[0:1].astype(np.float32)})[0] # without preprocessing
    pred = session.run(None, {input_name: [input]})[0]

    return pred

pred_onx = make_inference(test_data)


print(pred_onx)

