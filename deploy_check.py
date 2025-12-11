from src.logger.logger import logging
from src.exception.exception import customexception
import os
import sys
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
import json
import shutil

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
ENV = params.environment.env
RESULTS_DIR = Path(params.base.results_dir)
MONITOR_DIR = Path(params.base.monitor_dir)
PROD_MODEL_DIR = Path(params.base.models_dir) / Path(params.environment.prod_model_path)

PROD_MODEL_DIR.mkdir(exist_ok=True)

prod_model_src_path = Path(params.base.models_dir) /params.environment.prod_model
prod_model_dest_path = PROD_MODEL_DIR /params.environment.prod_model

model_performance = RESULTS_DIR / Path(params.evaluate.model_performance)
drift_report = MONITOR_DIR / Path(params.monitor.drift_report)

model_baseline = params.environment.model_baseline
drift_threshhold = params.environment.drift_threshhold

def passModelBaselineComparison():
    try:
        logging.info(f"Check model performance. Baseline: {model_baseline}")
        with open(model_performance, 'r') as file:
            perf_data = json.load(file)

        f1 = perf_data["metrics"][3]["value"]
        accuracy = perf_data["metrics"][0]["value"]

        if f1 > model_baseline:
            logging.info("assessment passed")
            return True
        else:
            logging.info(f"assessment failed. Value: {f1}")
            return False    

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   

def passDriftCheck():
    try:
        logging.info(f"Checking drift. Threshhold: {drift_threshhold}")
        with open(drift_report, 'r') as file:
            drift_data = json.load(file)

        drifted_cols = []

        for dict in drift_data["metrics"]:
            if type(dict["value"]) == float and dict["value"] >= 0.003: #drift_threshhold:
                column = dict["config"]["column"]
                drift_value = dict["value"]
                drifted_cols.append({
                    column: drift_value
                })
            else:
                pass

        if drifted_cols == []:
            logging.info("No drift detected")
            return True
        else:
            drifted_cols_str = str(drifted_cols)
            logging.info(f"Drift detected in columns {drifted_cols_str}")
            return False

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   
    


if __name__ == "__main__":
    perf_check_bool = passModelBaselineComparison()
    drift_check_bool = passDriftCheck()

    if ENV == "local":
        os.system('dvc repro app-local/dvc.yaml')

    if ENV == "production":

        if drift_check_bool != True  and perf_check_bool == True:
            logging.info("Redeploying model due to drift...")
            
            shutil.copyfile(prod_model_src_path, prod_model_dest_path)

            os.system('dvc repro app/dvc.yaml')
            #os.system('dvc repro app/dvc.yaml:run-container')
    else:
        pass
