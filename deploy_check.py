from src.logger.logger import logging
from src.exception.exception import customexception
import os
import sys
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
import json

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
ENV = params.environment.env
RESULTS_DIR = Path(params.base.results_dir)
MONITOR_DIR = Path(params.base.monitor_dir)

model_performance = RESULTS_DIR / Path(params.evaluate.model_performance)
drift_report = MONITOR_DIR / Path(params.monitor.drift_report)

model_baseline = params.environment.model_baseline
drift_threshhold = params.environment.drift_threshhold

def passModelBaselineComparison():
    try:
        with open(model_performance, 'r') as file:
            perf_data = json.load(file)

        f1 = perf_data["metrics"][3]["value"]
        accuracy = perf_data["metrics"][0]["value"]

        if f1 > model_baseline:
            print(f1)
            return True
        else:
            return False    

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   

def passDriftCheck():
    try:
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
            print(drifted_cols)
            return True
        else:
            print(drifted_cols)
            return False

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   
    


if __name__ == "__main__":
    perf_check_bool = passModelBaselineComparison()
    drift_check_bool = passDriftCheck()

    if ENV == "local":
        print(ENV)
        os.system('dvc repro app-local/dvc.yaml')
    else:
        pass

    print(perf_check_bool)
    print(drift_check_bool)