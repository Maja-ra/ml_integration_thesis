from logger.logger import logging
from exception.exception import customexception
from evidently.presets import DataDriftPreset, DataSummaryPreset
from evidently import BinaryClassification
from evidently import Dataset
from evidently import DataDefinition
from evidently import Report
import pandas as pd
from ruamel.yaml import YAML
from box import ConfigBox
from pathlib import Path
import sys
from evidently_service.connect import connectEvidentlyCloud
from dvc_service.reference import getDvcVersion 

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
y_column = params.data.y_column
pred_column = params.data.pred_column
features_used = params.data.features_used
categorical_features = list(params.data.categorical_columns)
numerical_features = [col for col in features_used if col not in categorical_features]


MONITOR_DIR = Path(params.base.monitor_dir)
MONITOR_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(params.base.data_dir)
REF_DATA_DIR = Path(params.data.ref_data_dir)

ref_data_path = DATA_DIR / Path(params.data.ref_data)
current_data_path = DATA_DIR / Path(params.data.train_unscaled)
summary_report_path = MONITOR_DIR / Path(params.monitor.summary_report)
drift_report_path = MONITOR_DIR / Path(params.monitor.drift_report)

ws, evidently_project = connectEvidentlyCloud()

dvc_data_ref = getDvcVersion(str(current_data_path))
dvc_data_ref_ref = getDvcVersion(str(ref_data_path))


def load_data():
    # logging.info("Load data")
    reference_df = pd.read_csv(ref_data_path)
    current_df = pd.read_csv(current_data_path)

    return current_df, reference_df


def prepareEvidentlyDatasets(current_df, reference_df):
    try:
        logging.info("Prepare datasets for monitoring")

        data_definition=DataDefinition(
            classification=[BinaryClassification(
            target=y_column,
            prediction_labels=pred_column)],
            categorical_columns = categorical_features,
            numerical_columns = numerical_features
        )

        current_dataset = Dataset.from_pandas(
            current_df,
            data_definition=data_definition
        )

        ref_dataset = Dataset.from_pandas(
            reference_df,
            data_definition=data_definition
        )

        return current_dataset, ref_dataset
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

# tests for data quality
def createSummaryReport(current_dataset):
    try:
        logging.info("Build Data summary report...")
        summary_report = Report([
        DataSummaryPreset()
        ],
        include_tests=True, 
        tags=["summary", "production"],
        metadata = {
            "data_ref": dvc_data_ref,
        })

        summary = summary_report.run(current_dataset, None)

        summary.save_json(str(summary_report_path))
        ws.add_run(evidently_project.id, summary, include_data=False)
        logging.info(f"Data summary report saved to: {summary_report_path}. Cloud - Evidently project id: {evidently_project.id}")
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 


def createDriftReport(current_dataset, ref_dataset):
    try:
        # Data Drift
        logging.info("Build Data drift report...")
        drift_report = Report([
        DataDriftPreset()
        ],
        include_tests=True,
        tags=["drift", "production"],
        metadata = {
            "data_ref": dvc_data_ref,
            "data_ref_ref": dvc_data_ref_ref,
        })

        drift_eval = drift_report.run(current_data=current_dataset, reference_data=ref_dataset)

        drift_eval.save_json(str(drift_report_path))
        ws.add_run(evidently_project.id, drift_eval, include_data=False)
        logging.info(f"Data drift report saved to: {drift_report_path}. Cloud - Evidently project id: {evidently_project.id}")
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 



if __name__ == "__main__":

    current_df, reference_df = load_data()

    current_dataset, ref_dataset = prepareEvidentlyDatasets(current_df, reference_df)

    createSummaryReport(current_dataset)
    createDriftReport(current_df, reference_df)

    