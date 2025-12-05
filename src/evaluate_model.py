import pickle
import numpy as np
import pandas as pd
from logger.logger import logging
from exception.exception import customexception
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from skl2onnx import convert_sklearn, update_registered_converter
from skl2onnx.common.data_types import FloatTensorType

from ruamel.yaml import YAML
from box import ConfigBox
from pathlib import Path
import shap
import csv

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
conf_matrix_bool = params.evaluate.conf_matrix
y_column = params.data.y_column
num_features = params.data.num_features


eval_results_dir = Path("results") / "evaluate"


def loadTestData():
    try:
        X_test = np.load('./data/test_data/X_test.npy')
        y_test = pd.read_csv('./data/test_data/y_test.csv')

        return  X_test,  y_test
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

def evaluateModel(y_test, pred):
    logging.info("Results: accuracy = " + str(accuracy_score(y_test, pred)) + ", f1 = " + str(f1_score(y_test, pred)))
    try:
        #print("Accuracy:", accuracy_score(y_test, pred))
        #print(classification_report(y_test, pred))

        report = classification_report(y_test, pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        eval_results_dir.mkdir(exist_ok=True)
        logging.info("saving result metrics to " + str(eval_results_dir))
        df_report.to_csv(str(eval_results_dir) + "/" + 'evaluation_metrics.csv', index = False)
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

def createConfusionMatrix(y_test, pred):
    logging.info("creating confusion matrix")
    try:
        eval_results_dir.mkdir(exist_ok=True)

        conf_matrix = confusion_matrix(y_test, pred)

        cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1])

        cm_display.plot()

        logging.info("saving confusion matrix to " + str(eval_results_dir))
        plt.savefig(str(eval_results_dir) + "/" + "confusion_matrix.png")
        plt.close()

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

def modelExplanation(ada_model, X_test):

    logging.info("creating model explanation")
    try:
        eval_results_dir.mkdir(exist_ok=True)

        clean_data_dir = Path("data") / "clean_data"
        with open(clean_data_dir / 'clean_data.csv', 'r') as f:
            dict_reader = csv.DictReader(f)

            #get header fieldnames from DictReader and store in list
            headers = dict_reader.fieldnames

        headers.remove(y_column)

        explainer = shap.TreeExplainer(ada_model)
        shap_values = explainer.shap_values(X_test)

        # Plot feature importance using SHAP values
        shap.summary_plot(shap_values, X_test, feature_names=headers, show=False)

        logging.info("saving model explanation to " + str(eval_results_dir))
        plt.savefig(str(eval_results_dir) + "/" + "model_explanation.png")
        plt.close()
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

if __name__ == "__main__":
    # load the model from disk
    logging.info("starting model evaluation:")

    try: 
        filename = 'models/model.pkl'
        loaded_model = pickle.load(open(filename, 'rb'))
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   
    
    X_test,  y_test = loadTestData()

    pred = loaded_model.predict(X_test)

    evaluateModel(y_test, pred)

    if conf_matrix_bool == "yes":
        createConfusionMatrix(y_test, pred)

    modelExplanation(loaded_model, X_test)

    try:
    # Convert into ONNX format.
        num_features = num_features
        initial_type = [("feature_input", FloatTensorType([None, num_features]))]
        onx = convert_sklearn(loaded_model, initial_types = initial_type)

        with open("models/model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   


