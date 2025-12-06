import pickle
import pandas as pd
from logger.logger import logging
from exception.exception import customexception
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.pipeline import Pipeline
import sys
from pathlib import Path
from matplotlib import pyplot as plt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType

from ruamel.yaml import YAML
from box import ConfigBox
from pathlib import Path
import shap
import numpy as np
# from scipy import sparse

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
conf_matrix_bool = params.evaluate.conf_matrix
expl_plot_bool = params.evaluate.expl_plot
y_column = params.data.y_column
num_features = params.data.num_features
features_used = params.data.features_used
categorical_features = list(params.data.categorical_columns)
num_categorical_features = len(categorical_features)
numerical_features = []
num_numerical_features = num_features - num_categorical_features


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
        # print("Accuracy:", accuracy_score(y_test, pred))
        # print(classification_report(y_test, pred))

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

def modelExplanation(boost_model, X_test):

    logging.info("creating model explanation")
    try:
        eval_results_dir.mkdir(exist_ok=True)

        explainer = shap.TreeExplainer(boost_model, X_test[:1000])
        # shap_values = explainer.shap_values(X_test[:500])
        shap_values = explainer(X_test[:1000])

        # print(shap_values.shape)
        # print(X_test.shape)

        # Plot feature importance using SHAP values
        # shap.summary_plot(shap_values, X_test, feature_names=features_used, show=False)
        # shap.summary_plot(shap_values, feature_names=features_used, plot_type = 'bar', show=False)

        shap.plots.beeswarm(shap_values, show=False)

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

    # print("Accuracy:", accuracy_score(y_test, pred))
    # print(classification_report(y_test, pred))

    evaluateModel(y_test, pred)

    if conf_matrix_bool:
        createConfusionMatrix(y_test, pred)

    if expl_plot_bool:
        modelExplanation(loaded_model, X_test)

    try:
        preprocessor = pickle.load(open("models/preprocessor.pkl", "rb"))
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   
    
    model_pipeline = Pipeline(steps=[
        ("preprocessor" , preprocessor),
        ("classifier", loaded_model)
    ])


    try:
    # Convert into ONNX format.
        # only model without pipeline
        # initial_type = [("feature_input", FloatTensorType([None, num_features]))]
        # onx = convert_sklearn(loaded_model, initial_types = initial_type)

        # initial_types = []
        # for col in categorical_features:
        #     i_type = (col, StringTensorType([None, 1]))
        #     initial_types.append(i_type)


        if num_numerical_features == 0:
            initial_type = [('string_feature_input', StringTensorType([None, num_categorical_features]))]     # 'string_feature_input'
        else:
            initial_type = [('number_feature_input', FloatTensorType([None, num_numerical_features])),
                ('strfeat', StringTensorType([None, num_categorical_features]))]
        
        onx = convert_sklearn(model_pipeline, initial_types = initial_type)

        with open("models/model.onnx", "wb") as f:
            f.write(onx.SerializeToString())
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   


