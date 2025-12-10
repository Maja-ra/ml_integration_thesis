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
import shap
import numpy as np
from evidently import Dataset
from evidently import DataDefinition
from evidently import BinaryClassification
from evidently.presets import ClassificationPreset
from evidently import Report
from evidently_service.connect import connectEvidentlyCloud
from dvc_service.reference import getDvcVersion
# from scipy import sparse

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
conf_matrix_bool = params.evaluate.conf_matrix
expl_plot_bool = params.evaluate.expl_plot
y_column = params.data.y_column
pred_column = params.data.pred_column
num_features = params.data.num_features
features_used = params.data.features_used
categorical_features = list(params.data.categorical_columns)
num_categorical_features = len(categorical_features)
numerical_features = [col for col in features_used if col not in categorical_features]
num_numerical_features = num_features - num_categorical_features


RESULTS_DIR = Path(params.base.results_dir)
RESULTS_DIR.mkdir(exist_ok=True)

DATA_DIR = Path(params.base.data_dir)
MODELS_DIR = Path(params.base.models_dir)
REF_DATA_DIR = Path(params.data.ref_data_dir)

model_file = MODELS_DIR / Path(params.train.model_pkl)
preprocessor_file = MODELS_DIR / Path(params.data.preprocessor)
ref_data_path = DATA_DIR / Path(params.data.ref_data)
model_performance_report_path = RESULTS_DIR / "model_performance.json"

train_unscaled = DATA_DIR / Path(params.data.train_unscaled)
test_unscaled = DATA_DIR / Path(params.data.test_unscaled)
X_test_path = DATA_DIR / Path(params.data.X_test)
y_test_path = DATA_DIR / Path(params.data.y_test)
X_train_path = DATA_DIR / Path(params.data.X_train)
y_train_path = DATA_DIR / Path(params.data.y_train)
ref_data_path = DATA_DIR / Path(params.data.ref_data)
clean_data_path = DATA_DIR / Path(params.data.clean_data)

ws, evidently_project = connectEvidentlyCloud()

dvc_model_ref = getDvcVersion(str(model_file))
dvc_data_ref = getDvcVersion(str(clean_data_path))



def loadTestData():
    try:
        X_test = np.load(X_test_path)
        y_test = pd.read_csv(y_test_path)

        return  X_test,  y_test
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

def loadTrainData():
    try:
        X_train = np.load('./data/train_data/X_train.npy')
        y_train = pd.read_csv('./data/train_data/y_train.csv')

        return  X_train,  y_train
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 
    
def loadUnscaledData():
    try:
        train_unscaled = pd.read_csv('./data/train_data/train_unscaled.csv')
        test_unscaled = pd.read_csv('./data/test_data/test_unscaled.csv')

        return  test_unscaled,  train_unscaled
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

def evaluateModel(y_test,pred_test):
    logging.info("Results: accuracy = " + str(accuracy_score(y_test,pred_test)) + ", f1 = " + str(f1_score(y_test,pred_test)))
    try:
        # print("Accuracy:", accuracy_score(y_test,pred_test))
        # print(classification_report(y_test,pred_test))

        report = classification_report(y_test,pred_test, output_dict=True)
        df_report = pd.DataFrame(report).transpose()

        # RESULTS_DIR.mkdir(exist_ok=True)
        logging.info("saving result metrics to " + str(RESULTS_DIR))
        df_report.to_csv(str(RESULTS_DIR) + "/" + 'evaluation_metrics.csv', index = False)
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

def createConfusionMatrix(y_test,pred_test):
    logging.info("creating confusion matrix")
    try:
        RESULTS_DIR.mkdir(exist_ok=True)

        conf_matrix = confusion_matrix(y_test,pred_test)

        cm_display = ConfusionMatrixDisplay(confusion_matrix = conf_matrix, display_labels = [0, 1])

        cm_display.plot()

        logging.info("saving confusion matrix to " + str(RESULTS_DIR))
        plt.savefig(str(RESULTS_DIR) + "/" + "confusion_matrix.png")
        plt.close()

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

def modelExplanation(boost_model, X_test):

    logging.info("creating model explanation")
    try:
        RESULTS_DIR.mkdir(exist_ok=True)

        explainer = shap.TreeExplainer(boost_model, X_test[:1000])
        # shap_values = explainer.shap_values(X_test[:500])
        shap_values = explainer(X_test[:1000])

        # print(shap_values.shape)
        # print(X_test.shape)

        # Plot feature importance using SHAP values
        # shap.summary_plot(shap_values, X_test, feature_names=features_used, show=False)
        # shap.summary_plot(shap_values, feature_names=features_used, plot_type = 'bar', show=False)

        shap.plots.beeswarm(shap_values, show=False)

        logging.info("saving model explanation to " + str(RESULTS_DIR))
        plt.savefig(str(RESULTS_DIR) + "/" + "model_explanation.png")
        plt.close()
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 
    
def createDataframes(X_test, y_test, pred_test, X_train, y_train, pred_train):  # from np array -> not needed when using categorical, unscaled df
    try:
        # logging.info("Prepare datasets for monitoring")
        test_dataframe = pd.Dataframe(X_test, columns=features_used)
        test_dataframe[y_column] = y_test
        test_dataframe[pred_column] = pred_test

        train_dataframe = pd.Dataframe(X_train, columns=features_used)
        train_dataframe[y_column] = y_train
        train_dataframe[pred_column] = pred_train

        reference_dataframe = train_dataframe.sample(frac=0.3)

        return test_dataframe, train_dataframe, reference_dataframe
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

# def prepareEvidentlyDatasets(test_dataframe, train_dataframe, pred_test, pred_train):
def prepareEvidentlyDatasets(test_dataframe, pred_test):
    try:
        logging.info("Prepare datasets for monitoring")
        test_dataframe[pred_column] = pred_test
        # train_dataframe[pred_column] = pred_train

        data_definition=DataDefinition(
            classification=[BinaryClassification(
            target=y_column,
            prediction_labels=pred_column)],
            categorical_columns = categorical_features,
            numerical_columns = numerical_features
        )
        test_dataset = Dataset.from_pandas(
            test_dataframe,
            data_definition=data_definition
        )

        # train_dataset = Dataset.from_pandas(
        #     train_dataframe,
        #     data_definition=data_definition
        # )

        # reference_dataframe = train_dataframe.sample(frac=0.3)
        # REF_DATA_DIR.mkdir(exist_ok=True)
        # reference_dataframe.to_csv(ref_data_path, index = False)

        reference_dataframe = pd.read_csv(ref_data_path)

        ref_dataset = Dataset.from_pandas(
            reference_dataframe,
            data_definition=data_definition
        )

        return test_dataset, ref_dataset

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 
    
def createAndSaveModelPerformanceReport(test_dataset,ref_dataset):
    try:
        logging.info("Create model performance report")
        model_performance_report = Report([
            ClassificationPreset()
        ],
        include_tests=True,
        tags=["classification", "production"],
        metadata = {
            "model_ref": dvc_model_ref,
            "data_ref": dvc_data_ref,
        })
        eval_report = model_performance_report.run(test_dataset, ref_dataset)

        # Save reports in HTML format
        #model_performance_report.save_html(str(model_performance_report_path))
        eval_report.save_json(str(model_performance_report_path))
        ws.add_run(evidently_project.id, eval_report, include_data=False)   # upload report to cloud

        # print(eval_report.dict())
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 



if __name__ == "__main__":
    # load the model from disk
    logging.info("starting model evaluation:")

    try: 
        loaded_model = pickle.load(open(model_file, 'rb'))
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   
    
    X_test,  y_test = loadTestData()
    # X_train,  y_train = loadTrainData()
    test_unscaled, train_unscaled = loadUnscaledData()

    # pred_train = loaded_model.predict(X_train)
    pred_test = loaded_model.predict(X_test)

    #test_dataset, train_dataset, ref_dataset = prepareEvidentlyDatasets(test_unscaled, train_unscaled, pred_test, pred_train)
    test_dataset, ref_dataset = prepareEvidentlyDatasets(test_unscaled, pred_test) # for evidentely monitoring
    createAndSaveModelPerformanceReport(test_dataset, ref_dataset)

    # print("Accuracy:", accuracy_score(y_test,pred_test))
    # print(classification_report(y_test,pred_test))

    evaluateModel(y_test, pred_test)

    if conf_matrix_bool:
        createConfusionMatrix(y_test, pred_test)

    if expl_plot_bool:
        modelExplanation(loaded_model, X_test)

    try:
        preprocessor = pickle.load(open(preprocessor_file, "rb"))
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


