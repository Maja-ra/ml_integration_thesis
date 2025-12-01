import numpy as np
import pandas as pd

from box import ConfigBox
from ruamel.yaml import YAML
import numpy as np
import dvc.api
import mlflow
from pathlib import Path
import dagshub
from dotenv import load_dotenv
import os


yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
y_column = params.data.y_column
rand_state = params.base.random_seed

from sklearn.metrics import accuracy_score, classification_report


def initializeMlflowServer():
    try:
        load_dotenv("environments/dev.env")

        MLFLOW_HOST_SERVER = os.getenv("MLFLOW_HOST_SERVER")
    except Exception as e:
        print(e)

    # print(MLFLOW_HOST_SERVER)

    try:

        if MLFLOW_HOST_SERVER == "databricks":

            # Load environment variables from .env file
            load_dotenv(".databricks/.databricks.env")

            #exp_name = "/Users/majathe.rapp@gmail.com/thesis_log_reg"
            exp_workspace_url = "/Users/majathe.rapp@gmail.com/"

            print("Mlflow server set to Databricks")


        elif MLFLOW_HOST_SERVER == "dagshub":

            exp_workspace_url = 'ml_integration_thesis'

            dagshub.init(repo_owner='Maja-ra', repo_name=exp_workspace_url, mlflow=True)
            print("Mlflow server set to Dagshub")

    except Exception as e:
        print(e)

    return MLFLOW_HOST_SERVER, exp_workspace_url



def loadTrainTestData():
    try:
        X_train = np.load('./data/train_data/X_train.npy')
        X_test = np.load('./data/test_data/X_test.npy')
        y_train = pd.read_csv('./data/train_data/y_train.csv')[y_column]
        y_test = pd.read_csv('./data/test_data/y_test.csv')[y_column]

        return  X_train,  X_test,  y_train,  y_test
    except Exception as e:
        print(e)

def loadCleanData():
    try:
        clean_data = pd.read_csv('./data/clean_data/clean_data.csv', index_col=None)

        return  clean_data
    except Exception as e:
        print(e)

#subfolder of datadirectory eg. "train_data"
def getDvcDatasetVersion(subfolder_pathstring, dataset_name):
    # Fetch dataset version from DVC
    if subfolder_pathstring == "":
        data_path_string = "data/" + dataset_name
    else:
        data_path_string = "data/" + subfolder_pathstring + "/" + dataset_name

    repos_path = Path().absolute()
    repos_path = str(repos_path).replace("\\", "/")

    try:
        dvc_url = dvc.api.get_url(path = data_path_string, repo = repos_path)
        print(f" DVC data url for {dataset_name} is: {dvc_url}")

        return dvc_url
    except Exception as e:
        print(e)

    

    #print(data_path_string)
    #print(repos_path)
    
    
def mlflowSetDatasets(clean_data_df, X_train, X_test, y_train, y_test):
    cols = list(clean_data_df.columns)
    # print(clean_data_df.head())
    # print(type(cols))

    cols.remove("treatment")

    # print(cols)
    # print(len(cols))
    # print(X_train.shape)
    # print(X_train[0:5])
    # print(y_train.value_counts())
    # print(y_train.shape)
    
    train_data_df = pd.DataFrame(X_train, columns=cols)
    train_data_df["treatment"] = y_train

    test_data_df = pd.DataFrame(X_test, columns=cols)
    test_data_df["treatment"] = y_test

    # print(train_data_df.shape)
    # print(train_data_df.head())

    # print(train_data_df.head())

    dataset = mlflow.data.from_pandas(clean_data_df,
                                      source="data/clean_data/clean_data.csv", 
                                      name="Mental Health Dataset local - Cleaned", 
                                      targets=y_column
                                      )
    
    dataset_train = mlflow.data.from_pandas(train_data_df, 
                                            source = "data/train_data/X_train.npy",
                                            name="Training dataset", 
                                            targets=y_column
                                            )

    dataset_test = mlflow.data.from_pandas(test_data_df, 
                                            source = "data/test_data/X_test.npy",
                                            name="Testing dataset", 
                                            targets=y_column
                                            )    
    
    
    return dataset, dataset_train, dataset_test
    # # Log dataset version in MLflow
    # mlflow.log_artifact(dataset_path, artifact_path="datasets")

def evaluateModel(y_test, pred):
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))




if __name__ == "__main__":
    print(f"Loading training and test data")

    #X_train,  X_test,  y_train,  y_test = loadTrainTestData()

    #print(X_train[0:5])
    #print(y_test.head())
    #print(X_train[0:5])
    #print(y_test.shape)

    print ("Data successfully loaded")

    #print(getDvcDatsetVersion("train_data", "X_train.npy"))

    #clean_data = loadCleanData()
    #X_train,  X_test,  y_train,  y_test = loadTrainTestData()

    #dataset, dataset_train = 
    #dataset, dataset_train, dataset_test = mlflowSetDatasets(clean_data, X_train, y_train, X_test, y_test)

    print("datasets defined")


    server, exp_workspace_url = initializeMlflowServer()

    #print(server)
    #print(exp_workspace_url)

