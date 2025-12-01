# At the beginning of your Python script
from dotenv import load_dotenv
import mlflow
import numpy as np
import reusableFunctions

from sklearn.tree import DecisionTreeClassifier

MLFLOW_HOST_SERVER, exp_workspace_url = reusableFunctions.initializeMlflowServer()

exp_name = "dec_tree"
model_name = "dec_tree_treatment_model"

if MLFLOW_HOST_SERVER == "databricks":
    exp_name = exp_workspace_url + exp_name

params = {
    "max_depth": 10,
    "criterion": "gini",

    "random_state": reusableFunctions.rand_state,
}
#     "max_features": "log2",


# # Test logging to verify connection
# print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
# with mlflow.start_run():
#     print("✓ Successfully connected to MLflow!")


def connectToExperiment(name):
    try:
        print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

        # Set the experiment path in the remote server
        mlflow.set_experiment(name)
    except Exception as e:
        print(e)
    


if __name__ == "__main__":
    print(f"Start Experiment in {exp_name}")
    connectToExperiment(exp_name)

    # Enable autologging for scikit-learn
    # mlflow.sklearn.autolog()

    clean_data = reusableFunctions.loadCleanData()
    X_train,  X_test,  y_train,  y_test = reusableFunctions.loadTrainTestData()

    model = DecisionTreeClassifier(random_state=params["random_state"], max_depth=params["max_depth"], criterion=params["criterion"])

    reusableFunctions.mlflowTrain(model, clean_data, X_train,  X_test,  y_train,  y_test, params)