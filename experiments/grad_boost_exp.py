# At the beginning of your Python script
from dotenv import load_dotenv
import mlflow
import numpy as np
import reusableFunctions

from sklearn.ensemble import GradientBoostingClassifier

MLFLOW_HOST_SERVER, exp_workspace_url = reusableFunctions.initializeMlflowServer()

exp_name = "grad_boost"
model_name = "grad_boost_treatment_model"

if MLFLOW_HOST_SERVER == "databricks":
    exp_name = exp_workspace_url + exp_name

params = {
    "max_depth": 9,
    "n_estimators": 9,
    "learning_rate": 0.1,
    "random_state": reusableFunctions.rand_state,
}


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

    model = GradientBoostingClassifier(n_estimators= params["n_estimators"], learning_rate= params["learning_rate"], max_depth= params["max_depth"], random_state= params["max_depth"])

    reusableFunctions.mlflowTrain(model, clean_data, X_train,  X_test,  y_train,  y_test, params)
