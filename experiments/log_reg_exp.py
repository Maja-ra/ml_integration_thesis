# At the beginning of your Python script
from dotenv import load_dotenv
import mlflow
import numpy as np
import reusableFunctions

from sklearn.linear_model import LogisticRegression

# Load environment variables from .env file
load_dotenv(".databricks/.databricks.env")

exp_name = "/Users/majathe.rapp@gmail.com/thesis_log_reg"

params = {
    "solver": "lbfgs",
    "max_iter": 10,
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
    mlflow.sklearn.autolog()

    X_train,  X_test,  y_train,  y_test = reusableFunctions.loadTrainTestData()

    try:
        model = LogisticRegression(max_iter=params["max_iter"], random_state=params["random_state"], solver=params["solver"])
        model.fit(X_train, y_train)
    except Exception as e:
        print(e)

    pred = model.predict(X_test)

    reusableFunctions.evaluateModel(y_test, pred)
