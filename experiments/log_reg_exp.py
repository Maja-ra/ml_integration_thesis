# At the beginning of your Python script

import mlflow
import numpy as np
import reusableFunctions
from mlflow.models import infer_signature
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


MLFLOW_HOST_SERVER, exp_workspace_url = reusableFunctions.initializeMlflowServer()

exp_name = "log_reg"
model_name = "log_reg_treatment_model"

if MLFLOW_HOST_SERVER == "databricks":
    exp_name = exp_workspace_url + exp_name

params = {
    "solver": "lbfgs",
    "max_iter": 100,
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

    dvc_url_clean = reusableFunctions.getDvcDatasetVersion("clean_data", "clean_data.csv")
    dvc_url_train = reusableFunctions.getDvcDatasetVersion("train_data", "X_train.npy")
    dvc_url_test = reusableFunctions.getDvcDatasetVersion("test_data", "X_test.npy")

    dvc_params = {'dvc_clean_data_url': dvc_url_clean, 'dvc_train_data_url': dvc_url_train, 'dvc_test_data_url': dvc_url_test}


    # Log dataset version in MLflow
    # with mlflow.start_run():
    #     mlflow.log_artifact("data/clean_data", artifact_path="datasets")

    # Enable autologging for scikit-learn
    # mlflow.sklearn.autolog()

    clean_data = reusableFunctions.loadCleanData()
    X_train,  X_test,  y_train,  y_test = reusableFunctions.loadTrainTestData()

    dataset, dataset_train, dataset_test = reusableFunctions.mlflowSetDatasets(clean_data, X_train,  X_test,  y_train,  y_test)

    model = LogisticRegression(max_iter=params["max_iter"], random_state=params["random_state"], solver=params["solver"])

    try: 
        with mlflow.start_run():
            model.fit(X_train, y_train)
            mlflow.log_params(dvc_params)
            mlflow.log_params(params)
            mlflow.log_input(dataset, context="clean data")
            mlflow.log_input(dataset_train, context="training")
            mlflow.log_input(dataset_test, context="testing")

            pred = model.predict(X_test)
            reusableFunctions.evaluateModel(y_test, pred) # prints evaluation report

            signature = infer_signature(X_train[:1], y_train[:1])      # creates input example to validate and insure consistent input, X_test, model.predict(X_test)
            print(signature)
            #model_info = 
            #mlflow.sklearn.log_model(model, artifact_path = "log_reg_model", signature=signature, registered_model_name=model_name) #artifact_path=model_name
            # mlflow.sklearn.log_model(
            # model, 
            # artifact_path= model_name, 
            # signature=signature
            # )


            # Build the Evaluation Dataset from the test set
            #eval_data = pd.DataFrame(X_test)
            #print(eval_data.head())
            #print(eval_data.shape)
            #eval_data["label"] = y_test
            #eval_data["predictions"] = pred

            

            eval_params = classification_report(y_test, pred, output_dict=True)

            # Evaluate the static dataset without providing a model -> timeout
            # result = mlflow.evaluate(
            #     data=eval_data,
            #     targets="label",
            #     predictions="predictions",
            #     model_type="classifier",
            # )

            print(eval_params)
            mlflow.log_metrics(eval_params["weighted avg"])
            mlflow.log_metric("accuracy", eval_params["accuracy"])
            #mlflow.log_metrics(result.metrics)

        # model.fit(X_train, y_train)
        # pred = model.predict(X_test)
        # reusableFunctions.evaluateModel(y_test, pred)
        
    except Exception as e:
        print(e)

    # try:

    # except Exception as e:
    #     print(e)



    
