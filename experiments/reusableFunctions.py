import numpy as np
import pandas as pd

from box import ConfigBox
from ruamel.yaml import YAML
import numpy as np

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
y_column = params.data.y_column
rand_state = params.base.random_seed

from sklearn.metrics import accuracy_score, classification_report

def loadTrainTestData():
    try:
        X_train = np.load('./data/train_data/X_train.npy')
        X_test = np.load('./data/test_data/X_test.npy')
        y_train = pd.read_csv('./data/train_data/y_train.csv')[y_column]
        y_test = pd.read_csv('./data/test_data/y_test.csv')[y_column]

        return  X_train,  X_test,  y_train,  y_test
    except Exception as e:
        print(e)

def evaluateModel(y_test, pred):
    print("Accuracy:", accuracy_score(y_test, pred))
    print(classification_report(y_test, pred))


if __name__ == "__main__":
    print(f"Loading training and test data")

    X_train,  X_test,  y_train,  y_test = loadTrainTestData()

    #print(X_train[0:5])
    #print(y_test.head())
    #print(X_train[0:5])
    #print(y_test.shape)

    print ("Data successfully loaded")
