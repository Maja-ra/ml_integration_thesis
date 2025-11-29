import numpy as np
import pandas as pd

def loadData():
    try:
        X_train = np.load('./data/train_data/X_train.npy')
        X_test = np.load('./data/test_data/X_test.npy')
        y_train = pd.read_csv('./data/train_data/y_train.csv')
        y_test = pd.read_csv('./data/test_data/y_test.csv')

        return  X_train,  X_test,  y_train,  y_test
    except Exception as e:
        print(e)
    

if __name__ == "__main__":
    print(f"Loading training and test data")

    X_train,  X_test,  y_train,  y_test = loadData()

    #print(X_train[0:5])
    #print(y_test.head())

    print ("Data successfully loaded")
