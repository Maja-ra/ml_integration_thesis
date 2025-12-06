from logger.logger import logging
from exception.exception import customexception
import sys
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import pickle
# from scipy import sparse

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
y_column = params.data.y_column
random_seed = params.base.random_seed
train_params = params.train

def loadCleanData():
    try:
        logging.info("loading clean data")
        clean_data_dir = Path("data") / "clean_data"
        df = pd.read_csv(clean_data_dir / 'clean_data.csv')
        return df
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   
    

def loadTrainData():
    try:
        logging.info("loading training data")
        X_train = np.load('./data/train_data/X_train.npy')
        y_train = pd.read_csv('./data/train_data/y_train.csv')[y_column]

        return  X_train, y_train
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   


if __name__ == "__main__":
    logging.info("starting model training:")

    # clean_data = loadCleanData()
    X_train, y_train = loadTrainData()

    model = GradientBoostingClassifier(n_estimators= train_params.n_estimators, learning_rate= train_params.learning_rate, max_depth= train_params.max_depth, random_state= random_seed)

    try:
        logging.info("training model")
        model.fit(X_train, y_train)
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   


    try:
        folder = Path("models")
        filepath = folder / 'model.pkl'
        folder.mkdir(exist_ok=True)
        logging.info(f"saving model to path: {filepath}")
        pickle.dump(model, open(str(filepath), 'wb'))
        
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)  
    # save the model to disk
