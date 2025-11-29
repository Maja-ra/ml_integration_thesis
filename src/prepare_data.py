import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logger.logger import logging
from exception.exception import customexception
import sys
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
import numpy as np

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
y_column = params.data.y_column
random_seed = params.base.random_seed
test_size = params.data_split.test_size

def loadCleanData():
    try:
        logging.info("loading clean data")
        clean_data_dir = Path("data") / "clean_data"
        df = pd.read_csv(clean_data_dir / 'clean_data.csv')
        return df
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   

def encodeCategoricalData(df):
    try:
        logging.info("encoding categorical data")
        categorical_columns = df.select_dtypes(include="object").columns
        for column in categorical_columns:
            label_encoder = LabelEncoder()
            if column == 'treatment':
                df[column] = df[column].replace({'No': 0, 'Yes': 1})

            else:
                df[column] = label_encoder.fit_transform(df[column])

        return df
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   

def scaleData(X):
    try:
        logging.info("scaling the data")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   


# Split the DataFrame into training and testing sets and save them in data
def splitTestTrain(X_scaled, y):
    try:
        logging.info("splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = test_size, random_state=random_seed)

        train_data_dir = Path("data") / "train_data"
        test_data_dir = Path("data") / "test_data"

        logging.info("saving train data to {train_data_dir} and test data to {test_data_dir}")
        train_data_dir.mkdir(exist_ok=True)
        test_data_dir.mkdir(exist_ok=True)

        np.save(train_data_dir / "X_train.npy", X_train) # is numpy array because of scaling
        y_train.to_csv(train_data_dir / "y_train.csv", index_label = False)

        np.save(test_data_dir / "X_test.npy", X_test)
        y_test.to_csv(test_data_dir / "y_test.csv", index_label = False)

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   


if __name__ == "__main__":
    logging.info("starting data preparation for ml training:")

    df = loadCleanData()
    df = encodeCategoricalData(df)

    X = df.drop(y_column, axis=1)
    y = df[y_column]                # 'treatment'

    X_scaled = scaleData(X)

    splitTestTrain(X_scaled, y)
