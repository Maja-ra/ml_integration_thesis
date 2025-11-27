import pandas as pd
from exception.exception import customexception
from logger.logger import logging
import sys
from ruamel.yaml import YAML
from box import ConfigBox
from pathlib import Path

yaml = YAML(typ="safe")

# read the raw data and store as dataframe
def readData():
    try:
        logging.info("reading data")
        df = pd.read_csv('./data/m_health_dataset.csv')
        return df
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)

# delete all duplicate rows of the dataframe
def deleteDuplicateRows(df):
    try:
        logging.info("deleting duplicate rows")
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)

# remove columns not wanted for training
def removeColumns(df, columns_to_drop):
    try:
        logging.info("removing columns")
        df_cols_removed = df.drop(columns=columns_to_drop)
        return df_cols_removed
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)

# remove every row that has a nan value somewhere
def removeNanRows(df):
    try:
        logging.info("removing rows with missing values")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)

def saveCleanData(df):
    try:
        clean_data_dir = Path("data") / "clean_data"
        logging.info("saving clean data to {clean_data_dir}")
        clean_data_dir.mkdir(exist_ok=True)
        df.to_csv(clean_data_dir)
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)


if __name__ == "__main__":
    logging.info("starting data cleaning:")
    params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
    cols_to_drop = params.data.cols_to_drop
    df = readData()
    df = deleteDuplicateRows(df)
    df = removeColumns(df, cols_to_drop)
    df = removeNanRows(df)
    saveCleanData(df)