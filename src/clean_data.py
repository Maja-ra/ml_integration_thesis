import pandas as pd
from exception.exception import customexception
from logger.logger import logging
import sys

# read the raw data and store as dataframe
def readData():
    try:
        logging.info("reading data")
        df = pd.read_csv('./data/m_health_dataset.csv')
        return df
    except Exception as e:
        logging.info()
        raise customexception(e,sys)

# delete all duplicate rows of the dataframe
def deleteDuplicateRows(df):
    try:
        logging.info("deleting duplicate rows")
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        logging.info()
        raise customexception(e,sys)

# remove columns not wanted for training
def removeColumns(df):
    try:
        logging.info("removing columns")
        columns_to_drop = [ 'mental_health_interview','Timestamp'] 
        df_cols_removed = df.drop(columns=columns_to_drop)
        return df_cols_removed
    except Exception as e:
        logging.info()
        raise customexception(e,sys)

# remove every row that has a nan value somewhere
def removeNanRows(df):
    try:
        logging.info("removing rows with missing values")
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.info()
        raise customexception(e,sys)

if __name__ == "__main__":
    logging.info("starting data cleaning:")
    df = readData()
    df = deleteDuplicateRows(df)
    df = removeColumns(df)
    removeNanRows(df)