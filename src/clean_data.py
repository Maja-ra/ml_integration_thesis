import pandas as pd
from exception.exception import customexception
import sys

# read the raw data and store as dataframe
def readData():
    try:
        df = pd.read_csv('./data/m_health_dataset.csv')
        return df
    except Exception as e:
        raise customexception(e,sys)

# delete all duplicate rows of the dataframe
def deleteDuplicateRows(df):
    try:
        df.drop_duplicates(inplace=True)
        return df
    except Exception as e:
        raise customexception(e,sys)

# remove columns not wanted for training
def removeColumns(df):
    try:
        columns_to_drop = [ 'mental_health_interview','Timestamp'] 
        df_cols_removed = df.drop(columns=columns_to_drop)
        return df_cols_removed
    except Exception as e:
        raise customexception(e,sys)

# remove every row that has a nan value somewhere
def removeNanRows(df):
    try:
        df.dropna(inplace=True)
        return df
    except Exception as e:
        raise customexception(e,sys)

if __name__ == "__main__":
    df = readData()
    df = deleteDuplicateRows(df)
    df = removeColumns(df)
    removeNanRows(df)