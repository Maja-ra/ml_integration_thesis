import pandas as pd
from exception.exception import customexception
from logger.logger import logging
import sys
from ruamel.yaml import YAML
from box import ConfigBox
from pathlib import Path
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from MySQLdb import _mysql

load_dotenv("environments/prod.env")

SQL_HOST = os.getenv("SQL_HOST")
SQL_USER = os.getenv("SQL_USER")
SQL_PASSWORD = os.getenv("SQL_PASSWORD")
SQL_DATABASE = os.getenv("SQL_DATABASE")
SQL_DATABASE_FEATURES = os.getenv("SQL_DATABASE_FEATURES")

yaml = YAML(typ="safe")

params = ConfigBox(yaml.load(open("params.yaml", encoding="utf-8")))
cols_to_drop = params.data.cols_to_drop
y_column = params.data.y_column

DATA_DIR = Path(params.base.data_dir)
CLEAN_DATA_DIR = Path(params.data.clean_data_dir)

clean_data_path = DATA_DIR / Path(params.data.clean_data)
raw_data_path = DATA_DIR / Path(params.data.raw_data)

# read the raw data and store as dataframe
def readData():
    try:
        logging.info("reading data")
        df = pd.read_csv(raw_data_path)
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
        logging.info("saving clean data to {clean_data_dir}")
        CLEAN_DATA_DIR.mkdir(exist_ok=True)
        df.to_csv(clean_data_path, index = False)
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)
    
def saveDataOptionsSQL(cleanData):

    try:
        mydb = _mysql.connect(
            user= SQL_USER,
            password= SQL_PASSWORD,
        )

        #mycursor = mydb.cursor()                                               #using db connector instead of mysqlclient
        #mycursor.execute(f"CREATE DATABASE IF NOT EXISTS {SQL_DATABASE}")
        mydb.query(f"CREATE DATABASE IF NOT EXISTS {SQL_DATABASE}")

        engine_string = f'mysql+mysqldb://{SQL_USER}:{SQL_PASSWORD}@{SQL_HOST}/{SQL_DATABASE}'
        cnx = create_engine(engine_string)  

        cols = []
    
        for column in cleanData.columns:
            if column != y_column:
                data_options_df = pd.DataFrame()
                unique_values = cleanData[column].unique()
                data_options_df[column] = unique_values
                data_options_df.to_sql(str(column).lower() +'_info', cnx, if_exists='replace', index = False)
                cols.append(column)
        
        data_features_df = pd.DataFrame()
        data_features_df["features"] = cols
        data_features_df.to_sql('cols_info', cnx, if_exists='replace', index = False)

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)

def saveDataFeatures(cleanData):
    
    try:
        mydb = _mysql.connect(
            user= SQL_USER,
            password= SQL_PASSWORD,
        )

        #mycursor = mydb.cursor()                                               #using db connector instead of mysqlclient
        #mycursor.execute(f"CREATE DATABASE IF NOT EXISTS {SQL_DATABASE}")
        mydb.query(f"CREATE DATABASE IF NOT EXISTS {SQL_DATABASE_FEATURES}")

        engine_string = f'mysql+mysqldb://{SQL_USER}:{SQL_PASSWORD}@{SQL_HOST}/{SQL_DATABASE_FEATURES}'
        cnx = create_engine(engine_string)  


        cleanData.to_sql('mental_health', cnx, if_exists='replace', index = False)
        


    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)
    

if __name__ == "__main__":
    logging.info("starting data cleaning:")

    df = readData()
    df = deleteDuplicateRows(df)
    df = removeColumns(df, cols_to_drop)
    df = removeNanRows(df)
    saveDataOptionsSQL(df)
    saveCleanData(df)
    saveDataFeatures(df)