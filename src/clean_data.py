import pandas as pd
from sklearn.preprocessing import LabelEncoder

# read the raw data and store as dataframe
def readData():
    df = pd.read_csv('./data/m_health_dataset.csv')
    return df

# delete all duplicate rows of the dataframe
def deleteDuplicateRows(df):
    df.drop_duplicates(inplace=True)
    return df

# remove columns not wanted for training
def removeColumns(df):
    columns_to_drop = [ 'mental_health_interview','Timestamp'] 
    df_cols_removed = df.drop(columns=columns_to_drop)
    return df_cols_removed

# remove every row that has a nan value somewhere
def removeNanRows(df):
    df.dropna(inplace=True)
    return df

if __name__ == "__main__":
    df = readData()
    df = deleteDuplicateRows(df)
    df = removeColumns(df)
    removeNanRows(df)