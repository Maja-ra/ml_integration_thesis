import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from logger.logger import logging
from exception.exception import customexception
import sys
from pathlib import Path
from box import ConfigBox
from ruamel.yaml import YAML
import numpy as np
import pickle
# from scipy import sparse

yaml = YAML(typ="safe")
yaml_name = "params.yaml"
params = ConfigBox(yaml.load(open(yaml_name, encoding="utf-8")))
y_column = params.data.y_column
random_seed = params.base.random_seed
test_size = params.data_split.test_size

TRAIN_DATA_DIR = Path(params.data.train_data_dir)
TEST_DATA_DIR = Path(params.data.test_data_dir)
DATA_DIR = Path(params.base.data_dir)
MODELS_DIR = Path(params.base.models_dir)
preprocessor_path = MODELS_DIR / Path(params.data.preprocessor)

clean_data_path = DATA_DIR / Path(params.data.clean_data)
test_unscaled_path = DATA_DIR / Path(params.data.test_unscaled)
train_unscaled_path = DATA_DIR / Path(params.data.train_unscaled)
X_test_path = DATA_DIR / Path(params.data.X_test)
y_test_path = DATA_DIR / Path(params.data.y_test)
X_train_path = DATA_DIR / Path(params.data.X_train)
y_train_path = DATA_DIR / Path(params.data.y_train)



def definePipeline(all_columns, categorical_columns):
    try:
        numerical_columns = [col for col in all_columns if col not in categorical_columns]

        numeric_preprocessor = Pipeline(
            steps=[
                ("imputation_mean", SimpleImputer(missing_values=np.nan, strategy="mean")),
                #("scaler", StandardScaler()),
            ]
        )

        categorical_preprocessor = Pipeline(
            steps=[
                (
                    "imputation_constant",
                    SimpleImputer(fill_value="missing", strategy="constant"),
                ),
                ("onehot", OneHotEncoder(sparse_output = False, handle_unknown="ignore")),       # handle_unknown="ignore"
                #("scaler", StandardScaler()),
            ]
        )

        col_transformer = ColumnTransformer(
            transformers = [
                ("categorical",categorical_preprocessor, categorical_columns),
                ("numerical", numeric_preprocessor, numerical_columns),
            ]
        )

        preprocessor = Pipeline(
            steps = [
                ("column transformer", col_transformer),
                ("scaler", StandardScaler()),                   # with_mean=False
            ]
        )

        cat_preprocessor = Pipeline(
            steps=[
                ("onehot", OneHotEncoder(sparse_output = False, handle_unknown="ignore")),       # handle_unknown="ignore"
                ("scaler", StandardScaler()),
            ]
        )

        return cat_preprocessor
    
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   
 

def loadCleanData():
    try:
        logging.info("loading clean data")
        # clean_data_dir = Path("data") / "clean_data"
        # df = pd.read_csv(clean_data_dir / 'clean_data.csv')
        df = pd.read_csv(clean_data_path)
        return df
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   

def encodeCategoricalData(df, categorical_columns):
    try:
        logging.info("encoding categorical data")
        
        for column in categorical_columns:
            encoder = OneHotEncoder(sparse_output = False, handle_unknown="ignore")
            y_encoder = LabelEncoder()
            if column == y_column:
                df[column] = y_encoder.fit_transform(df[column])

            else:
                df[column] = encoder.fit_transform(df[column])

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
def splitTestTrain(X_scaled, y, X):
    try:
        logging.info("splitting data into train and test sets")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = test_size, random_state=random_seed)
        X[y_column] = y
        train_unscaled, test_unscaled = train_test_split(X, test_size = test_size, random_state=random_seed)

        #train_data_dir = Path("data") / "train_data"
        #test_data_dir = Path("data") / "test_data"

        logging.info("saving train data to {TRAIN_DATA_DIR} and test data to {TEST_DATA_DIR}")

        TEST_DATA_DIR.mkdir(exist_ok=True)
        TRAIN_DATA_DIR.mkdir(exist_ok=True)

        #sparse.save_npz(train_data_dir / "X_train.npz", X_train) # is sparse matrix because of preprocessing (if not specified in onehot encoder)
        np.save(X_train_path, X_train)
        y_train.to_csv(y_train_path, index = False)
        train_unscaled.to_csv(train_unscaled_path, index = False)

        np.save(X_test_path, X_test)
        y_test.to_csv(y_test_path, index = False)
        test_unscaled.to_csv(test_unscaled_path, index = False)

    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)   
    
def saveFeatureInfo(X_columns, categorical_columns):
    try:
        with open(yaml_name) as f :
            doc = yaml.load(f)     

        doc["data"]["features_used"] = list(X_columns)
        doc["data"]["num_features"] = len(X_columns)
        doc["data"]["categorical_columns"] = list(categorical_columns)

        with open(yaml_name, 'w',) as f :
            #yaml.dump(feature_info_dict,f) 
            yaml.dump(doc,f) 

        logging.info(f'Written features_used and num_features to params.yaml successfully')
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 



if __name__ == "__main__":
    logging.info("starting data preparation for ml training:")

    df = loadCleanData()

    X = df.drop(y_column, axis=1)
    y = df[y_column]                # 'treatment'

    categorical_columns = X.select_dtypes(include="object").columns

    #X = encodeCategoricalData(X, categorical_columns)

    preprocessor = definePipeline(X.columns, categorical_columns)
    label_encoder = LabelEncoder()
    y = pd.DataFrame(label_encoder.fit_transform(y), columns=[y_column])

    X_scaled = preprocessor.fit_transform(X)

    saveFeatureInfo(X.columns, categorical_columns)   # dynamically updates the features used in training so the info can be accessed for the model

    #X_scaled = scaleData(X)

    print(type(X_scaled))

    splitTestTrain(X_scaled, y, X) #and save data

    
    try:

        MODELS_DIR.mkdir(exist_ok=True)
        logging.info(f"saving preprocessor to path: {preprocessor_path}")
        pickle.dump(preprocessor, open(str(preprocessor_path), 'wb'))
        
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys)  
    

