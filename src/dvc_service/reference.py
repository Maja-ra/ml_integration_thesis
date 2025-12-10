#subfolder of datadirectory eg. "train_data"
from exception.exception import customexception
from logger.logger import logging
from pathlib import Path
import sys
import dvc.api

def getDvcVersion(data_path_string): 
    # get version of dvc tracked data/artifact, path relative to repos 

    repos_path = Path().absolute()
    repos_path = str(repos_path).replace("\\", "/")

    try:
        logging.info(f"Getting dvc reference for {data_path_string}")
        dvc_url = dvc.api.get_url(path = data_path_string, repo = repos_path)
        # print(f" DVC data url for {dataset_name} is: {dvc_url}")

        return dvc_url
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 