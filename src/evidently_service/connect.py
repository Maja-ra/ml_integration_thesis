from evidently.ui.workspace import CloudWorkspace
import os
from dotenv import load_dotenv
from logger.logger import logging
from exception.exception import customexception
import sys

load_dotenv("environments/prod.env")

EVIDENTLY_API_KEY = os.getenv("EVIDENTLY_API_KEY")
EVIDENTLY_PROJECT_ID = os.getenv("EVIDENTLY_PROJECT_ID")
EVIDENTLY_URL = os.getenv("EVIDENTLY_URL")


def connectEvidentlyCloud():
    try:
        logging.info(f"Connecting to evidently cloud: {EVIDENTLY_URL}")
        ws = CloudWorkspace(
        token= EVIDENTLY_API_KEY,
        url= EVIDENTLY_URL)

        project = ws.get_project(EVIDENTLY_PROJECT_ID)

        return ws, project
    except Exception as e:
        logging.error(e)
        raise customexception(e,sys) 

if __name__=="__main__":
    try:
        connectEvidentlyCloud()
        # print(EVIDENTLY_API_KEY)

    except Exception as e:
        print(e)