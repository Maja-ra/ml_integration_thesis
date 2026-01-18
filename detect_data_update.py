from src.logger.logger import logging
from src.exception.exception import customexception
import os
import sys


def checkChange():
    return False


if __name__ == "__main__":
    change = checkChange()

    if change == False:
        logging.info("No change detected")
        pass
    else:
        logging.info("Data change detected. Rerun Pipeline.")
        os.system('dvc repro')