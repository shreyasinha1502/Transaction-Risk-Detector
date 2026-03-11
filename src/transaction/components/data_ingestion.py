import os
import sys
from dataclasses import dataclass
from src.transaction.logger import logger
from src.transaction.exception import CustomException


@dataclass
class DataIngestionConfig:
    train_path: str = os.path.join("artifacts", "train.csv")
    test_path: str = os.path.join("artifacts", "test.csv")


class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started")

        try:
            if not os.path.exists(self.config.train_path):
                raise FileNotFoundError("train.csv not found")

            if not os.path.exists(self.config.test_path):
                raise FileNotFoundError("test.csv not found")

            logger.info("Train & Test files found")
            logger.info("Data Ingestion completed")

            return self.config.train_path, self.config.test_path

        except Exception as e:
            logger.error("Error in Data Ingestion")
            raise CustomException(e, sys)
