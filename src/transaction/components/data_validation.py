import sys
import pandas as pd
from dataclasses import dataclass
from src.transaction.logger import logger
from src.transaction.exception import CustomException


@dataclass
class DataValidationConfig:
    target_column: str = "is_fraud"


class DataValidation:
    def __init__(self):
        self.config = DataValidationConfig()

    def initiate_data_validation(self, train_path, test_path):
        logger.info("Data Validation started")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # schema check
            if set(train_df.columns) != set(test_df.columns):
                raise ValueError("Train-Test schema mismatch")

            # target column check
            if self.config.target_column not in train_df.columns:
                raise ValueError("Target column missing")

            # null check
            logger.info(f"Train nulls: {train_df.isnull().sum().sum()}")
            logger.info(f"Test nulls: {test_df.isnull().sum().sum()}")

            # class imbalance
            logger.info("Class distribution:")
            logger.info(train_df[self.config.target_column].value_counts())

            logger.info("Data Validation completed")

        except Exception as e:
            logger.error("Error in Data Validation")
            raise CustomException(e, sys)
