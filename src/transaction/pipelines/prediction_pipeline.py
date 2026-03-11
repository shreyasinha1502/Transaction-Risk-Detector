import os
import joblib
import numpy as np
import pandas as pd

from src.transaction.logger import logger
from src.transaction.exception import CustomException
import sys


class PredictionPipeline:
    def __init__(self):
        try:
            self.model_path = os.path.join("artifacts", "model.pkl")
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            self.model = joblib.load(self.model_path)
            self.preprocessor = joblib.load(self.preprocessor_path)

            # tuned threshold
            self.threshold = 0.97807443

            logger.info("Prediction pipeline initialized")

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_df: pd.DataFrame):
        try:
            logger.info("Starting prediction")

            transformed_data = self.preprocessor.transform(input_df)

            probs = self.model.predict_proba(transformed_data)[:, 1]

            predictions = (probs >= self.threshold).astype(int)

            result = input_df.copy()
            result["fraud_probability"] = probs
            result["prediction"] = predictions

            return result

        except Exception as e:
            raise CustomException(e, sys)
