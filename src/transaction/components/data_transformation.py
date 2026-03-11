import os
import sys
import joblib
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from src.transaction.logger import logger
from src.transaction.exception import CustomException


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        logger.info("Data Transformation started")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # drop useless columns
            drop_cols = [
                "Unnamed: 0", "first", "last", "street",
                "city", "cc_num", "trans_num"
            ]

            train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
            test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

            # datetime features
            for df in [train_df, test_df]:
                df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
                df["trans_hour"] = df["trans_date_trans_time"].dt.hour
                df["trans_day"] = df["trans_date_trans_time"].dt.day
                df["trans_month"] = df["trans_date_trans_time"].dt.month
                df.drop(columns=["trans_date_trans_time"], inplace=True)

                df["dob"] = pd.to_datetime(df["dob"])
                df["age"] = (pd.Timestamp.now() - df["dob"]).dt.days // 365
                df.drop(columns=["dob"], inplace=True)

            X_train = train_df.drop("is_fraud", axis=1)
            y_train = train_df["is_fraud"]
            X_test = test_df.drop("is_fraud", axis=1)
            y_test = test_df["is_fraud"]

            num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
            cat_cols = X_train.select_dtypes(include=["object"]).columns

            preprocessor = ColumnTransformer([
                ("num", StandardScaler(), num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
            ])

            X_train_arr = preprocessor.fit_transform(X_train)
            X_test_arr = preprocessor.transform(X_test)

            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(preprocessor, self.config.preprocessor_path)

            logger.info("Preprocessor saved")

            return X_train_arr, X_test_arr, y_train, y_test

        except Exception as e:
            logger.error("Error in Data Transformation")
            raise CustomException(e, sys)
