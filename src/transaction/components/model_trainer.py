import os
import sys
import joblib
import numpy as np
from dataclasses import dataclass

from sklearn.metrics import (
    classification_report,
    precision_recall_curve,
    average_precision_score,
)

from xgboost import XGBClassifier

from src.transaction.logger import logger
from src.transaction.exception import CustomException


# =========================
# CONFIG
# =========================
@dataclass
class ModelTrainerConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")


# =========================
# TRAINER
# =========================
class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        logger.info("Model Training started")

        try:
            # -------------------------
            # handle imbalance
            # -------------------------
            fraud_ratio = np.sum(y_train == 1) / len(y_train)
            scale_pos_weight = (1 - fraud_ratio) / fraud_ratio

            logger.info(f"Fraud ratio: {fraud_ratio:.6f}")
            logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

            # -------------------------
            # model
            # -------------------------
            model = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1
            )

            logger.info("Training XGBoost model")
            model.fit(X_train, y_train)

            # -------------------------
            # evaluation
            # -------------------------
            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred = (y_proba > 0.5).astype(int)

            pr_auc = average_precision_score(y_test, y_proba)

            logger.info(f"PR-AUC Score: {pr_auc:.4f}")
            logger.info("Classification Report:")
            logger.info("\n" + classification_report(y_test, y_pred))

            # -------------------------
            # save model
            # -------------------------
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            joblib.dump(model, self.config.model_path)

            logger.info("Model saved successfully")

            return pr_auc

        except Exception as e:
            logger.error("Error in Model Trainer")
            raise CustomException(e, sys)
