import os
import sys
import joblib
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_curve,
    average_precision_score
)

from src.transaction.logger import logger
from src.transaction.exception import CustomException


# =========================
# CONFIG
# =========================
@dataclass
class ModelEvaluationConfig:
    model_path: str = os.path.join("artifacts", "model.pkl")
    evaluation_report_path: str = os.path.join("artifacts", "evaluation_report.csv")


# =========================
# EVALUATOR
# =========================
class ModelEvaluation:
    def __init__(self):
        self.config = ModelEvaluationConfig()

    def initiate_model_evaluation(self, X_test, y_test):
        logger.info("Model Evaluation started")

        try:
            # -------------------------
            # Load model
            # -------------------------
            model = joblib.load(self.config.model_path)
            logger.info("Model loaded successfully")

            # -------------------------
            # Predictions
            # -------------------------
            y_proba = model.predict_proba(X_test)[:, 1]

            pr_auc = average_precision_score(y_test, y_proba)
            logger.info(f"PR-AUC Score: {pr_auc:.4f}")

            # -------------------------
            # Threshold tuning
            # -------------------------
            precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            best_threshold = thresholds[np.argmax(f1_scores)]

            logger.info(f"Best Threshold (F1-optimal): {best_threshold:.4f}")

            # -------------------------
            # Final prediction
            # -------------------------
            y_pred = (y_proba >= best_threshold).astype(int)

            cm = confusion_matrix(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)

            logger.info("Confusion Matrix:")
            logger.info(f"\n{cm}")

            # -------------------------
            # Save report
            # -------------------------
            report_df = pd.DataFrame(report).transpose()
            report_df["pr_auc"] = pr_auc
            report_df["best_threshold"] = best_threshold

            os.makedirs(os.path.dirname(self.config.evaluation_report_path), exist_ok=True)
            report_df.to_csv(self.config.evaluation_report_path, index=True)

            logger.info("Evaluation report saved")

            return {
                "pr_auc": pr_auc,
                "best_threshold": best_threshold,
                "confusion_matrix": cm.tolist()
            }

        except Exception as e:
            logger.error("Error in Model Evaluation")
            raise CustomException(e, sys)
