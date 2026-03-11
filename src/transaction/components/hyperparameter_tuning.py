import os
import sys
import joblib
import numpy as np
from dataclasses import dataclass

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import average_precision_score, make_scorer

from src.transaction.logger import logger
from src.transaction.exception import CustomException


@dataclass
class HyperparameterTuningConfig:
    tuned_model_path: str = os.path.join("artifacts", "model_tuned.pkl")


class HyperparameterTuning:
    def __init__(self):
        self.config = HyperparameterTuningConfig()

    def initiate_hyperparameter_tuning(self, X_train, y_train):
        logger.info("Hyperparameter tuning started")

        try:
            # handle imbalance
            fraud_ratio = np.sum(y_train == 1) / len(y_train)
            scale_pos_weight = (1 - fraud_ratio) / fraud_ratio

            # base model
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1
            )

            # parameter space (CONTROLLED)
            param_dist = {
                "n_estimators": [200, 300, 400],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9],
            }

            pr_auc_scorer = make_scorer(
                average_precision_score, needs_proba=True
            )

            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dist,
                n_iter=15,              # 🔥 realistic
                scoring=pr_auc_scorer,
                cv=3,
                verbose=1,
                random_state=42,
                n_jobs=-1
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            logger.info(f"Best PR-AUC (CV): {search.best_score_:.4f}")
            logger.info(f"Best Parameters: {search.best_params_}")

            os.makedirs("artifacts", exist_ok=True)
            joblib.dump(best_model, self.config.tuned_model_path)

            logger.info("Tuned model saved")

            return best_model, search.best_params_

        except Exception as e:
            logger.error("Error in Hyperparameter Tuning")
            raise CustomException(e, sys)
