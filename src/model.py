"""
Fraud Detection Model
---------------------
Production-style utilities for training and scoring the frozen
class-weighted XGBoost fraud model.
"""

import numpy as np
import pandas as pd
import xgboost as xgb


RANDOM_STATE = 42

MODEL_CONFIG = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def calculate_scale_pos_weight(y):
    """Calculate negative-to-positive class ratio."""
    negative = (y == 0).sum()
    positive = (y == 1).sum()

    if positive == 0:
        raise ValueError("No positive fraud observations found.")

    return negative / positive


def train_weighted_xgboost(X_train, y_train):
    """Train the class-weighted XGBoost model."""

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    model = xgb.XGBClassifier(
        **MODEL_CONFIG,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    return model


def predict_fraud_probability(model, X):
    """Return fraud probabilities."""
    return model.predict_proba(X)[:, 1]


def classify_risk(probability, threshold=0.56):
    """Convert fraud probability into a binary decision."""

    return np.where(
        probability >= threshold,
        "REVIEW",
        "APPROVE"
    )


def score_transactions(model, X, threshold=0.56):
    """Score transactions and generate decisions."""

    probabilities = predict_fraud_probability(model, X)

    return pd.DataFrame({
        "Fraud_Probability": probabilities,
        "Decision": classify_risk(probabilities, threshold),
    })
