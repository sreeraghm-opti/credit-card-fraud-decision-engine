"""
Fraud Decision Engine
---------------------

Converts model probabilities into operational decisions using
a frozen threshold and investigator capacity.
"""

import numpy as np


DEFAULT_THRESHOLD = 0.56


def apply_threshold(probabilities, threshold=DEFAULT_THRESHOLD):
    """Apply a fixed fraud-review threshold."""

    probabilities = np.asarray(probabilities)

    return np.where(
        probabilities >= threshold,
        "REVIEW",
        "APPROVE"
    )


def calculate_expected_cost(
    y_true,
    decisions,
    fraud_cost=5000,
    review_cost=150,
):
    """
    Calculate operational decision cost.

    False negatives:
        Actual fraud incorrectly approved.

    False positives:
        Legitimate transactions sent for review.
    """

    y_true = np.asarray(y_true)
    decisions = np.asarray(decisions)

    predicted_review = decisions == "REVIEW"
    actual_fraud = y_true == 1

    false_negatives = (~predicted_review & actual_fraud).sum()
    false_positives = (predicted_review & ~actual_fraud).sum()

    total_cost = (
        false_negatives * fraud_cost
        + false_positives * review_cost
    )

    return {
        "false_negatives": int(false_negatives),
        "false_positives": int(false_positives),
        "fraud_loss": float(false_negatives * fraud_cost),
        "review_cost": float(false_positives * review_cost),
        "total_cost": float(total_cost),
    }


def enforce_investigator_capacity(
    probabilities,
    capacity=50,
):
    """
    Send at most `capacity` highest-risk transactions
    to investigators.
    """

    probabilities = np.asarray(probabilities)

    decisions = np.array(
        ["APPROVE"] * len(probabilities),
        dtype=object,
    )

    if capacity <= 0:
        return decisions

    capacity = min(capacity, len(probabilities))

    top_indices = np.argsort(probabilities)[-capacity:]

    decisions[top_indices] = "REVIEW"

    return decisions
