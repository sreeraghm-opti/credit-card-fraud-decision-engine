"""
Investigator Case Engine
------------------------

Creates investigator-facing fraud cases from model scores.
"""

import pandas as pd


def assign_risk_band(probability):
    """Assign operational risk band."""

    if probability >= 0.90:
        return "CRITICAL"

    if probability >= 0.75:
        return "HIGH"

    if probability >= 0.56:
        return "MEDIUM"

    return "LOW"


def assign_priority(probability):
    """Assign investigator priority."""

    if probability >= 0.90:
        return "P1 — URGENT"

    if probability >= 0.75:
        return "P2 — HIGH"

    if probability >= 0.56:
        return "P3 — REVIEW"

    return "P4 — LOW"


def create_case_queue(
    scored_transactions,
    reason_codes=None,
):
    """Create an investigator review queue."""

    queue = scored_transactions[
        scored_transactions["Decision"] == "REVIEW"
    ].copy()

    if queue.empty:
        return queue

    queue["Risk_Band"] = queue["Fraud_Probability"].apply(
        assign_risk_band
    )

    queue["Investigation_Priority"] = queue[
        "Fraud_Probability"
    ].apply(assign_priority)

    if reason_codes is not None:
        queue["Reason_Codes"] = reason_codes

    queue = queue.sort_values(
        "Fraud_Probability",
        ascending=False,
    )

    return queue.reset_index(drop=True)
