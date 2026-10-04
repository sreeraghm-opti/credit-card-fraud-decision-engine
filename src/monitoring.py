"""
Model Monitoring
----------------

Population Stability Index (PSI) utilities for monitoring
feature and model-score distribution changes.
"""

import numpy as np


def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index."""

    expected = np.asarray(expected)
    actual = np.asarray(actual)

    edges = np.quantile(
        expected,
        np.linspace(0, 1, bins + 1),
    )

    edges = np.unique(edges)

    if len(edges) < 3:
        return 0.0

    expected_counts, _ = np.histogram(
        expected,
        bins=edges,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=edges,
    )

    expected_pct = expected_counts / max(
        expected_counts.sum(),
        1,
    )

    actual_pct = actual_counts / max(
        actual_counts.sum(),
        1,
    )

    epsilon = 1e-6

    expected_pct = np.clip(
        expected_pct,
        epsilon,
        None,
    )

    actual_pct = np.clip(
        actual_pct,
        epsilon,
        None,
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(actual_pct / expected_pct)
    )

    return float(psi)


def classify_psi(psi):
    """Classify PSI according to project thresholds."""

    if psi < 0.10:
        return "Stable"

    if psi < 0.25:
        return "Moderate Drift"

    return "Significant Drift"


def monitor_model_scores(train_scores, current_scores):
    """Monitor model-score distribution."""

    psi = calculate_psi(
        train_scores,
        current_scores,
    )

    return {
        "PSI": psi,
        "Status": classify_psi(psi),
    }
