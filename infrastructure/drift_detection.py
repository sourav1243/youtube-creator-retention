from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)

DRIFT_STATE_FILE = ROOT_DIR / "infrastructure" / "drift_state.json"


def _load_previous_state() -> dict:
    if DRIFT_STATE_FILE.exists():
        with open(DRIFT_STATE_FILE) as f:
            return json.load(f)
    return {}


def _save_state(state: dict):
    DRIFT_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DRIFT_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def detect_feature_drift(
    features_df: pd.DataFrame,
    threshold_std: float = 2.0,
) -> list[str]:
    """Detect if feature distributions have shifted significantly from previous run."""
    previous = _load_previous_state()
    alarms: list[str] = []

    if not previous:
        _save_state({"feature_means": features_df.select_dtypes(include=[np.number]).mean().to_dict()})
        return alarms

    prev_means = previous.get("feature_means", {})
    current = features_df.select_dtypes(include=[np.number]).mean()

    for col, curr_val in current.items():
        prev_val = prev_means.get(col)
        if prev_val is not None and abs(prev_val) > 0:
            z_score = abs(curr_val - prev_val) / max(abs(prev_val), 0.001)
            if z_score > threshold_std:
                msg = f"Drift detected: {col} mean changed from {prev_val:.4f} to {curr_val:.4f} (z={z_score:.2f})"
                alarms.append(msg)
                logger.warning(msg)

    _save_state({"feature_means": current.to_dict()})
    return alarms


def detect_cluster_drift(
    clusters_df: pd.DataFrame,
    window_size: int = 3,
) -> list[str]:
    """Detect if cluster distribution has shifted."""
    alarms: list[str] = []
    prev = _load_previous_state()

    if not prev.get("cluster_distribution"):
        _save_state({
            **prev,
            "cluster_distribution": clusters_df["cluster_label"].value_counts().to_dict(),
        })
        return alarms

    prev_dist = prev.get("cluster_distribution", {})
    curr_dist = clusters_df["cluster_label"].value_counts().to_dict()

    for label, curr_count in curr_dist.items():
        prev_count = prev_dist.get(label, 0)
        total = curr_count + prev_count
        if total > 0:
            change_pct = abs(curr_count - prev_count) / total * 100
            if change_pct > 30:
                msg = f"Cluster drift: {label} changed {prev_count} \u2192 {curr_count} ({change_pct:.0f}%)"
                alarms.append(msg)
                logger.warning(msg)

    _save_state({
        **prev,
        "cluster_distribution": curr_dist,
    })
    return alarms
