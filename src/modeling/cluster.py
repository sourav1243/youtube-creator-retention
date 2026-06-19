from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from src.config import ROOT_DIR, settings

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "upload_freq_30d",
    "upload_freq_90d",
    "freq_trend_ratio",
    "momentum_ratio",
    "avg_engagement_rate",
    "days_since_last_upload",
]

CLUSTER_LABELS = {
    0: "High Momentum — Frequent Uploaders",
    1: "Mid Momentum — Regular Creators",
    2: "Low Momentum — Declining Engagement",
}

RISK_MAP = {
    "High Momentum — Frequent Uploaders": "Healthy",
    "Mid Momentum — Regular Creators": "Watch",
    "Low Momentum — Declining Engagement": "At-Risk",
}


def load_features(parquet_path: str | Path | None = None) -> pd.DataFrame:
    parquet_path = Path(parquet_path or ROOT_DIR / "data" / "processed" / "creator_features.parquet")
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)
    return df


def determine_k(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    output_plot: str | Path | None = None,
) -> tuple[int, dict]:
    inertias: dict[int, float] = {}
    silhouettes: dict[int, float] = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        inertias[k] = float(km.inertia_)
        if k > 1 and len(set(labels)) > 1:
            silhouettes[k] = float(silhouette_score(X, labels))
        else:
            silhouettes[k] = 0.0
        logger.info("  K=%d: inertia=%.2f, silhouette=%.4f", k, inertias[k], silhouettes[k])

    best_k = max(silhouettes, key=silhouettes.get)
    logger.info("Best K by silhouette: %d (score=%.4f)", best_k, silhouettes[best_k])

    if output_plot:
        output_plot = Path(output_plot)
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ks = list(inertias.keys())
        ax1.plot(ks, list(inertias.values()), "bo-")
        ax1.set_xlabel("K")
        ax1.set_ylabel("Inertia")
        ax1.set_title("Elbow Method")

        ax2.plot(ks, [silhouettes.get(k, 0) for k in ks], "ro-")
        ax2.set_xlabel("K")
        ax2.set_ylabel("Silhouette Score")
        ax2.set_title(f"Silhouette Analysis (best K={best_k})")

        plt.tight_layout()
        fig.savefig(str(output_plot), dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Elbow/silhouette plot saved to %s", output_plot)

    return best_k, {"inertias": inertias, "silhouettes": silhouettes}


def label_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    centroids = df.groupby(cluster_col)[feature_cols].mean()
    momentum_rank = centroids["momentum_ratio"].rank(ascending=True)
    freq_rank = centroids["upload_freq_30d"].rank(ascending=True)
    combined_rank = momentum_rank + freq_rank

    sorted_clusters = combined_rank.sort_values().index.tolist()
    n = len(sorted_clusters)

    label_map = {}
    for i, cid in enumerate(sorted_clusters):
        if n == 1:
            label_map[cid] = "Single Cluster — All Creators"
        elif i == 0:
            label_map[cid] = "Low Momentum — Declining Engagement"
        elif i == n - 1:
            label_map[cid] = "High Momentum — Frequent Uploaders"
        else:
            label_map[cid] = "Mid Momentum — Regular Creators"

    df["cluster_label"] = df[cluster_col].map(label_map)
    df["risk_flag"] = df["cluster_label"].map(RISK_MAP).fillna("Watch")
    return df


def fit_and_persist(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    model_path: str | Path | None = None,
    plot_path: str | Path | None = None,
    random_state: int = 42,
) -> tuple[KMeans, RobustScaler, pd.DataFrame, pd.DataFrame]:
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    scored = df[~df["insufficient_history"]].copy()
    unscored = df[df["insufficient_history"]].copy()
    logger.info("Scored: %d, Unscored: %d", len(scored), len(unscored))

    X = scored[feature_cols].fillna(0).values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    best_k, eval_results = determine_k(X_scaled, output_plot=plot_path)

    model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
    scored["cluster_id"] = model.fit_predict(X_scaled)

    scored = label_clusters(scored)

    now = datetime.now(timezone.utc)
    scored["model_version"] = "kmeans_v1"
    scored["scored_at"] = now
    scored["distance_to_centroid"] = model.transform(X_scaled).min(axis=1)

    unscored["cluster_id"] = -1
    unscored["cluster_label"] = "Unscored"
    unscored["risk_flag"] = "Unscored"
    unscored["model_version"] = "kmeans_v1"
    unscored["scored_at"] = now
    unscored["distance_to_centroid"] = np.nan

    result = pd.concat([scored, unscored], ignore_index=True)

    model_path = Path(model_path or ROOT_DIR / "models" / "kmeans_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols, "best_k": best_k}, model_path)
    logger.info("Model persisted to %s", model_path)

    return model, scaler, scored, result


def write_model_card(
    model: KMeans,
    scaler: RobustScaler,
    scored_df: pd.DataFrame,
    eval_results: dict,
    best_k: int,
    feature_cols: list[str] | None = None,
    output_path: str | Path | None = None,
):
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    # Get centroid values
    centroids = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=feature_cols,
    )
    centroids.insert(0, "cluster", range(best_k))

    # Add label and size
    cluster_info = scored_df.groupby("cluster_id").agg(
        label=("cluster_label", "first"),
        risk_flag=("risk_flag", "first"),
        size=("channel_id", "count"),
    ).reset_index()

    lines = [
        "# Model Card: K-Means Clustering (v1)",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Model Overview",
        "",
        f"- Algorithm: K-Means",
        f"- Number of clusters (K): {best_k}",
        f"- Silhouette score: {eval_results['silhouettes'].get(best_k, 'N/A'):.4f}",
        f"- Random state: {model.random_state}",
        f"- Feature columns: {', '.join(feature_cols)}",
        f"- Scaler: RobustScaler (median/IQR — justified: YouTube metrics are outlier-heavy)",
        "",
        "## Cluster Sizes & Labels",
        "",
        "| Cluster | Label | Risk Flag | Size |",
        "|---|---|---|---|",
    ]
    for _, row in cluster_info.iterrows():
        lines.append(f"| {int(row['cluster_id'])} | {row['label']} | {row['risk_flag']} | {row['size']} |")

    lines.extend([
        "",
        "## Centroid Values (unscaled)",
        "",
        "| Feature | " + " | ".join(str(i) for i in range(best_k)) + " |",
        "|---|" + "---|" * best_k,
    ])
    for col in feature_cols:
        vals = " | ".join(f"{centroids.loc[i, col]:.4f}" for i in range(best_k))
        lines.append(f"| {col} | {vals} |")

    lines.extend([
        "",
        "## Labeling Rule",
        "",
        "- Clusters are ranked by combined (momentum_ratio + upload_freq_30d) centroid values.",
        "- Lowest-ranked cluster → 'Low Momentum — Declining Engagement' → At-Risk",
        "- Mid-ranked cluster(s) → 'Mid Momentum — Regular Creators' → Watch",
        "- Highest-ranked cluster → 'High Momentum — Frequent Uploaders' → Healthy",
        "- Channels with insufficient_history → Unscored (not included in clustering)",
    ])

    if output_path is None:
        output_path = ROOT_DIR / "models" / "model_card.md"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    logger.info("Model card written to %s", output_path)


def run_clustering_pipeline(
    parquet_path: str | Path | None = None,
    output_parquet: str | Path | None = None,
    model_path: str | Path | None = None,
    plot_path: str | Path | None = ROOT_DIR / "reports" / "figures" / "elbow_silhouette.png",
    card_path: str | Path | None = None,
) -> pd.DataFrame:
    df = load_features(parquet_path)
    model, scaler, scored_df, result = fit_and_persist(df, plot_path=plot_path, model_path=model_path)

    eval_results = {"silhouettes": {}}
    best_k = model.n_clusters
    if best_k > 1:
        X = scored_df[FEATURE_COLUMNS].fillna(0).values
        X_scaled = RobustScaler().fit_transform(X)
        sil = silhouette_score(X_scaled, scored_df["cluster_id"])
        eval_results["silhouettes"][best_k] = sil

    write_model_card(model, scaler, scored_df, eval_results, best_k, output_path=card_path)

    output_parquet = Path(output_parquet or ROOT_DIR / "data" / "processed" / "creator_clusters.parquet")
    result.to_parquet(output_parquet, index=False)
    logger.info("Cluster results saved to %s", output_parquet)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_clustering_pipeline()
    print("\nCluster distribution:")
    print(result["cluster_label"].value_counts())
