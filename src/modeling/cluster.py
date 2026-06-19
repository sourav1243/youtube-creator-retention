from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.config import ROOT_DIR

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
    "upload_regularity",
]

LABEL_TEMPLATES = [
    "Low Momentum — Declining Engagement",
    "Mid-Low Momentum — Occasional Uploaders",
    "Mid Momentum — Regular Creators",
    "Mid-High Momentum — Active Creators",
    "High Momentum — Frequent Uploaders",
]

RISK_WEIGHTS = {
    "upload_freq_30d": 0.20,
    "freq_trend_ratio": 0.10,
    "momentum_ratio": 0.30,
    "days_since_last_upload": 0.15,
    "avg_engagement_rate": 0.15,
    "upload_regularity": 0.10,
}


def load_features(parquet_path: str | Path | None = None) -> pd.DataFrame:
    parquet_path = Path(parquet_path or ROOT_DIR / "data" / "processed" / "creator_features.parquet")
    df = pd.read_parquet(parquet_path)
    logger.info("Loaded %d rows from %s", len(df), parquet_path)
    return df


def compute_direct_risk_score(df: pd.DataFrame, feature_cols: list[str] | None = None) -> np.ndarray:
    """Compute a 0-1 risk score directly from features, independent of clustering.

    Higher score = higher risk (more likely to churn).
    """
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    weights = {k: RISK_WEIGHTS.get(k, 0.1) for k in feature_cols}
    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    score = np.zeros(len(df), dtype=float)
    for col, weight in weights.items():
        if col not in df.columns:
            continue
        vals = df[col].fillna(0).values.astype(float)
        if col == "days_since_last_upload":
            comp = vals / (vals + 30.0)
        elif col in ("upload_freq_30d", "upload_freq_90d"):
            comp = 1.0 / (1.0 + vals)
        elif col in ("momentum_ratio", "freq_trend_ratio"):
            comp = 1.0 / (1.0 + vals)
        elif col == "avg_engagement_rate":
            comp = 1.0 / (1.0 + vals * 100.0)
        elif col == "upload_regularity":
            comp = vals / (vals + 14.0)
        else:
            comp = vals / (vals + 1.0)
        score += weight * comp

    return np.clip(score, 0.0, 1.0)


def assign_risk_flags(risk_scores: np.ndarray) -> np.ndarray:
    """Assign risk flags based on percentile thresholds for balanced segmentation.

    - Healthy: bottom 30% (lowest risk)
    - At-Risk: top 30% (highest risk)
    - Watch: middle 40%
    """
    sorted_scores = np.sort(risk_scores)
    n = len(sorted_scores)

    low_idx = max(int(np.floor(n * 0.30)) - 1, 0)
    high_idx = min(int(np.ceil(n * 0.70)), n - 1)

    low_thresh = sorted_scores[low_idx]
    high_thresh = sorted_scores[high_idx]

    if low_thresh >= high_thresh:
        mid = float(np.median(risk_scores))
        low_thresh = mid * 0.95
        high_thresh = mid * 1.05

    flags = np.where(
        risk_scores <= low_thresh, "Healthy",
        np.where(risk_scores >= high_thresh, "At-Risk", "Watch"),
    )
    return flags


def label_clusters_by_risk(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    risk_col: str = "risk_score",
) -> pd.DataFrame:
    """Label clusters based on mean risk score of members.

    Highest risk cluster gets 'Low Momentum'.
    Lowest risk cluster gets 'High Momentum'.
    Only sets cluster_label; risk_flag is set separately by assign_risk_flags.
    """
    cluster_risk = df.groupby(cluster_col)[risk_col].mean().sort_values(ascending=False)
    ranked = cluster_risk.index.tolist()
    n = len(ranked)

    label_map = {}
    for i, cid in enumerate(ranked):
        if n >= 5:
            label_map[cid] = LABEL_TEMPLATES[min(i, len(LABEL_TEMPLATES) - 1)]
        elif n == 1:
            label_map[cid] = "Single Cluster — All Creators"
        elif n == 2:
            label_map[cid] = LABEL_TEMPLATES[0] if i == 0 else LABEL_TEMPLATES[-1]
        elif n == 3:
            label_map[cid] = LABEL_TEMPLATES[0] if i == 0 else (LABEL_TEMPLATES[-1] if i == n - 1 else LABEL_TEMPLATES[2])
        elif n == 4:
            mapping = [0, 1, 3, 4]
            label_map[cid] = LABEL_TEMPLATES[mapping[i]]

    df["cluster_label"] = df[cluster_col].map(label_map)
    return df


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    log_cols = [c for c in feature_cols if "days_since" in c]
    std_cols = [c for c in feature_cols if "days_since" not in c]
    transformers = []
    if log_cols:
        transformers.append(("log", FunctionTransformer(_safe_log1p), log_cols))
    if std_cols:
        transformers.append(("scale", RobustScaler(), std_cols))
    return ColumnTransformer(transformers, remainder="passthrough")


def _find_healthy_centroid(model: Any, scored: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if hasattr(model, "cluster_centers_"):
        healthy_cluster = scored[scored["risk_flag"] == "Healthy"]
        if not healthy_cluster.empty:
            healthy_id = int(healthy_cluster["cluster_id"].mode().iloc[0])
            return model.cluster_centers_[healthy_id]
        fallback_id = int(scored.groupby("cluster_id")["momentum_ratio"].mean().idxmax())
        return model.cluster_centers_[fallback_id]
    if hasattr(model, "means_"):
        healthy_cluster = scored[scored["risk_flag"] == "Healthy"]
        if not healthy_cluster.empty:
            healthy_id = int(healthy_cluster["cluster_id"].mode().iloc[0])
            return model.means_[healthy_id]
        fallback_id = int(scored.groupby("cluster_id")["momentum_ratio"].mean().idxmax())
        return model.means_[fallback_id]
    return np.zeros(len(feature_cols))


def _bootstrap_confidence(
    X_scaled: np.ndarray,
    model: Any,
    model_type: str,
    n_iter: int = 500,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    n_samples = X_scaled.shape[0]
    n_clusters = getattr(model, "n_clusters", None)
    if n_clusters is None:
        n_clusters = getattr(model, "n_components", None)

    if n_clusters is None:
        predicted = model.fit_predict(X_scaled) if hasattr(model, "fit_predict") else model.predict(X_scaled)
        n_clusters = len(set(predicted) - {-1})

    if n_clusters is None or n_clusters <= 1:
        return {
            "assignment_probabilities": np.ones((n_samples, 1)) * 0.5,
            "stability_mean": 0.5,
            "stability_std": 0.0,
            "n_iter": n_iter,
        }

    assignment_counts = np.zeros((n_samples, n_clusters))
    valid_iterations = 0

    for iteration in range(n_iter):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_scaled[idx]

        if model_type == "kmeans":
            boot_model = KMeans(n_clusters=n_clusters, random_state=random_state + iteration, n_init=5)
            boot_model.fit(X_boot)
            preds = boot_model.predict(X_scaled)
        elif model_type == "gmm":
            boot_model = GaussianMixture(n_components=n_clusters, random_state=random_state + iteration)
            boot_model.fit(X_boot)
            preds = boot_model.predict(X_scaled)
        elif model_type == "dbscan":
            eps = model.get_params().get("eps", 0.5)
            min_samples = model.get_params().get("min_samples", 5)
            boot_model = DBSCAN(eps=eps, min_samples=min_samples)
            preds = boot_model.fit_predict(X_scaled)
        else:
            continue

        valid = preds != -1
        for c in range(n_clusters):
            assignment_counts[valid & (preds == c), c] += 1
        valid_iterations += 1

    if valid_iterations == 0:
        return {
            "assignment_probabilities": np.ones((n_samples, 1)) * 0.5,
            "stability_mean": 0.5,
            "stability_std": 0.0,
            "n_iter": n_iter,
        }

    assignment_probs = assignment_counts / valid_iterations
    stability = assignment_probs.max(axis=1)
    return {
        "assignment_probabilities": assignment_probs,
        "stability_mean": float(stability.mean()),
        "stability_std": float(stability.std()),
        "n_iter": n_iter,
    }


def select_best_model(
    X: np.ndarray,
    k_min: int = 3,
    k_max: int = 5,
    random_state: int = 42,
    output_plot: str | Path | None = None,
) -> tuple[KMeans | GaussianMixture | DBSCAN, str, int, dict]:
    best_silhouette = -1.0
    best_model = None
    best_type = ""
    best_k = k_min
    all_results: dict = {"kmeans": {}, "gmm": {}, "dbscan": {}}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil = float(silhouette_score(X, labels)) if k > 1 and len(set(labels)) > 1 else 0.0
        all_results["kmeans"][k] = {"model": km, "silhouette": sil}
        logger.info("  KMeans K=%d: silhouette=%.4f", k, sil)
        if sil > best_silhouette:
            best_silhouette = sil
            best_model = km
            best_type = "kmeans"
            best_k = k

    for k in range(k_min, k_max + 1):
        gmm = GaussianMixture(n_components=k, random_state=random_state)
        gmm.fit(X)
        labels = gmm.predict(X)
        sil = float(silhouette_score(X, labels)) if k > 1 and len(set(labels)) > 1 else 0.0
        all_results["gmm"][k] = {"model": gmm, "silhouette": sil}
        logger.info("  GMM K=%d: silhouette=%.4f", k, sil)
        if sil > best_silhouette:
            best_silhouette = sil
            best_model = gmm
            best_type = "gmm"
            best_k = k

    best_db_sil = -1.0
    best_db_model = None
    for eps in [0.3, 0.5, 0.8, 1.0]:
        for min_samples in [2, 3, 5]:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X)
            n_noise = int((labels == -1).sum())
            unique_labels = set(labels) - {-1}
            sil = 0.0
            if len(unique_labels) > 1:
                mask = labels != -1
                if mask.sum() > 1:
                    sil = float(silhouette_score(X[mask], labels[mask]))
            all_results["dbscan"][(eps, min_samples)] = {"model": db, "silhouette": sil, "noise_count": n_noise}
            logger.info("  DBSCAN eps=%.1f min_samples=%d: silhouette=%.4f noise=%d", eps, min_samples, sil, n_noise)
            if sil > best_db_sil:
                best_db_sil = sil
                best_db_model = db

    if best_db_sil > best_silhouette and best_db_model is not None:
        labels = best_db_model.fit_predict(X)
        noise_frac = (labels == -1).mean()
        if noise_frac > 0.2:
            logger.warning("DBSCAN noise ratio %.2f exceeds 20%% — falling back to %s", noise_frac, best_type)
        else:
            best_silhouette = best_db_sil
            best_model = best_db_model
            best_type = "dbscan"
            best_k = len(set(labels) - {-1})

    logger.info("Best model: %s (K=%d, silhouette=%.4f)", best_type, best_k, best_silhouette)

    if output_plot:
        output_plot = Path(output_plot)
        output_plot.parent.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ks = list(range(k_min, k_max + 1))
        km_sils = [all_results["kmeans"][k]["silhouette"] for k in ks]
        gmm_sils = [all_results["gmm"][k]["silhouette"] for k in ks]
        axes[0].plot(ks, km_sils, "bo-", label="KMeans")
        axes[0].plot(ks, gmm_sils, "rs-", label="GMM")
        axes[0].set_xlabel("K")
        axes[0].set_ylabel("Silhouette")
        axes[0].set_title("KMeans vs GMM")
        axes[0].legend()
        axes[0].grid(True)

        db_params = list(all_results["dbscan"].keys())
        db_sils = [all_results["dbscan"][p]["silhouette"] for p in db_params]
        labels_str = [f"eps={p[0]}\nm={p[1]}" for p in db_params]
        axes[1].bar(range(len(db_params)), db_sils, tick_label=labels_str)
        axes[1].set_xticklabels(labels_str, fontsize=7)
        axes[1].set_title("DBSCAN Silhouette")
        axes[1].set_ylabel("Silhouette")

        axes[2].axis("off")
        axes[2].text(
            0.1, 0.5,
            f"Winner: {best_type}\nK={best_k}\nSilhouette={best_silhouette:.4f}",
            fontsize=14, transform=axes[2].transAxes, verticalalignment="center",
        )

        plt.tight_layout()
        fig.savefig(str(output_plot), dpi=100, bbox_inches="tight")
        plt.close(fig)
        logger.info("Model selection plot saved to %s", output_plot)

    return best_model, best_type, best_k, all_results


def fit_and_persist(
    df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    model_path: str | Path | None = None,
    plot_path: str | Path | None = None,
    random_state: int = 42,
    model_type: str = "auto",
    min_clusters: int = 3,
    max_clusters: int = 5,
) -> tuple[any, ColumnTransformer | None, pd.DataFrame, pd.DataFrame, dict]:
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    scored_mask = ~df["insufficient_history"]
    scored = df[scored_mask].copy()
    unscored = df[~scored_mask].copy()
    logger.info("Scored: %d, Unscored: %d", len(scored), len(unscored))

    if scored.empty:
        logger.warning("No scored channels — skipping clustering")
        result = df.copy()
        result["cluster_id"] = -1
        result["cluster_label"] = "Unscored"
        result["risk_flag"] = "Unscored"
        result["risk_score"] = np.nan
        result["model_version"] = "kmeans_v1"
        result["scored_at"] = datetime.now(UTC)
        result["distance_to_centroid"] = np.nan
        result["confidence"] = np.nan
        return None, None, scored, result, {"silhouettes": {}}

    X_raw = scored[feature_cols].fillna(0)
    preprocessor = _build_preprocessor(feature_cols)
    X_scaled = preprocessor.fit_transform(X_raw)

    if model_type == "auto":
        model, selected_type, best_k, eval_results = select_best_model(
            X_scaled,
            k_min=min_clusters,
            k_max=max_clusters,
            random_state=random_state,
            output_plot=plot_path,
        )
    elif model_type == "gmm":
        best_k = max(min_clusters, 2)
        model = GaussianMixture(n_components=best_k, random_state=random_state)
        model.fit(X_scaled)
        selected_type = "gmm"
        eval_results = {"silhouettes": {}}
        if best_k > 1:
            labels = model.predict(X_scaled)
            sil = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0
            eval_results["silhouettes"][best_k] = sil
    elif model_type == "dbscan":
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(X_scaled)
        noise_frac = (labels == -1).mean()
        if noise_frac > 0.2:
            logger.warning("DBSCAN noise ratio %.2f exceeds 20%% — falling back to KMeans", noise_frac)
            best_k = max(min_clusters, 2)
            model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
            model.fit(X_scaled)
            selected_type = "kmeans"
        else:
            selected_type = "dbscan"
            best_k = len(set(labels) - {-1})
            model.n_clusters = best_k
        eval_results = {"silhouettes": {}}
        if best_k > 1:
            mask = labels != -1
            if mask.sum() > 1:
                eval_results["silhouettes"][best_k] = float(silhouette_score(X_scaled[mask], labels[mask]))
    else:
        best_k = max(min_clusters, 2)
        model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
        model.fit(X_scaled)
        selected_type = "kmeans"
        eval_results = {"silhouettes": {}}
        if best_k > 1:
            labels = model.predict(X_scaled)
            eval_results["silhouettes"][best_k] = float(silhouette_score(X_scaled, labels)) if len(set(labels)) > 1 else 0.0

    if selected_type == "dbscan":
        scored["cluster_id"] = model.fit_predict(X_scaled)
        model.n_clusters = int(scored["cluster_id"].nunique())
    else:
        scored["cluster_id"] = model.predict(X_scaled)

    direct_risk = compute_direct_risk_score(scored, feature_cols)
    scored["risk_score"] = direct_risk

    scored = label_clusters_by_risk(scored)

    scored["risk_flag"] = assign_risk_flags(direct_risk)

    healthy_center = _find_healthy_centroid(model, scored, feature_cols)
    if healthy_center is not None and hasattr(model, "cluster_centers_"):
        distances = np.linalg.norm(X_scaled - healthy_center, axis=1)
        scored["distance_to_centroid"] = distances
    else:
        scored["distance_to_centroid"] = np.nan

    boot_results = _bootstrap_confidence(X_scaled, model, selected_type, random_state=random_state)

    probs = boot_results["assignment_probabilities"]
    scored["confidence"] = probs.max(axis=1)[: len(scored)]

    if best_k > 1 and len(set(scored["cluster_id"])) > 1:
        final_sil = float(silhouette_score(X_scaled, scored["cluster_id"]))
    else:
        final_sil = 0.0
    eval_results.setdefault("silhouettes", {})[best_k] = final_sil

    eval_results.update({
        "bootstrap_mean": boot_results["stability_mean"],
        "bootstrap_std": boot_results["stability_std"],
        "bootstrap_n": boot_results["n_iter"],
        "model_type": selected_type,
    })

    now = datetime.now(UTC)
    scored["model_version"] = f"{selected_type}_v1"
    scored["scored_at"] = now

    unscored["cluster_id"] = -1
    unscored["cluster_label"] = "Unscored"
    unscored["risk_flag"] = "Unscored"
    unscored["risk_score"] = np.nan
    unscored["model_version"] = f"{selected_type}_v1"
    unscored["scored_at"] = now
    unscored["distance_to_centroid"] = np.nan
    unscored["confidence"] = np.nan

    result = pd.concat([scored, unscored], ignore_index=True)

    model_path = Path(model_path or ROOT_DIR / "models" / f"{selected_type}_v1.joblib")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "preprocessor": preprocessor, "feature_cols": feature_cols, "best_k": best_k, "eval_results": eval_results}, model_path)
    logger.info("Model persisted to %s", model_path)

    return model, preprocessor, scored, result, eval_results


def write_model_card(
    model: Any,
    preprocessor: ColumnTransformer,
    scored_df: pd.DataFrame,
    eval_results: dict,
    best_k: int,
    feature_cols: list[str] | None = None,
    output_path: str | Path | None = None,
):
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    centroids = scored_df.groupby("cluster_id")[feature_cols].mean().fillna(0).reset_index(drop=True)
    centroids.insert(0, "cluster", range(best_k))

    cluster_info = scored_df.groupby("cluster_id").agg(
        label=("cluster_label", "first"),
        risk_flag=("risk_flag", "first"),
        size=("channel_id", "count"),
        mean_risk_score=("risk_score", "mean"),
    ).reset_index()

    model_type = eval_results.get("model_type", "kmeans")
    sil_scores = eval_results.get("silhouettes", {})
    sil_score = sil_scores.get(best_k, "N/A")
    sil_str = f"{sil_score:.4f}" if isinstance(sil_score, float) else str(sil_score)
    boot_mean = eval_results.get("bootstrap_mean", "N/A")
    boot_str = f"{boot_mean:.4f}" if isinstance(boot_mean, float) else str(boot_mean)

    model_state = str(getattr(model, "random_state", eval_results.get("random_state", "N/A")))

    risk_weights_str = "\n".join(f"  - `{k}`: {v:.0%}" for k, v in sorted(RISK_WEIGHTS.items()))

    lines = [
        f"# Model Card: {model_type.upper()} Clustering",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Model Overview",
        "",
        f"- Algorithm: {model_type.upper()}",
        f"- Number of clusters (K): {best_k}",
        f"- Silhouette score: {sil_str}",
        f"- Bootstrap stability (mean confidence): {boot_str}",
        f"- Random state: {model_state}",
        f"- Feature columns: {', '.join(feature_cols)}",
        "- Preprocessing: RobustScaler on continuous features, log1p on days_since_last_upload",
        "",
        "## Risk Score Formula",
        "",
        "A direct feature-based risk score is computed independently of clustering:",
        "",
        risk_weights_str,
        "",
        "Each component is normalized to [0,1] and weighted. Higher score = higher churn risk.",
        "",
        "## Cluster Sizes & Labels",
        "",
        "| Cluster | Label | Risk Flag | Size | Mean Risk Score |",
        "|---|---|---|---|---|",
    ]
    for _, row in cluster_info.iterrows():
        lines.append(
            f"| {int(row['cluster_id'])} | {row['label']} | {row['risk_flag']} | {int(row['size'])} | {row['mean_risk_score']:.4f} |"
        )

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
        "- Clusters are ranked by mean risk_score of their members.",
        "- Lowest-risk cluster(s) → Healthy",
        "- Mid-risk cluster(s) → Watch",
        "- Highest-risk cluster(s) → At-Risk",
        "- Channels with insufficient_history → Unscored (not included in clustering)",
    ])

    lines.append("")
    lines.append("## Risk Score Weights (JSON)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(RISK_WEIGHTS, indent=2))
    lines.append("```")

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
    model_type: str = "auto",
    min_clusters: int = 3,
    max_clusters: int = 5,
) -> pd.DataFrame:
    df = load_features(parquet_path)
    model, preprocessor, scored_df, result, eval_results = fit_and_persist(
        df, plot_path=plot_path, model_path=model_path,
        model_type=model_type, min_clusters=min_clusters, max_clusters=max_clusters,
    )

    if model is None:
        output_parquet = Path(output_parquet or ROOT_DIR / "data" / "processed" / "creator_clusters.parquet")
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output_parquet, index=False)
        logger.info("Cluster results saved to %s (all unscored)", output_parquet)
        return result

    if hasattr(model, "n_clusters"):
        best_k = model.n_clusters
    elif hasattr(model, "n_components"):
        best_k = model.n_components
    else:
        best_k = scored_df["cluster_id"].nunique()

    write_model_card(model, preprocessor, scored_df, eval_results, best_k, output_path=card_path)

    output_parquet = Path(output_parquet or ROOT_DIR / "data" / "processed" / "creator_clusters.parquet")
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_parquet, index=False)
    logger.info("Cluster results saved to %s", output_parquet)

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_clustering_pipeline()
    print("\nCluster distribution:")
    print(result["cluster_label"].value_counts())
