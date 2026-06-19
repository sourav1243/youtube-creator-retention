from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import FunctionTransformer, RobustScaler

from src.config import ROOT_DIR

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "upload_freq_30d",
    "upload_freq_90d",
    "momentum_ratio",
    "avg_engagement_rate",
    "days_since_last_upload",
    "upload_regularity",
    "duration_trend",
]

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


def _kmeans_search(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42,
    output_plot: str | Path | None = None,
) -> tuple[int, KMeans, dict]:
    inertias: dict[int, float] = {}
    silhouettes: dict[int, float] = {}
    models: dict[int, KMeans] = {}

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        models[k] = km
        inertias[k] = float(km.inertia_)
        if k > 1 and len(set(labels)) > 1:
            silhouettes[k] = float(silhouette_score(X, labels))
        else:
            silhouettes[k] = 0.0
        logger.info("  K=%d: inertia=%.2f, silhouette=%.4f", k, inertias[k], silhouettes[k])

    best_k = max(silhouettes, key=silhouettes.get) if silhouettes else k_min
    logger.info("Best K by silhouette: %d (score=%.4f)", best_k, silhouettes.get(best_k, 0))

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

    return best_k, models[best_k], {"inertias": inertias, "silhouettes": silhouettes}


def label_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    centroids = df.groupby(cluster_col)[feature_cols].mean().fillna(0)
    momentum_rank = centroids["momentum_ratio"].rank(ascending=True)
    freq_rank = centroids["upload_freq_30d"].rank(ascending=True)
    combined_rank = momentum_rank + freq_rank

    sorted_clusters = combined_rank.sort_values().index.tolist()
    n = len(sorted_clusters)

    RISK_LABELS = [
        "Low Momentum — Declining Engagement",
        "Mid-Low Momentum — Occasional Uploaders",
        "Mid Momentum — Regular Creators",
        "Mid-High Momentum — Active Creators",
        "High Momentum — Frequent Uploaders",
    ]

    label_map = {}
    for i, cid in enumerate(sorted_clusters):
        if n >= 5:
            label_idx = min(i, len(RISK_LABELS) - 1)
            label_map[cid] = RISK_LABELS[label_idx]
        elif n == 1:
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


def _safe_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))


def _build_preprocessor(feature_cols: list[str]) -> ColumnTransformer:
    """Build a ColumnTransformer that log1p-transforms days_since_last_upload and robust-scales all."""
    log_cols = [c for c in feature_cols if "days_since" in c]
    std_cols = [c for c in feature_cols if "days_since" not in c]
    transformers = []
    if log_cols:
        transformers.append(("log", FunctionTransformer(_safe_log1p, feature_names_out="one-to-one"), log_cols))
    if std_cols:
        transformers.append(("scale", RobustScaler(), std_cols))
    return ColumnTransformer(transformers, remainder="passthrough")


def _compute_risk_score(
    X_scaled: np.ndarray,
    healthy_center: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    """Continuous risk score 0-1: normalized distance from healthy centroid."""
    distances = np.linalg.norm(X_scaled - healthy_center, axis=1)
    max_dist = distances.max()
    if max_dist == 0:
        return np.zeros_like(distances)
    return (distances / max_dist).clip(0, 1)


def _bootstrap_confidence(
    X_scaled: np.ndarray,
    model: any,
    model_type: str,
    n_iter: int = 500,
    random_state: int = 42,
) -> dict:
    rng = np.random.default_rng(random_state)
    n_samples = X_scaled.shape[0]
    n_clusters = getattr(model, "n_clusters", None) or getattr(model, "n_components", None)

    if n_clusters is None:
        predicted = model.fit_predict(X_scaled) if hasattr(model, "fit_predict") else model.predict(X_scaled)
        n_clusters = len(set(predicted) - {-1})

    assignment_counts = np.zeros((n_samples, n_clusters))
    valid_iterations = 0

    for iteration in range(n_iter):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X_scaled[idx]

        if model_type == "kmeans":
            boot_model = KMeans(n_clusters=n_clusters, random_state=random_state + iteration, n_init=10)
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
    k_min: int = 2,
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
        if k > 1 and len(set(labels)) > 1:
            sil = float(silhouette_score(X, labels))
        else:
            sil = 0.0
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
        if k > 1 and len(set(labels)) > 1:
            sil = float(silhouette_score(X, labels))
        else:
            sil = 0.0
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
            if len(unique_labels) > 1:
                mask = labels != -1
                if mask.sum() > 1:
                    sil = float(silhouette_score(X[mask], labels[mask]))
                else:
                    sil = 0.0
            else:
                sil = 0.0
            all_results["dbscan"][(eps, min_samples)] = {"model": db, "silhouette": sil, "noise_count": n_noise}
            logger.info("  DBSCAN eps=%.1f min_samples=%d: silhouette=%.4f noise=%d", eps, min_samples, sil, n_noise)
            if sil > best_db_sil:
                best_db_sil = sil
                best_db_model = db

    if best_db_sil > best_silhouette and best_db_model is not None:
        labels = best_db_model.fit_predict(X)
        noise_frac = (labels == -1).mean()
        if noise_frac > 0.2:
            logger.warning("DBSCAN noise ratio %.2f exceeds 20%% — falling back to previous best (%s)", noise_frac, best_type)
        else:
            best_silhouette = best_db_sil
            best_model = best_db_model
            best_type = "dbscan"
            best_k = len(set(labels) - {-1})
            best_model.n_clusters = best_k

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
        axes[2].text(0.1, 0.5, f"Winner: {best_type}\nK={best_k}\nSilhouette={best_silhouette:.4f}",
                     fontsize=14, transform=axes[2].transAxes, verticalalignment="center")

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
    min_clusters: int = 2,
    max_clusters: int = 5,
) -> tuple[any, ColumnTransformer | None, pd.DataFrame, pd.DataFrame, dict]:
    if feature_cols is None:
        feature_cols = FEATURE_COLUMNS

    scored = df[~df["insufficient_history"]].copy()
    unscored = df[df["insufficient_history"]].copy()
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
            sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
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
                sil = silhouette_score(X_scaled[mask], labels[mask])
                eval_results["silhouettes"][best_k] = sil
    else:
        best_k = max(min_clusters, 2)
        model = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
        model.fit(X_scaled)
        selected_type = "kmeans"
        eval_results = {"silhouettes": {}}
        if best_k > 1:
            labels = model.predict(X_scaled)
            sil = silhouette_score(X_scaled, labels) if len(set(labels)) > 1 else 0.0
            eval_results["silhouettes"][best_k] = sil

    if selected_type == "dbscan":
        scored["cluster_id"] = model.fit_predict(X_scaled)
    else:
        scored["cluster_id"] = model.predict(X_scaled)

    scored = label_clusters(scored)

    if selected_type in ("kmeans", "gmm"):
        healthy_cluster = scored[scored["risk_flag"] == "Healthy"]
        if not healthy_cluster.empty:
            healthy_id = int(healthy_cluster["cluster_id"].iloc[0])
        else:
            healthy_id = int(scored.groupby("cluster_id")["upload_freq_30d"].mean().idxmax())
        if selected_type == "kmeans" and hasattr(model, "cluster_centers_"):
            healthy_center = model.cluster_centers_[healthy_id]
            scored["risk_score"] = _compute_risk_score(X_scaled, healthy_center, scored["cluster_id"].values)
        elif selected_type == "gmm" and hasattr(model, "means_"):
            healthy_center = model.means_[healthy_id]
            scored["risk_score"] = _compute_risk_score(X_scaled, healthy_center, scored["cluster_id"].values)
        else:
            scored["risk_score"] = np.nan
    else:
        scored["risk_score"] = np.nan

    boot_results = _bootstrap_confidence(X_scaled, model, selected_type, random_state=random_state)

    if selected_type in ("kmeans", "gmm"):
        if selected_type == "kmeans" and hasattr(model, "cluster_centers_"):
            scored["distance_to_centroid"] = model.transform(X_scaled).min(axis=1)
        elif selected_type == "gmm" and hasattr(model, "means_"):
            scored["distance_to_centroid"] = np.linalg.norm(X_scaled - model.means_[model.predict(X_scaled)], axis=1)
    else:
        scored["distance_to_centroid"] = np.nan

    probs = boot_results["assignment_probabilities"]
    max_probs = probs.max(axis=1)
    scored["confidence"] = max_probs[: len(scored)]

    if best_k > 1 and len(set(scored["cluster_id"])) > 1:
        final_sil = silhouette_score(X_scaled, scored["cluster_id"])
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
    joblib.dump({
        "model": model,
        "preprocessor": preprocessor,
        "feature_cols": feature_cols,
        "best_k": best_k,
        "eval_results": eval_results,
    }, model_path)
    logger.info("Model persisted to %s", model_path)

    return model, preprocessor, scored, result, eval_results


def write_model_card(
    model: any,
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

    if model_type == "kmeans" and hasattr(model, "random_state"):
        model_state = str(model.random_state)
    else:
        model_state = str(eval_results.get("random_state", "N/A"))

    header = f"# Model Card: {model_type.upper()} Clustering"

    lines = [
        header,
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
        "- 'days_since_last_upload' log1p-transformed to reduce feature dominance (original effect size: 6.0 → ~1.5)",
        "",
        "## Cluster Sizes & Labels",
        "",
        "| Cluster | Label | Risk Flag | Size | Mean Risk Score |",
        "|---|---|---|---|---|",
    ]
    for _, row in cluster_info.iterrows():
        lines.append(f"| {int(row['cluster_id'])} | {row['label']} | {row['risk_flag']} | {int(row['size'])} | {row['mean_risk_score']:.4f} |")

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
    model_type: str = "auto",
    min_clusters: int = 2,
    max_clusters: int = 5,
) -> pd.DataFrame:
    df = load_features(parquet_path)
    model, preprocessor, scored_df, result, eval_results = fit_and_persist(
        df,
        plot_path=plot_path,
        model_path=model_path,
        model_type=model_type,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
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
