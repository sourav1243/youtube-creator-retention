"""Deep analysis of data quality and methodology."""
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from src.config import ROOT_DIR

features = pd.read_parquet(ROOT_DIR / "data" / "processed" / "creator_features.parquet")
clusters = pd.read_parquet(ROOT_DIR / "data" / "processed" / "creator_clusters.parquet")
videos = pd.read_parquet(ROOT_DIR / "data" / "processed" / "videos_clean.parquet")

# Align by merging features and clusters
merged_df = features.merge(clusters, on="channel_id", suffixes=("", "_cluster"))
scored = merged_df[~merged_df["insufficient_history"]]

print("=== DATA QUALITY REPORT ===")
print(f"\nTotal channels: {len(features)}")
print(f"Scored channels: {len(scored)}")
print(f"Unscored channels: {features['insufficient_history'].sum()}")

# Feature columns
feat_cols = [
    "upload_freq_30d",
    "upload_freq_90d",
    "freq_trend_ratio",
    "momentum_ratio",
    "avg_engagement_rate",
    "days_since_last_upload",
    "upload_regularity",
    "duration_trend"
]

print(f"\n--- Feature Statistics (scored only) ---")
for col in feat_cols:
    if col in scored.columns:
        s = scored[col].dropna()
        nulls = scored[col].isna().sum()
        print(f"{col:30s}: n={len(s):3d}  nulls={nulls:2d}  mean={s.mean():.4f}  median={s.median():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")

print(f"\n--- Per-Channel Video Coverage ---")
vc = videos.groupby("channel_id").agg(
    n_videos=("video_id", "count"),
    first_upload=("published_at", "min"),
    last_upload=("published_at", "max"),
).reset_index()
vc["days_span"] = (vc["last_upload"] - vc["first_upload"]).dt.days
print(f"Videos per channel:  min={vc['n_videos'].min()}  median={vc['n_videos'].median():.0f}  mean={vc['n_videos'].mean():.0f}  max={vc['n_videos'].max()}")
print(f"Data span (days):    min={vc['days_span'].min()}  median={vc['days_span'].median():.0f}  mean={vc['days_span'].mean():.0f}  max={vc['days_span'].max()}")
print(f"Channels with <10 videos:  {(vc['n_videos'] < 10).sum()}")
print(f"Channels with <30d span:   {(vc['days_span'] < 30).sum()}")

print(f"\n--- Cluster Quality ---")
scored_clusters = clusters[clusters["risk_flag"] != "Unscored"].copy()
for cid in sorted(scored_clusters["cluster_id"].unique()):
    sub = scored_clusters[scored_clusters["cluster_id"] == cid]
    print(f"Cluster {int(cid)} ({sub['cluster_label'].iloc[0]}): n={len(sub)}")
    for col in feat_cols:
        if col in sub.columns:
            v = sub[col].mean()
            print(f"   {col:30s}: {v:.4f}")

print(f"\n--- Feature Correlation Matrix ---")
corr_cols = [c for c in feat_cols if c in scored.columns]
corr = scored[corr_cols].corr()
print(corr.to_string())

print(f"\n--- Distance to Centroid Analysis ---")
dist = clusters[clusters["distance_to_centroid"].notna()]["distance_to_centroid"]
if not dist.empty:
    print(f"Distance stats: mean={dist.mean():.4f}  median={dist.median():.4f}  std={dist.std():.4f}  min={dist.min():.4f}  max={dist.max():.4f}")
    far = clusters[clusters["distance_to_centroid"].notna()].nlargest(5, "distance_to_centroid")
    print("\nChannels farthest from centroid:")
    for _, r in far.iterrows():
        print(f"  {r['channel_id']:30s} dist={r['distance_to_centroid']:.4f}  label={r['cluster_label']}")

print(f"\n--- At-Risk Channels Detail ---")
at_risk = clusters[clusters["risk_flag"] == "At-Risk"]
for _, r in at_risk.iterrows():
    # Find matching features
    feat_row = scored[scored["channel_id"] == r["channel_id"]]
    days = feat_row["days_since_last_upload"].iloc[0] if not feat_row.empty else "N/A"
    freq = feat_row["upload_freq_30d"].iloc[0] if not feat_row.empty else "N/A"
    mom = feat_row["momentum_ratio"].iloc[0] if not feat_row.empty else "N/A"
    eng = feat_row["avg_engagement_rate"].iloc[0] if not feat_row.empty else "N/A"
    
    # print values cleanly
    days_str = f"{days:.0f}" if isinstance(days, (int, float)) and not pd.isna(days) else str(days)
    freq_str = f"{freq:.4f}" if isinstance(freq, float) and not pd.isna(freq) else str(freq)
    mom_str = f"{mom:.4f}" if isinstance(mom, float) and not pd.isna(mom) else str(mom)
    eng_str = f"{eng:.4f}" if isinstance(eng, float) and not pd.isna(eng) else str(eng)
    print(f"{r['channel_id']}: days_since={days_str}  upload_30d={freq_str}  momentum={mom_str}  engagement={eng_str}")

# Load the model joblib to do exact Silhouette and Bootstrap stability analysis
model_dir = ROOT_DIR / "models"
joblib_files = list(model_dir.glob("*.joblib"))

model_loaded = False
if joblib_files:
    try:
        model_data = joblib.load(joblib_files[0])
        model = model_data["model"]
        preprocessor = model_data["preprocessor"]
        model_feat_cols = model_data["feature_cols"]
        best_k = model_data["best_k"]
        eval_results = model_data["eval_results"]
        model_type = eval_results.get("model_type", "kmeans")
        model_loaded = True
        print(f"\nLoaded persisted model: {model_type.upper()} (K={best_k}) from {joblib_files[0].name}")
    except Exception as e:
        print(f"\nError loading persisted model: {e}")

if not model_loaded:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import RobustScaler
    model_type = "kmeans"
    best_k = 2
    model_feat_cols = [c for c in feat_cols if c != "freq_trend_ratio"]
    preprocessor = None
    print(f"\nFalling back to default KMeans (K=2) and robust scaling.")

# Prepare scaled features
X_raw = scored[model_feat_cols].fillna(0)
if preprocessor:
    X_scaled = preprocessor.transform(X_raw)
else:
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_raw.values)

print(f"\n--- K-Means Silhouette Analysis ---")
from sklearn.metrics import silhouette_samples, silhouette_score
labels = scored["cluster_id"].values

# If labels are invalid or all same (e.g. all unscored), handle it
unique_labels = sorted(scored["cluster_id"].unique())
if len(unique_labels) > 1:
    sil_vals = silhouette_samples(X_scaled, labels)
    for cid in unique_labels:
        vals = sil_vals[labels == cid]
        print(f"  Cluster {int(cid)}: mean={vals.mean():.4f}  min={vals.min():.4f}  max={vals.max():.4f}  n={len(vals)}")
    print(f"  Overall: {silhouette_score(X_scaled, labels):.4f}")
else:
    print("  Cannot perform Silhouette analysis (fewer than 2 active clusters)")

# Feature contribution (Healthy vs At-Risk effect size)
cluster_healthy = scored[scored["risk_flag"] == "Healthy"]
cluster_at_risk = scored[scored["risk_flag"] == "At-Risk"]
if not cluster_healthy.empty and not cluster_at_risk.empty:
    print(f"\n--- Feature Contribution to Separation (Healthy vs At-Risk effect size) ---")
    for col in feat_cols:
        if col in scored.columns:
            m0, m1 = cluster_healthy[col].mean(), cluster_at_risk[col].mean()
            s0, s1 = cluster_healthy[col].std(), cluster_at_risk[col].std()
            pooled = np.sqrt(((s0**2 if pd.notna(s0) else 0) + (s1**2 if pd.notna(s1) else 0)) / 2)
            d = abs(m0 - m1) / pooled if pooled > 0 else 0
            print(f"  {col:30s}: Healthy={m0:.4f}  At-Risk={m1:.4f}  effect={d:.4f}")
else:
    print(f"\n--- Feature Contribution to Separation (effect size) ---")
    print("  Cannot compare Healthy vs At-Risk (one or both clusters empty)")

# Bootstrap stability
print(f"\n--- Bootstrap Cluster Stability ---")
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

def match_labels(orig, boot, k):
    # Greedy label matching
    conf = np.zeros((k, k))
    for o, b in zip(orig, boot):
        if 0 <= o < k and 0 <= b < k:
            conf[int(o), int(b)] += 1
    
    mapping = {}
    used_boot = set()
    pairs = []
    for o in range(k):
        for b in range(k):
            pairs.append((conf[o, b], o, b))
    pairs.sort(reverse=True, key=lambda x: x[0])
    
    for count, o, b in pairs:
        if o not in mapping and b not in used_boot:
            mapping[b] = o
            used_boot.add(b)
    
    unmapped_boot = set(range(k)) - used_boot
    unmapped_orig = set(range(k)) - set(mapping.values())
    for b, o in zip(unmapped_boot, unmapped_orig):
        mapping[b] = o
        
    return np.array([mapping.get(x, -1) for x in boot])

if len(unique_labels) > 1:
    n_boot = 100
    agreements = []
    for b in range(n_boot):
        idx = resample(range(len(X_scaled)), n_samples=len(X_scaled))
        X_boot = X_scaled[idx]
        if model_type == "kmeans":
            boot_model = KMeans(n_clusters=best_k, random_state=42 + b, n_init=10)
        elif model_type == "gmm":
            boot_model = GaussianMixture(n_components=best_k, random_state=42 + b)
        else:
            boot_model = KMeans(n_clusters=best_k, random_state=42 + b, n_init=10)
        
        # Fit and predict labels
        if hasattr(boot_model, "fit_predict"):
            boot_labels = boot_model.fit_predict(X_boot)
        else:
            boot_labels = boot_model.fit(X_boot).predict(X_boot)
            
        orig_labels_boot = labels[idx]
        mapped_boot = match_labels(orig_labels_boot, boot_labels, best_k)
        agreements.append((mapped_boot == orig_labels_boot).mean())
        
    print(f"  Bootstrap replicate agreement (mean): {np.mean(agreements):.4f}")
    print(f"  Bootstrap replicate agreement (std):  {np.std(agreements):.4f}")
    print(f"  Min agreement across 100 boots:       {np.min(agreements):.4f}")
else:
    print("  Cannot perform bootstrap stability analysis (fewer than 2 active clusters)")
