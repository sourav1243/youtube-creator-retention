"""Streamlit dashboard — CDN-free fallback for YouTube Creator Retention."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent

COLOR_MAP = {"Healthy": "#22c55e", "Watch": "#f59e0b", "At-Risk": "#ef4444", "Unscored": "#6b7280"}
LABEL_ORDER = ["Healthy", "Watch", "At-Risk", "Unscored"]

FEATURE_LABELS = {
    "upload_freq_30d": "Uploads / Day (30d)",
    "upload_freq_90d": "Uploads / Day (90d)",
    "freq_trend_ratio": "Freq Trend",
    "momentum_ratio": "Momentum",
    "avg_engagement_rate": "Engagement",
    "days_since_last_upload": "Days Idle",
    "upload_regularity": "Regularity",
}

FEATURE_COLS = [
    "upload_freq_30d", "upload_freq_90d", "freq_trend_ratio",
    "momentum_ratio", "avg_engagement_rate", "days_since_last_upload",
    "upload_regularity",
]

RISK_WEIGHTS = {
    "momentum_ratio": 0.30,
    "upload_freq_30d": 0.20,
    "avg_engagement_rate": 0.15,
    "days_since_last_upload": 0.15,
    "upload_regularity": 0.10,
    "freq_trend_ratio": 0.10,
}


@st.cache_data
def load_data():
    clust = pd.read_parquet(ROOT / "data" / "processed" / "creator_clusters.parquet")
    channels = pd.read_parquet(ROOT / "data" / "processed" / "channels_clean.parquet")
    merged = channels.merge(clust, on="channel_id", how="left")
    scored = merged[~merged["insufficient_history"].fillna(True)].copy()
    return merged, scored


def norm_for_radar(val, series):
    if len(series) < 2:
        return 0.5
    return (val - series.min()) / (series.max() - series.min() + 1e-10)


def compute_action(row):
    days = row.get("days_since_last_upload") or 0
    momentum = row.get("momentum_ratio") or 1.0
    risk = row.get("risk_score") or 0.5
    if days >= 30:
        return "Urgent outreach"
    if days >= 14:
        return "Check in with creator"
    if momentum < 0.5 and risk > 0.5:
        return "Priority review"
    if days >= 7:
        return "Monitor closely"
    return "Auto-email check-in"


def main():
    st.set_page_config(page_title="YouTube Partner Health", layout="wide", page_icon="📊")

    merged, scored = load_data()
    total = len(merged)
    healthy = int((merged["risk_flag"] == "Healthy").sum())
    watch = int((merged["risk_flag"] == "Watch").sum())
    at_risk = int((merged["risk_flag"] == "At-Risk").sum())
    unscored = int((merged["risk_flag"] == "Unscored").sum())
    scored_count = healthy + watch + at_risk

    avg_momentum = round(scored["momentum_ratio"].mean(), 2) if len(scored) else None
    avg_eng = f"{scored['avg_engagement_rate'].mean()*100:.2f}%" if len(scored) else "N/A"

    st.markdown("""
        <style>
        .block-container {max-width: 1440px; padding: 1rem 2rem;}
        .stTabs [data-baseweb="tab-list"] {gap: 2px;}
        .stTabs [data-baseweb="tab"] {padding: 8px 16px; font-size: 13px; font-weight: 500;}
        h1 {font-size: 20px !important; font-weight: 600 !important;}
        h2 {font-size: 14px !important; font-weight: 600 !important; color: #64748b !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; margin-top: 24px !important;}
        .kpi-label {font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.6px; font-weight: 500;}
        .kpi-value {font-size: 28px; font-weight: 700;}
        .kpi-sub {font-size: 12px; color: #475569;}
        .metric-card {background: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155;}
        </style>
    """, unsafe_allow_html=True)

    st.title("YouTube Partner Health — Creator Retention")

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="kpi-label">Total Creators</div><div class="kpi-value">{total}</div><div class="kpi-sub">{scored_count} scored + {unscored} unscored</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="kpi-label">Healthy</div><div class="kpi-value" style="color:#22c55e">{healthy}</div><div class="kpi-sub">{round(healthy/total*100)}% of total</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="kpi-label">Watch</div><div class="kpi-value" style="color:#f59e0b">{watch}</div><div class="kpi-sub">{round(watch/total*100)}% of total</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="kpi-label">At-Risk</div><div class="kpi-value" style="color:#ef4444">{at_risk}</div><div class="kpi-sub">{round(at_risk/total*100,1)}% of total</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="kpi-label">Avg Momentum</div><div class="kpi-value" style="color:#60a5fa">{avg_momentum or "N/A"}</div><div class="kpi-sub">recent vs past views</div></div>', unsafe_allow_html=True)
    with col6:
        st.markdown(f'<div class="metric-card"><div class="kpi-label">Avg Engagement</div><div class="kpi-value" style="color:#a78bfa">{avg_eng}</div><div class="kpi-sub">likes+comments/view</div></div>', unsafe_allow_html=True)

    st.markdown("## Risk Overview")
    tab1, tab2 = st.tabs(["Portfolio Distribution", "Channels by Tier"])
    risk_counts = merged["risk_flag"].value_counts()

    with tab1:
        fig_pie = px.pie(
            names=risk_counts.index, values=risk_counts.values,
            color=risk_counts.index, color_discrete_map=COLOR_MAP,
            hole=0.55, category_orders={"risk_flag": LABEL_ORDER},
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label", rotation=45,
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
            marker=dict(line=dict(width=2, color="#0b1120")),
        )
        fig_pie.update_layout(
            showlegend=False, height=400,
            margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0", size=13),
            annotations=[dict(text=f"{total}", showarrow=False, font=dict(size=26, color="#e2e8f0"))],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with tab2:
        fig_bar = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index, color_discrete_map=COLOR_MAP,
            text_auto=True, category_orders={"risk_flag": LABEL_ORDER},
        )
        fig_bar.update_traces(marker=dict(line=dict(width=0)), textposition="outside", textfont=dict(size=13))
        fig_bar.update_layout(
            xaxis_title=None, yaxis_title=None, showlegend=False, height=400,
            margin=dict(t=10, b=20, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
            yaxis=dict(tickfont=dict(color="#e2e8f0"), showgrid=True, gridcolor="#1e293b"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("## Momentum vs Upload Frequency")
    fig_scatter = px.scatter(
        scored, x="upload_freq_30d", y="momentum_ratio",
        color="risk_flag", color_discrete_map=COLOR_MAP,
        hover_data=["title", "avg_engagement_rate", "days_since_last_upload", "risk_score"],
        labels={"upload_freq_30d": "Uploads per Day (30d)", "momentum_ratio": "Momentum Ratio"},
        category_orders={"risk_flag": LABEL_ORDER},
    )
    fig_scatter.update_traces(marker=dict(size=7, line=dict(width=1, color="#0f172a")), opacity=0.85)
    fig_scatter.update_layout(
        height=500, margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0")),
        yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0")),
        legend=dict(font=dict(color="#e2e8f0"), orientation="h", y=1.06, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("## Risk Score Distribution")
    fig_hist = go.Figure()
    for flag in ["Healthy", "Watch", "At-Risk"]:
        subset = scored[scored["risk_flag"] == flag]["risk_score"].dropna()
        if len(subset) == 0:
            continue
        fig_hist.add_trace(go.Histogram(
            x=subset, name=flag, marker_color=COLOR_MAP[flag],
            opacity=0.65, nbinsx=20,
            hovertemplate="Risk Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))
    fig_hist.update_layout(
        barmode="overlay", height=400,
        xaxis_title="Risk Score (0 = lowest, 1 = highest)",
        yaxis_title="Channels",
        margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0"), range=[0, 1]),
        yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0")),
        legend=dict(font=dict(color="#e2e8f0"), orientation="h", y=1.06, x=0.5, xanchor="center"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("## Feature Analysis")
    tab3, tab4 = st.tabs(["Feature Profiles by Cluster", "Feature Correlation"])

    with tab3:
        fig_centroids = go.Figure()
        for label, group in scored.groupby("cluster_label"):
            numeric = group.select_dtypes(include=[np.number])
            centroid = numeric.mean()
            vals = [norm_for_radar(centroid.get(c, 0), scored[c].dropna()) for c in FEATURE_COLS]
            flag = group["risk_flag"].mode().iloc[0] if not group["risk_flag"].mode().empty else "Unscored"
            fig_centroids.add_trace(go.Scatterpolar(
                r=vals + [vals[0]],
                theta=[FEATURE_LABELS.get(c, c) for c in FEATURE_COLS] + [FEATURE_LABELS.get(FEATURE_COLS[0], FEATURE_COLS[0])],
                name=label,
                line=dict(color=COLOR_MAP.get(flag, "#6b7280"), width=2),
                fill="toself", opacity=0.3,
            ))
        fig_centroids.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e293b", tickfont=dict(color="#e2e8f0")),
                angularaxis=dict(gridcolor="#1e293b", tickfont=dict(size=10, color="#e2e8f0")),
            ),
            height=500, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            legend=dict(font=dict(color="#e2e8f0"), orientation="h", y=1.06, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig_centroids, use_container_width=True)

    with tab4:
        fig_corr = px.imshow(
            scored[FEATURE_COLS].corr(),
            text_auto=".2f", color_continuous_scale="RdBu_r",
            aspect="auto", zmin=-1, zmax=1,
            labels=dict(x="Feature", y="Feature", color="Corr"),
        )
        fig_corr.update_layout(
            height=500, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis=dict(tickfont=dict(size=9, color="#e2e8f0")),
            yaxis=dict(tickfont=dict(size=9, color="#e2e8f0")),
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("## Methodology")
    with st.expander("Risk Score Calculation", expanded=True):
        st.markdown("""
        Each creator receives a risk score **0.0–1.0** (higher = more likely to churn) based on six weighted signals.
        """)
        wcols = st.columns(len(RISK_WEIGHTS))
        for i, (k, v) in enumerate(sorted(RISK_WEIGHTS.items())):
            wcols[i].metric(k.replace("_", " ").title(), f"{v:.0%}")

        st.markdown("""
        Each signal is normalized to [0, 1] so that lower upload frequency, declining momentum, longer idle periods,
        lower engagement, erratic upload schedules, and declining upload frequency all contribute to higher risk.
        The final score is a weighted sum. Channels are divided into three tiers by risk score percentile:
        **Healthy** (bottom 30%), **Watch** (middle 40%), and **At-Risk** (top 30%).
        """)

    with st.expander("Clustering", expanded=False):
        st.markdown("""
        K-means, Gaussian Mixture, or DBSCAN (selected by silhouette score) segments creators by behavioral
        patterns using the same six features. Cluster labels describe each group's profile.
        The cluster label is **independent** of the risk tier — a creator in a "High Momentum" cluster may still be
        At-Risk if their recent trend is declining.
        """)

    with st.expander("Confidence", expanded=False):
        st.markdown("""
        Bootstrapped stability (500 iterations with replacement) measures how consistently each creator is
        assigned to their cluster. Higher confidence (closer to 100%) means the assignment is stable.
        Lower confidence suggests the creator sits near a cluster boundary.
        """)

    st.markdown("## Creator Directory")
    tier_filter = st.selectbox("Filter by tier", ["All", "Healthy", "Watch", "At-Risk", "Unscored"])
    search = st.text_input("Search by channel name")

    table_df = merged.sort_values("risk_score", ascending=False).copy()
    table_df["action"] = table_df.apply(compute_action, axis=1)
    table_df["subs_display"] = table_df["subscriber_count"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")
    table_df["freq_display"] = table_df["upload_freq_30d"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    table_df["mom_display"] = table_df["momentum_ratio"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    table_df["eng_display"] = table_df["avg_engagement_rate"].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    table_df["days_display"] = table_df["days_since_last_upload"].apply(lambda x: str(int(x)) if pd.notna(x) else "N/A")
    table_df["risk_display"] = table_df["risk_score"].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    table_df["conf_display"] = table_df["confidence"].apply(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")

    if tier_filter != "All":
        table_df = table_df[table_df["risk_flag"] == tier_filter]
    if search:
        table_df = table_df[table_df["title"].str.lower().str.contains(search.lower(), na=False)]

    display_cols = ["title", "subs_display", "freq_display", "mom_display", "eng_display",
                    "days_display", "risk_display", "conf_display", "cluster_label", "action"]
    col_names = ["Channel", "Subscribers", "Freq 30d", "Momentum", "Engagement",
                 "Days Idle", "Risk Score", "Confidence", "Cluster", "Action"]

    st.dataframe(
        table_df[display_cols].rename(columns=dict(zip(display_cols, col_names))),
        use_container_width=True, height=min(60 * len(table_df), 600),
        column_config={c: st.column_config.TextColumn(c) for c in col_names},
    )

    csv = table_df.to_csv(index=False)
    st.download_button("Export CSV", data=csv, file_name="creator_dashboard_export.csv", mime="text/csv")

    st.caption(f"YouTube Creator Retention Pipeline v2 | {pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M UTC')} | {total} channels, {len(scored)} scored")


if __name__ == "__main__":
    main()
