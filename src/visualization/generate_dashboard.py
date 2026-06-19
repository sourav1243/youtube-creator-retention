"""Generate interactive HTML dashboard from pipeline data."""
import json
import base64
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent.parent

clust = pd.read_parquet(ROOT / "data" / "processed" / "creator_clusters.parquet")
channels = pd.read_parquet(ROOT / "data" / "processed" / "channels_clean.parquet")
videos = pd.read_parquet(ROOT / "data" / "processed" / "videos_clean.parquet")

merged = channels.merge(clust, on="channel_id", how="left")
scored = merged[~merged["insufficient_history"].fillna(True)].copy()

FEATURES = ["upload_freq_30d", "upload_freq_90d", "freq_trend_ratio", "momentum_ratio",
            "avg_engagement_rate", "days_since_last_upload", "upload_regularity", "duration_trend"]
FEATURE_LABELS = ["Upload Freq (30d)", "Upload Freq (90d)", "Freq Trend Ratio", "Momentum Ratio",
                  "Engagement Rate", "Days Since Last Upload", "Upload Regularity", "Duration Trend"]

color_map = {"Healthy": "#22c55e", "Watch": "#f59e0b", "At-Risk": "#ef4444", "Unscored": "#6b7280"}
label_order = ["Healthy", "Watch", "At-Risk", "Unscored"]

def cluster_centroids(df):
    numeric = df.select_dtypes(include=[np.number])
    return df.groupby("cluster_label")[numeric.columns].mean().reset_index()

def make_kpi(value, label, color, prefix="", suffix=""):
    return f'<div class="kpi"><span class="kpi-value" style="color:{color}">{prefix}{value}{suffix}</span><span class="kpi-label">{label}</span></div>'

def gen_dashboard():
    total = len(merged)
    at_risk = len(merged[merged["risk_flag"] == "At-Risk"])
    healthy = len(merged[merged["risk_flag"] == "Healthy"])
    unscored = len(merged[merged["risk_flag"] == "Unscored"])
    risk_pct = round(at_risk / total * 100, 1) if total else 0
    avg_momentum = round(scored["momentum_ratio"].mean(), 2) if len(scored) else "N/A"

    risk_counts = merged["risk_flag"].value_counts()
    fig_cluster = px.pie(
        names=risk_counts.index,
        values=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=color_map,
        title="Creator Risk Distribution",
        hole=0.5,
        category_orders={"risk_flag": label_order}
    )
    fig_cluster.update_traces(textposition="inside", textinfo="percent+label",
                              hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>")
    fig_cluster.update_layout(showlegend=False, margin=dict(t=40, b=10, l=10, r=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e2e8f0"))

    fig_bar = px.bar(
        x=risk_counts.index,
        y=risk_counts.values,
        color=risk_counts.index,
        color_discrete_map=color_map,
        title="Channels by Risk Tier",
        text_auto=True,
        category_orders={"risk_flag": label_order}
    )
    fig_bar.update_traces(marker=dict(line=dict(width=0)), textposition="outside")
    fig_bar.update_layout(xaxis_title=None, yaxis_title="Count", showlegend=False,
                          margin=dict(t=40, b=10, l=10, r=10),
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e2e8f0"), xaxis=dict(tickfont=dict(color="#e2e8f0")))

    fig_scatter = px.scatter(
        scored, x="upload_freq_30d", y="momentum_ratio",
        color="risk_flag", color_discrete_map=color_map,
        hover_data=["channel_id", "title", "avg_engagement_rate", "days_since_last_upload"],
        title="Momentum vs Upload Frequency (by Risk Tier)",
        labels={"upload_freq_30d": "Upload Frequency (30d)", "momentum_ratio": "Momentum Ratio"},
        category_orders={"risk_flag": label_order}
    )
    fig_scatter.update_traces(marker=dict(size=10, line=dict(width=1, color="#1e293b")), opacity=0.85)
    fig_scatter.update_layout(margin=dict(t=40, b=10, l=10, r=10),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              font=dict(color="#e2e8f0"),
                              xaxis=dict(gridcolor="#334155", tickfont=dict(color="#e2e8f0")),
                              yaxis=dict(gridcolor="#334155", tickfont=dict(color="#e2e8f0")),
                              legend=dict(font=dict(color="#e2e8f0")))

    centroids = cluster_centroids(scored)
    fig_radar = go.Figure()
    for _, row in centroids.iterrows():
        label = row["cluster_label"]
        vals = [float(row[c]) if pd.notna(row.get(c, np.nan)) else 0 for c in FEATURES]
        norm_vals = []
        for i, c in enumerate(FEATURES):
            col_vals = scored[c].dropna()
            if len(col_vals) > 1:
                norm = (vals[i] - col_vals.min()) / (col_vals.max() - col_vals.min() + 1e-10)
            else:
                norm = 0
            norm_vals.append(round(norm, 3))
        risk = scored[scored["cluster_label"] == label]["risk_flag"].iloc[0] if label != "Unscored" else "Unscored"
        fig_radar.add_trace(go.Scatterpolar(
            r=norm_vals + [norm_vals[0]],
            theta=FEATURE_LABELS + [FEATURE_LABELS[0]],
            name=label,
            line=dict(color=color_map.get(risk, "#6b7280"), width=2),
            fill="toself",
            opacity=0.3
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor="#334155",
                                    tickfont=dict(color="#e2e8f0")),
                   angularaxis=dict(gridcolor="#334155", tickfont=dict(size=9, color="#e2e8f0"))),
        title="Feature Profile by Cluster (Normalized)",
        margin=dict(t=40, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"), legend=dict(font=dict(color="#e2e8f0"))
    )

    fig_hist = make_subplots(rows=2, cols=4, subplot_titles=FEATURE_LABELS,
                             horizontal_spacing=0.06, vertical_spacing=0.1)
    for i, (col, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
        r, c = i // 4 + 1, i % 4 + 1
        vals = scored[col].dropna()
        if len(vals) > 0:
            fig_hist.add_trace(go.Histogram(x=vals, marker_color="#3b82f6", opacity=0.8,
                                            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>"),
                               row=r, col=c)
        fig_hist.update_xaxes(title_text="", row=r, col=c, gridcolor="#334155", tickfont=dict(size=8, color="#e2e8f0"))
        fig_hist.update_yaxes(title_text="", row=r, col=c, gridcolor="#334155", tickfont=dict(size=8, color="#e2e8f0"))
    fig_hist.update_layout(title="Feature Distributions", showlegend=False,
                           margin=dict(t=50, b=10, l=10, r=10), height=500,
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#e2e8f0"))

    fig_corr = px.imshow(
        scored[FEATURES].corr(),
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title="Feature Correlation Matrix",
        labels=dict(x="Feature", y="Feature", color="Correlation")
    )
    fig_corr.update_layout(margin=dict(t=40, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#e2e8f0"),
                           xaxis=dict(tickfont=dict(size=8, color="#e2e8f0")),
                           yaxis=dict(tickfont=dict(size=8, color="#e2e8f0")))

    at_risk_df = merged[merged["risk_flag"] == "At-Risk"].sort_values("risk_score", ascending=False)
    table_rows = ""
    for _, r in at_risk_df.head(20).iterrows():
        title = str(r.get("title", "Unknown"))[:40]
        subs = f'{int(r["subscriber_count"]):,}' if pd.notna(r.get("subscriber_count")) else "N/A"
        days = f'{int(r["days_since_last_upload"])}' if pd.notna(r.get("days_since_last_upload")) else "N/A"
        mom = f'{r["momentum_ratio"]:.2f}' if pd.notna(r.get("momentum_ratio")) else "N/A"
        eng = f'{r["avg_engagement_rate"]:.4f}' if pd.notna(r.get("avg_engagement_rate")) else "N/A"
        freq = f'{r["upload_freq_30d"]:.2f}' if pd.notna(r.get("upload_freq_30d")) else "N/A"
        risk_s = f'{r["risk_score"]:.2f}' if pd.notna(r.get("risk_score")) else "N/A"
        conf = f'{r["confidence"]:.0%}' if pd.notna(r.get("confidence")) else "N/A"

        if isinstance(days, str) or int(days) >= 21:
            action = "Urgent outreach needed"
        elif isinstance(days, str) or int(days) >= 14:
            action = "Check in with creator"
        else:
            action = "Monitor closely"

        table_rows += f"""<tr>
            <td>{title}</td>
            <td>{subs}</td>
            <td>{freq}</td>
            <td>{mom}</td>
            <td>{eng}</td>
            <td>{days}</td>
            <td>{risk_s}</td>
            <td>{conf}</td>
            <td><span class="action-badge">{action}</span></td>
        </tr>"""

    charts_json = json.dumps({
        "cluster_pie": fig_cluster.to_json(),
        "risk_bar": fig_bar.to_json(),
        "scatter": fig_scatter.to_json(),
        "radar": fig_radar.to_json(),
        "hist": fig_hist.to_json(),
        "corr": fig_corr.to_json(),
    })

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YouTube Creator Retention Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; background: #0f172a; color: #e2e8f0; }}
.header {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); padding: 32px 40px; border-bottom: 1px solid #334155; }}
.header h1 {{ font-size: 28px; font-weight: 700; background: linear-gradient(135deg, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.header p {{ color: #94a3b8; margin-top: 6px; font-size: 14px; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
.kpi-row {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.kpi-card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; text-align: center; }}
.kpi-value {{ font-size: 32px; font-weight: 700; display: block; }}
.kpi-label {{ font-size: 13px; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }}
.chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
.chart-full {{ grid-column: 1 / -1; }}
.chart-card {{ background: #1e293b; border-radius: 12px; padding: 16px; border: 1px solid #334155; }}
.table-wrap {{ overflow-x: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th {{ text-align: left; padding: 12px 8px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; font-size: 11px; border-bottom: 1px solid #334155; }}
td {{ padding: 10px 8px; border-bottom: 1px solid #1e293b; }}
tr:hover {{ background: #0f172a; }}
.action-badge {{ padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 500; }}
tr td:nth-child(9) .action-badge {{ background: rgba(239,68,68,0.15); color: #ef4444; }}
.footer {{ text-align: center; padding: 32px; color: #475569; font-size: 13px; }}
@media (max-width: 768px) {{ .chart-grid {{ grid-template-columns: 1fr; }} .header {{ padding: 20px; }} .container {{ padding: 12px; }} }}
</style>
</head>
<body>
<div class="header">
<h1>YouTube Creator Retention Dashboard</h1>
<p>Real-time clustering of {total} creators — pipeline segments channels by engagement momentum, upload frequency, and risk of churn</p>
</div>
<div class="container">
<div class="kpi-row">
<div class="kpi-card"><span class="kpi-value" style="color:#22c55e">{healthy}</span><span class="kpi-label">Healthy Creators</span></div>
<div class="kpi-card"><span class="kpi-value" style="color:#ef4444">{at_risk}</span><span class="kpi-label">At-Risk Creators</span></div>
<div class="kpi-card"><span class="kpi-value">{risk_pct}%</span><span class="kpi-label">At-Risk Rate</span></div>
<div class="kpi-card"><span class="kpi-value">{unscored}</span><span class="kpi-label">Unscored (Insufficient Data)</span></div>
<div class="kpi-card"><span class="kpi-value">{avg_momentum}</span><span class="kpi-label">Avg Momentum Ratio</span></div>
</div>

<div class="chart-grid">
<div class="chart-card" id="chart-pie"></div>
<div class="chart-card" id="chart-bar"></div>
</div>

<div class="chart-grid">
<div class="chart-card chart-full" id="chart-scatter"></div>
</div>

<div class="chart-grid">
<div class="chart-card chart-full" id="chart-radar"></div>
</div>

<div class="chart-grid">
<div class="chart-card chart-full" id="chart-hist"></div>
</div>

<div class="chart-grid">
<div class="chart-card chart-full" id="chart-corr"></div>
</div>

<div class="chart-card" style="margin-bottom:24px;">
<h3 style="font-size:15px;margin-bottom:12px;color:#e2e8f0;">At-Risk Creators — Top 20</h3>
<div class="table-wrap">
<table>
<thead><tr>
<th>Channel</th><th>Subscribers</th><th>Freq (30d)</th><th>Momentum</th><th>Engagement</th><th>Days Idle</th><th>Risk Score</th><th>Confidence</th><th>Action</th>
</tr></thead>
<tbody>{table_rows}</tbody>
</table>
</div>
</div>
</div>

<div class="footer">
<p>Generated by YouTube Creator Retention Pipeline &mdash; Data refreshed {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
</div>

<script>
const charts = {charts_json};
const render = (id, json) => Plotly.newPlot(id, json.data, json.layout, {{responsive: true, displayModeBar: false}});
render('chart-pie', JSON.parse(charts.cluster_pie));
render('chart-bar', JSON.parse(charts.risk_bar));
render('chart-scatter', JSON.parse(charts.scatter));
render('chart-radar', JSON.parse(charts.radar));
render('chart-hist', JSON.parse(charts.hist));
render('chart-corr', JSON.parse(charts.corr));
window.addEventListener('resize', () => {{ document.querySelectorAll('.chart-card').forEach(el => Plotly.Plots.resize(el)); }});
</script>
</body>
</html>"""

    out_path = ROOT / "dashboard" / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard generated: {out_path}")

if __name__ == "__main__":
    gen_dashboard()
