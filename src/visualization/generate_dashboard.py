"""Generate interactive HTML dashboard from pipeline data."""

import json

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.config import ROOT_DIR as ROOT
from src.modeling.cluster import FEATURE_COLUMNS, RISK_WEIGHTS

clust_path = ROOT / "data" / "processed" / "creator_clusters.parquet"
chan_path = ROOT / "data" / "processed" / "channels_clean.parquet"

FEATURE_LABELS = {
    "upload_freq_30d": "Uploads / Day (30d)",
    "upload_freq_90d": "Uploads / Day (90d)",
    "freq_trend_ratio": "Freq Trend",
    "momentum_ratio": "Momentum",
    "avg_engagement_rate": "Engagement",
    "days_since_last_upload": "Days Idle",
    "upload_regularity": "Regularity",
}

COLOR_MAP = {"Healthy": "#22c55e", "Watch": "#f59e0b", "At-Risk": "#ef4444", "Unscored": "#6b7280"}
LABEL_ORDER = ["Healthy", "Watch", "At-Risk", "Unscored"]


def _css() -> str:
    return """*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Inter',system-ui,-apple-system,sans-serif;background:#0b1120;color:#e2e8f0;-webkit-font-smoothing:antialiased}
.topbar{display:flex;align-items:center;justify-content:space-between;padding:0 32px;height:56px;border-bottom:1px solid #1e293b;background:#0f172a;flex-shrink:0}
.topbar h1{font-size:15px;font-weight:600;letter-spacing:.2px;display:flex;align-items:center;gap:8px}
.topbar h1 span{color:#60a5fa}
.topbar h1 small{font-weight:400;color:#475569;font-size:12px}
.topbar .version{font-size:11px;color:#475569;background:#1e293b;padding:3px 10px;border-radius:12px;white-space:nowrap}
.container{max-width:1440px;margin:0 auto;padding:24px;width:100%}
.section-title{font-size:13px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.8px;margin:24px 0 12px}
.section-title:first-of-type{margin-top:0}
.kpi-row{display:grid;grid-template-columns:repeat(6,1fr);gap:12px;margin-bottom:20px}
.kpi-card{background:#131c31;border-radius:10px;padding:16px;border:1px solid #1e293b}
.kpi-card .kpi-label{font-size:10px;color:#64748b;text-transform:uppercase;letter-spacing:.8px;font-weight:500}
.kpi-card .kpi-value{font-size:28px;font-weight:700;margin-top:4px;line-height:1.2}
.kpi-card .kpi-sub{font-size:11px;color:#475569;margin-top:2px}
.chart-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
.chart-full{grid-column:1/-1}
.chart-card{background:#131c31;border-radius:10px;padding:12px;border:1px solid #1e293b;overflow:visible}
.chart-card h3{font-size:12px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px;padding:0 4px}
.table-wrap{overflow-x:auto;border-radius:8px;border:1px solid #1e293b;background:#131c31}
.table-toolbar{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;border-bottom:1px solid #1e293b;flex-wrap:wrap;gap:8px}
.table-toolbar h3{font-size:12px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.6px}
.table-toolbar .controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.table-toolbar select,.table-toolbar input{background:#0b1120;border:1px solid #1e293b;border-radius:6px;padding:6px 12px;color:#e2e8f0;font-size:12px;font-family:inherit;outline:none}
.table-toolbar select{cursor:pointer}
.table-toolbar input:focus,.table-toolbar select:focus{border-color:#475569}
table{width:100%;border-collapse:collapse;font-size:12px}
th{text-align:left;padding:10px 12px;color:#64748b;font-weight:500;text-transform:uppercase;letter-spacing:.5px;font-size:10px;border-bottom:1px solid #1e293b;cursor:pointer;user-select:none;white-space:nowrap;position:sticky;top:0;background:#131c31}
th:hover{color:#94a3b8}
th .sort-arrow{margin-left:3px;font-size:9px;opacity:.3}
th .sort-arrow.active{opacity:1;color:#60a5fa}
td{padding:10px 12px;border-bottom:1px solid #0f172a;white-space:nowrap}
tr:hover td{background:rgba(255,255,255,.02)}
.risk-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;vertical-align:middle;flex-shrink:0}
.action-badge{padding:2px 8px;border-radius:12px;font-size:10px;font-weight:500;white-space:nowrap}
.action-urgent{background:rgba(239,68,68,.12);color:#fca5a5}
.action-check{background:rgba(245,158,11,.12);color:#fcd34d}
.action-monitor{background:rgba(59,130,246,.12);color:#93c5fd}
.action-auto{background:rgba(34,197,94,.12);color:#86efac}
.footer{text-align:center;padding:24px;color:#334155;font-size:11px}
.methodology{background:#131c31;border-radius:10px;padding:20px;border:1px solid #1e293b;margin-bottom:16px}
.methodology h3{font-size:12px;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:.6px;margin-bottom:12px}
.methodology p,.methodology li{font-size:13px;color:#94a3b8;line-height:1.6}
.methodology ul{padding-left:20px;margin-bottom:8px}
.methodology .weight-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;margin:12px 0}
.methodology .weight-item{background:#0b1120;border-radius:6px;padding:8px 12px;font-size:12px;color:#94a3b8;display:flex;justify-content:space-between}
.methodology .weight-item .wval{color:#60a5fa;font-weight:600}
.no-data{padding:40px;text-align:center;color:#475569;font-size:13px}
.loading{display:flex;align-items:center;justify-content:center;padding:60px;color:#475569;font-size:14px;gap:12px}
.spinner{width:20px;height:20px;border:2px solid #1e293b;border-top-color:#60a5fa;border-radius:50%;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
.error-card{background:#7f1d1d;border:1px solid #991b1b;border-radius:10px;padding:24px;margin:24px;color:#fca5a5;text-align:center}
.error-card h3{font-size:16px;margin-bottom:8px}
.error-card p{font-size:13px;color:#fca5a5;opacity:.8}
@media(max-width:1024px){.kpi-row{grid-template-columns:repeat(3,1fr)}}
@media(max-width:768px){.kpi-row{grid-template-columns:repeat(2,1fr)}.chart-grid{grid-template-columns:1fr}.topbar{padding:0 16px}.container{padding:12px}.table-toolbar .controls{width:100%}.table-toolbar input{flex:1}}
@media(max-width:480px){.kpi-row{grid-template-columns:1fr 1fr}.topbar h1{font-size:13px}}
"""  # noqa: E501


def _compute_action(row: pd.Series) -> str:
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


def _norm_for_radar(val, series):
    if len(series) < 2:
        return 0.5
    return (val - series.min()) / (series.max() - series.min() + 1e-10)


def gen_dashboard():
    try:
        clust = pd.read_parquet(clust_path)
        channels = pd.read_parquet(chan_path)
    except (FileNotFoundError, OSError) as exc:
        html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Dashboard Error</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>body{{font-family:'Inter',system-ui,sans-serif;background:#0b1120;color:#e2e8f0;display:flex;align-items:center;justify-content:center;min-height:100vh}}</style>
</head><body>
<div class="error-card"><h3>Dashboard generation failed</h3>
<p>Pipeline data not found. Run the pipeline first: <code>python -m src.run_pipeline</code></p>
<p style="font-size:11px;opacity:.5;margin-top:8px">{str(exc)}</p></div>
</body></html>"""
        out_path = ROOT / "docs" / "index.html"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(html, encoding="utf-8")
        print(f"Error dashboard generated: {out_path}")
        return

    merged = channels.merge(clust, on="channel_id", how="left")
    scored = merged[~merged["insufficient_history"].fillna(True)].copy()

    total = len(merged)
    healthy = int((merged["risk_flag"] == "Healthy").sum())
    watch = int((merged["risk_flag"] == "Watch").sum())
    at_risk = int((merged["risk_flag"] == "At-Risk").sum())
    unscored = int((merged["risk_flag"] == "Unscored").sum())
    risk_pct = round(at_risk / total * 100, 1) if total else 0
    scored_count = healthy + watch + at_risk

    avg_momentum = round(scored["momentum_ratio"].mean(), 2) if len(scored) else None
    avg_eng = f"{scored['avg_engagement_rate'].mean()*100:.2f}%" if len(scored) else "N/A"

    risk_counts = merged["risk_flag"].value_counts()

    all_table_data = []
    for _, r in merged.sort_values("risk_score", ascending=False).iterrows():
        title = str(r.get("title", "Unknown"))[:60]
        subs = f"{int(r['subscriber_count']):,}" if pd.notna(r.get("subscriber_count")) else "N/A"
        action = _compute_action(r)

        all_table_data.append({
            "title": title,
            "subs": subs,
            "freq": f"{r['upload_freq_30d']:.2f}" if pd.notna(r.get("upload_freq_30d")) else "N/A",
            "mom": f"{r['momentum_ratio']:.2f}" if pd.notna(r.get("momentum_ratio")) else "N/A",
            "eng": f"{r['avg_engagement_rate']*100:.2f}%" if pd.notna(r.get("avg_engagement_rate")) else "N/A",
            "days": str(int(r["days_since_last_upload"])) if pd.notna(r.get("days_since_last_upload")) else "N/A",
            "risk_s": f"{r['risk_score']:.3f}" if pd.notna(r.get("risk_score")) else "N/A",
            "conf": f"{r['confidence']:.0%}" if pd.notna(r.get("confidence")) else "N/A",
            "cluster": str(r.get("cluster_label", "Unknown")),
            "flag": str(r.get("risk_flag", "Unknown")),
            "action": action,
            "risk_raw": float(r["risk_score"]) if pd.notna(r.get("risk_score")) else 0.5,
        })

    pie = px.pie(
        names=risk_counts.index, values=risk_counts.values,
        color=risk_counts.index, color_discrete_map=COLOR_MAP,
        title=None, hole=0.55, category_orders={"risk_flag": LABEL_ORDER},
    )
    pie.update_traces(
        textposition="inside", textinfo="percent+label", rotation=45,
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
        marker=dict(line=dict(width=2, color="#0b1120")),
    )
    pie.update_layout(
        showlegend=False, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", size=13),
        annotations=[dict(text=f"{total}", showarrow=False, font=dict(size=26, color="#e2e8f0"))],
    )

    bar = px.bar(
        x=risk_counts.index, y=risk_counts.values,
        color=risk_counts.index, color_discrete_map=COLOR_MAP,
        title=None, text_auto=True, category_orders={"risk_flag": LABEL_ORDER},
    )
    bar.update_traces(marker=dict(line=dict(width=0)), textposition="outside", textfont=dict(size=13))
    bar.update_layout(
        xaxis_title=None, yaxis_title=None, showlegend=False,
        margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(tickfont=dict(color="#e2e8f0", size=12)),
        yaxis=dict(tickfont=dict(color="#e2e8f0"), showgrid=True, gridcolor="#1e293b"),
    )

    scatter = px.scatter(
        scored, x="upload_freq_30d", y="momentum_ratio",
        color="risk_flag", color_discrete_map=COLOR_MAP,
        hover_data=["title", "avg_engagement_rate", "days_since_last_upload", "risk_score"],
        title=None,
        labels={"upload_freq_30d": "Uploads per Day (30d)", "momentum_ratio": "Momentum Ratio"},
        category_orders={"risk_flag": LABEL_ORDER},
    )
    scatter.update_traces(marker=dict(size=9, line=dict(width=1, color="#0f172a")), opacity=0.85)
    scatter.update_layout(
        margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0")),
        yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0")),
        legend=dict(font=dict(color="#e2e8f0"), orientation="h", y=1.06, x=0.5, xanchor="center"),
    )

    risk_dist = go.Figure()
    for flag in ["Healthy", "Watch", "At-Risk"]:
        subset = scored[scored["risk_flag"] == flag]["risk_score"].dropna()
        if len(subset) == 0:
            continue
        risk_dist.add_trace(go.Histogram(
            x=subset, name=flag, marker_color=COLOR_MAP[flag],
            opacity=0.65, nbinsx=20,
            hovertemplate="Risk Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
        ))
    risk_dist.update_layout(
        barmode="overlay", title=None,
        xaxis_title="Risk Score (0 = lowest, 1 = highest)",
        yaxis_title="Channels",
        margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0"), range=[0, 1]),
        yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#e2e8f0")),
        legend=dict(font=dict(color="#e2e8f0"), orientation="h", y=1.06, x=0.5, xanchor="center"),
    )

    centroids = go.Figure()
    for label, group in scored.groupby("cluster_label"):
        numeric = group.select_dtypes(include=[np.number])
        centroid = numeric.mean()
        vals = [_norm_for_radar(centroid.get(c, 0), scored[c].dropna()) for c in FEATURE_COLUMNS]
        flag = group["risk_flag"].mode().iloc[0] if not group["risk_flag"].mode().empty else "Unscored"
        centroids.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=[FEATURE_LABELS.get(c, c) for c in FEATURE_COLUMNS] + [FEATURE_LABELS.get(FEATURE_COLUMNS[0], FEATURE_COLUMNS[0])],
            name=label,
            line=dict(color=COLOR_MAP.get(flag, "#6b7280"), width=2),
            fill="toself", opacity=0.3,
        ))
    centroids.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor="#1e293b", tickfont=dict(color="#e2e8f0")),
            angularaxis=dict(gridcolor="#1e293b", tickfont=dict(size=10, color="#e2e8f0")),
        ),
        title=None, margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        legend=dict(font=dict(color="#e2e8f0"), orientation="h", y=1.06, x=0.5, xanchor="center"),
    )

    corr = px.imshow(
        scored[FEATURE_COLUMNS].corr(),
        text_auto=".2f", color_continuous_scale="RdBu_r",
        aspect="auto", title=None,
        labels=dict(x="Feature", y="Feature", color="Corr"),
        zmin=-1, zmax=1,
    )
    corr.update_layout(
        margin=dict(t=10, b=10, l=10, r=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        xaxis=dict(tickfont=dict(size=9, color="#e2e8f0")),
        yaxis=dict(tickfont=dict(size=9, color="#e2e8f0")),
    )

    weight_items = "".join(
        f'<div class="weight-item"><span>{k}</span><span class="wval">{v:.0%}</span></div>'
        for k, v in sorted(RISK_WEIGHTS.items())
    )

    charts = json.dumps({
        "pie": pie.to_json(), "bar": bar.to_json(), "scatter": scatter.to_json(),
        "risk_dist": risk_dist.to_json(), "centroids": centroids.to_json(), "corr": corr.to_json(),
        "table_data": all_table_data,
    })

    ts = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>YouTube Partner Health | Creator Retention</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js" onerror="document.getElementById('root').innerHTML='<div class=\\'error-card\\'><h3>Failed to load dashboard</h3><p>Plotly library could not be loaded. Check your internet connection or try a different network.</p></div>'"></script>
<style>{_css()}</style>
</head>
<body>
<div class="topbar">
<h1><span>YouTube</span> Partner Health <small>Creator Retention Pipeline</small></h1>
<span class="version">{ts}</span>
</div>
<div id="root">
<div class="loading"><div class="spinner"></div>Loading dashboard...</div>
</div>
<div id="loading" class="loading"><div class="spinner"></div>Rendering charts...</div>
<div id="content" style="display:none">

<div class="container">
<div class="kpi-row">
<div class="kpi-card"><div class="kpi-label">Total Creators</div><div class="kpi-value" style="color:#e2e8f0">{total}</div><div class="kpi-sub">{scored_count} scored + {unscored} unscored</div></div>
<div class="kpi-card"><div class="kpi-label">Healthy</div><div class="kpi-value" style="color:#22c55e">{healthy}</div><div class="kpi-sub">{round(healthy/total*100)}% of total</div></div>
<div class="kpi-card"><div class="kpi-label">Watch</div><div class="kpi-value" style="color:#f59e0b">{watch}</div><div class="kpi-sub">{round(watch/total*100)}% of total</div></div>
<div class="kpi-card"><div class="kpi-label">At-Risk</div><div class="kpi-value" style="color:#ef4444">{at_risk}</div><div class="kpi-sub">{risk_pct}% of total</div></div>
<div class="kpi-card"><div class="kpi-label">Avg Momentum</div><div class="kpi-value" style="color:#60a5fa">{avg_momentum or "N/A"}</div><div class="kpi-sub">recent vs past viewership</div></div>
<div class="kpi-card"><div class="kpi-label">Avg Engagement</div><div class="kpi-value" style="color:#a78bfa">{avg_eng}</div><div class="kpi-sub">likes+comments per view</div></div>
</div>

<div class="section-title">Risk Overview</div>
<div class="chart-grid">
<div class="chart-card"><h3>Portfolio Distribution</h3><div id="c-pie"></div></div>
<div class="chart-card"><h3>Channels by Tier</h3><div id="c-bar"></div></div>
</div>

<div class="chart-grid">
<div class="chart-card chart-full"><h3>Momentum vs Upload Frequency</h3><div id="c-scatter"></div></div>
</div>

<div class="chart-grid">
<div class="chart-card chart-full"><h3>Risk Score Distribution</h3><div id="c-risk-dist"></div></div>
</div>

<div class="section-title">Feature Analysis</div>
<div class="chart-grid">
<div class="chart-card"><h3>Feature Profiles by Cluster</h3><div id="c-centroids"></div></div>
<div class="chart-card"><h3>Feature Correlation</h3><div id="c-corr"></div></div>
</div>

<div class="section-title">Methodology</div>
<div class="methodology">
<h3>Risk Score Calculation</h3>
<p>Each creator receives a risk score <b>0.0–1.0</b> (higher = more likely to churn) based on six weighted signals:</p>
<div class="weight-grid">{weight_items}</div>
<p>Each signal is normalized to [0, 1] so that lower upload frequency, declining momentum, longer idle periods, lower engagement, erratic upload schedules, and declining upload frequency all contribute to higher risk. The final score is a weighted sum.</p>
<p>Channels are divided into three tiers by risk score percentile: <b style="color:#22c55e">Healthy</b> (bottom 30%), <b style="color:#f59e0b">Watch</b> (middle 40%), and <b style="color:#ef4444">At-Risk</b> (top 30%).</p>
<h3 style="margin-top:16px">Clustering</h3>
<p>K-means, Gaussian Mixture, or DBSCAN (selected by silhouette score) segments creators by behavioral patterns using the same six features. Cluster labels (e.g., "High Momentum — Frequent Uploaders") describe each group's profile. The cluster label is <b>independent</b> of the risk tier — a creator in the "High Momentum" cluster may still be At-Risk if their recent trend is declining.</p>
<h3 style="margin-top:16px">Confidence</h3>
<p>Bootstrapped stability (500 iterations with replacement) measures how consistently each creator is assigned to their cluster. Higher confidence (closer to 100%) means the assignment is stable. Lower confidence suggests the creator sits near a cluster boundary.</p>
</div>

<div class="section-title">Creator Directory</div>
<div class="chart-card chart-full" style="overflow:visible">
<div class="table-toolbar">
<h3>All Channels</h3>
<div class="controls">
<select id="tf">
<option value="all">All Tiers</option>
<option value="Healthy">Healthy</option>
<option value="Watch">Watch</option>
<option value="At-Risk">At-Risk</option>
<option value="Unscored">Unscored</option>
</select>
<input id="search" type="text" placeholder="Search by channel name...">
<button onclick="exportCSV()" style="background:#0b1120;border:1px solid #1e293b;border-radius:6px;padding:6px 12px;color:#94a3b8;font-size:12px;font-family:inherit;cursor:pointer">Export CSV</button>
</div>
</div>
<div class="table-wrap" id="tc"><div class="loading"><div class="spinner"></div></div></div>
</div>

</div>

<div class="footer">
YouTube Creator Retention Pipeline v2 &middot; {ts}
</div>
</div>

<script>
const D = {charts};

function err(msg) {{
  document.getElementById('root').innerHTML = '<div class="error-card"><h3>Something went wrong</h3><p>' + msg + '</p></div>';
}}
function esc(s) {{ var d=document.createElement('div');d.appendChild(document.createTextNode(s));return d.innerHTML; }}

function render() {{
  try {{
    var ids = ['pie','bar','scatter','risk-dist','centroids','corr'];
    var keys = ['pie','bar','scatter','risk_dist','centroids','corr'];
    for (var i=0;i<ids.length;i++) {{
      Plotly.newPlot('c-'+ids[i], JSON.parse(D[keys[i]]).data, JSON.parse(D[keys[i]]).layout, {{responsive:true,displayModeBar:false,hovermode:'closest'}});
    }}
    renderTable();
    document.getElementById('loading').style.display='none';
    document.getElementById('content').style.display='block';
  }} catch(e) {{ err(e.message); }}
}}

var sortKey='risk_raw', sortAsc=false;

function renderTable() {{
  var tier=document.getElementById('tf').value;
  var q=document.getElementById('search').value.toLowerCase();
  var rows=D.table_data.slice();
  if(tier!=='all') rows=rows.filter(function(r){{return r.flag===tier}});
  if(q) rows=rows.filter(function(r){{return r.title.toLowerCase().includes(q)}});
  rows.sort(function(a,b){{var av=a[sortKey],bv=b[sortKey];if(av==='N/A')av='';if(bv==='N/A')bv='';var cmp=typeof av==='number'&&typeof bv==='number'?av-bv:String(av).localeCompare(String(bv));return sortAsc?cmp:-cmp}});
  var cols=[{{k:'title',l:'Channel'}},{{k:'subs',l:'Subscribers'}},{{k:'freq',l:'Freq 30d'}},{{k:'mom',l:'Momentum'}},{{k:'eng',l:'Engagement'}},{{k:'days',l:'Days Idle'}},{{k:'risk_s',l:'Risk Score'}},{{k:'conf',l:'Confidence'}},{{k:'cluster',l:'Cluster'}},{{k:'action',l:'Action'}}];
  var h='<table><thead><tr>';
  for(var c=0;c<cols.length;c++){{var col=cols[c],act=sortKey===col.k;h+='<th onclick="st(\\\\''+col.k+'\\\\')" tabindex="0" onkeydown="if(event.key===\\\'Enter\\\')st(\\\\''+col.k+'\\\\')">'+col.l+'<span class="sort-arrow'+(act?' active':'')+'">'+(act?(sortAsc?'\\u25B2':'\\u25BC'):'\\u25BD')+'</span></th>';}}
  h+='</tr></thead><tbody>';
  if(rows.length===0){{h+='<tr><td colspan="'+cols.length+'" class="no-data">No channels match your filter.</td></tr>';}}else{{
    for(var i=0;i<rows.length;i++){{var r=rows[i],dc=r.flag==='Healthy'?'#22c55e':r.flag==='Watch'?'#f59e0b':r.flag==='At-Risk'?'#ef4444':'#6b7280';var ac='action-auto';if(r.action.indexOf('Urgent')>=0)ac='action-urgent';else if(r.action.indexOf('Check')>=0)ac='action-check';else if(r.action.indexOf('Priority')>=0||r.action.indexOf('Monitor')>=0)ac='action-monitor';
    h+='<tr><td><span class="risk-dot" style="background:'+dc+'"></span>'+esc(r.title)+'</td><td>'+r.subs+'</td><td>'+r.freq+'</td><td>'+r.mom+'</td><td>'+r.eng+'</td><td>'+r.days+'</td><td>'+r.risk_s+'</td><td>'+r.conf+'</td><td>'+esc(r.cluster)+'</td><td><span class="action-badge '+ac+'">'+esc(r.action)+'</span></td></tr>';
    }}
  }}
  h+='</tbody></table>';
  document.getElementById('tc').innerHTML=h;
}}

function st(k){{if(sortKey===k)sortAsc=!sortAsc;else{{sortKey=k;sortAsc=k==='risk_raw'}}renderTable();}}

document.getElementById('tf').addEventListener('change',renderTable);
document.getElementById('search').addEventListener('input',renderTable);

function exportCSV() {{
  var rows=D.table_data.slice();
  var headers=['Channel','Subscribers','Freq 30d','Momentum','Engagement','Days Idle','Risk Score','Confidence','Cluster','Risk Flag','Action'];
  var csv=headers.join(',')+'\\n';
  for(var i=0;i<rows.length;i++){{var r=rows[i];csv+='"'+esc(r.title).replace(/"/g,'""')+'",'+r.subs+','+r.freq+','+r.mom+','+r.eng+','+r.days+','+r.risk_s+','+r.conf+',"'+esc(r.cluster).replace(/"/g,'""')+'","'+r.flag+'","'+esc(r.action).replace(/"/g,'""')+'"\\n';}}
  var blob=new Blob([csv],{{type:'text/csv'}});
  var a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='creator_dashboard_export.csv';a.click();
}}

if(typeof Plotly!=='undefined'){{render();}}else{{setTimeout(function(){{if(typeof Plotly!=='undefined'){{render();}}else{{err('Plotly library did not load.');}}}},5000);}}
</script>
</body>
</html>"""

    out_path = ROOT / "docs" / "index.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    print(f"Dashboard generated: {out_path}")


if __name__ == "__main__":
    gen_dashboard()
