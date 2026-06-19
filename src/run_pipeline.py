#!/usr/bin/env python3
"""
End-to-end pipeline runner.

Usage:
    python -m src.run_pipeline

Executes: extraction -> load -> clean/engineer -> duckdb eda -> cluster -> report
"""

from __future__ import annotations

import logging
import sys

from src.config import ROOT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(ROOT_DIR / "logs" / "pipeline.log", mode="a"),
    ],
)

logger = logging.getLogger("pipeline")


def run_pipeline():
    logger.info("=" * 60)
    logger.info("YOUTUBE CREATOR RETENTION PIPELINE")
    logger.info("=" * 60)

    # Phase 2-3: Extraction
    logger.info("[1/7] Extracting seed channels (Tier A)...")
    from src.extraction.extract_channels import extract_channels_from_seed, extract_channels_tier_a

    channel_ids = extract_channels_from_seed()
    channels = extract_channels_tier_a(channel_ids)
    logger.info("  %d channels extracted", len(channels))

    logger.info("[2/7] Extracting videos (Tier B)...")
    from src.extraction.youtube_client import YouTubeClient
    from src.extraction.extract_videos import extract_videos_tier_b

    client = YouTubeClient()
    pairs = []
    for c in channels:
        cid = c.get("id")
        upl = c.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
        if cid and upl:
            pairs.append((cid, upl))
    videos = extract_videos_tier_b(pairs, client=client)
    logger.info("  %d videos extracted from %d channels", len(videos), len(pairs))

    # Phase 4: Load to MySQL
    logger.info("[3/7] Loading to MySQL...")
    from src.load.load_mysql import load_all

    try:
        total_channels, total_videos = load_all()
        logger.info("  Loaded: %d channels, %d videos", total_channels, total_videos)
    except Exception as e:
        logger.warning("  MySQL load skipped (not available): %s", e)

    # Phase 5: Clean & Feature Engineer
    logger.info("[4/7] Cleaning and feature engineering...")
    from src.features.engineer import run_feature_pipeline

    features_df = run_feature_pipeline()
    logger.info("  Features computed for %d channels", len(features_df))

    from src.features.engineer import save_histograms
    save_histograms(features_df)

    # Phase 6: DuckDB EDA
    logger.info("[5/7] DuckDB analytical layer...")
    from src.analysis.duckdb_setup import setup_and_analyze

    conn = setup_and_analyze()
    conn.close()

    # Phase 7: Clustering
    logger.info("[6/7] K-Means clustering...")
    from src.modeling.cluster import run_clustering_pipeline

    result = run_clustering_pipeline()
    logger.info("  Cluster distribution:\n%s", result["cluster_label"].value_counts())

    # Phase 8: Reporting
    logger.info("[7/7] Generating at-risk report...")
    from src.reporting.at_risk_report import generate_report

    report = generate_report()
    risk_counts = report["risk_flag"].value_counts()
    logger.info("  Risk distribution:\n%s", risk_counts)
    logger.info("  Report saved to reports/at_risk_creators.csv")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
