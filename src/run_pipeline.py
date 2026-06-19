#!/usr/bin/env python3
"""
End-to-end pipeline runner.

Usage:
    python -m src.run_pipeline

Executes: extraction -> load -> clean/engineer -> duckdb eda -> cluster -> report
Exits with non-zero code on critical failure.
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler

from src.config import ROOT_DIR, settings

log_path = ROOT_DIR / settings.logging.file
log_path.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=getattr(logging, settings.logging.level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        RotatingFileHandler(
            log_path,
            mode="a",
            maxBytes=settings.logging.max_bytes,
            backupCount=settings.logging.backup_count,
            encoding="utf-8",
        ),
    ],
)

logger = logging.getLogger("pipeline")


def run_pipeline():
    logger.info("=" * 60)
    logger.info("YOUTUBE CREATOR RETENTION PIPELINE")
    logger.info("=" * 60)

    # Phase 2-3: Extraction
    logger.info("[1/7] Extracting seed channels (Tier A)...")
    from src.extraction.extract_channels import (
        extract_channels_from_seed,
        extract_channels_tier_a,
    )
    from src.extraction.extract_videos import extract_videos_tier_b
    from src.extraction.youtube_client import YouTubeClient

    channel_ids = extract_channels_from_seed()
    if not channel_ids:
        logger.error("No seed channels found — aborting pipeline")
        sys.exit(1)

    channels = extract_channels_tier_a(channel_ids)
    logger.info("  %d channels extracted", len(channels))

    if not channels:
        logger.warning("No channel data extracted — skipping video extraction")
        videos = []
    else:
        logger.info("[2/7] Extracting videos (Tier B)...")
        client = YouTubeClient()
        pairs = []
        skipped_channels = []
        for c in channels:
            cid = c.get("id")
            upl = c.get("contentDetails", {}).get("relatedPlaylists", {}).get("uploads")
            if cid and upl:
                pairs.append((cid, upl))
            else:
                skipped_channels.append(cid or "unknown")
        if skipped_channels:
            logger.info(
                "  %d channels have no uploads playlist: %s", len(skipped_channels), ", ".join(skipped_channels[:5])
            )
        if not pairs:
            logger.warning("No channel uploads playlists found — no videos to extract")
            videos = []
        else:
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
    from src.features.engineer import run_feature_pipeline, save_histograms

    features_df = run_feature_pipeline()
    logger.info("  Features computed for %d channels", len(features_df))

    if features_df.empty:
        logger.error("No features computed — aborting pipeline")
        sys.exit(1)

    save_histograms(features_df)

    # Phase 6: DuckDB EDA
    logger.info("[5/7] DuckDB analytical layer...")
    from src.analysis.duckdb_setup import setup_and_analyze

    try:
        conn = setup_and_analyze()
        conn.close()
    except FileNotFoundError as e:
        logger.warning("  DuckDB EDA skipped: %s", e)

    # Phase 7: Clustering
    logger.info("[6/7] K-Means clustering...")
    from src.modeling.cluster import run_clustering_pipeline

    result = run_clustering_pipeline()
    if result.empty:
        logger.warning("Clustering produced no results — skipping reporting")
        sys.exit(1)
    logger.info("  Cluster distribution:\n%s", result["cluster_label"].value_counts())

    # Phase 8: Reporting
    logger.info("[7/7] Generating at-risk report...")
    from src.reporting.at_risk_report import generate_report

    report = generate_report()
    if report.empty:
        logger.warning("Report empty — skipping")
        sys.exit(1)
    risk_counts = report["risk_flag"].value_counts()
    logger.info("  Risk distribution:\n%s", risk_counts)
    logger.info("  Report saved to reports/at_risk_creators.csv")

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_pipeline()
