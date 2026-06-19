"""
Quota budget calculator for YouTube Data API v3.

Computes concrete API call costs and designs a tiering strategy
that fits within daily quota limits.

Call costs (YouTube Data API v3 pricing):
  - channels.list:       1 unit  per call (up to 50 IDs)
  - playlistItems.list:  1 unit  per page (up to 50 items)
  - videos.list:          1 unit  per call (up to 50 IDs)
  - search.list:         100 units per call (up to 50 results)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from src.config import settings

logger = logging.getLogger(__name__)


@dataclass
class QuotaEstimate:
    stage: str
    api_calls: int
    cost_per_call: int
    total_units: int
    description: str


@dataclass
class QuotaPlan:
    n_channels_total: int
    n_channels_tier_b: int
    avg_videos_per_channel: int
    daily_quota: int
    estimates: list[QuotaEstimate]

    @property
    def total_cost_full(self) -> int:
        return sum(e.total_units for e in self.estimates if "Tier B" in e.stage or "Tier A" in e.stage)

    @property
    def total_cost_tier_a_only(self) -> int:
        return sum(e.total_units for e in self.estimates if "Tier A" in e.stage)

    @property
    def days_needed_for_full(self) -> int:
        return math.ceil(self.total_cost_full / self.daily_quota)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "QUOTA BUDGET CALCULATION",
            "=" * 60,
            f"Total channels:               {self.n_channels_total}",
            f"Tier B (video-level) sample:  {self.n_channels_tier_b}",
            f"Avg videos/channel (est.):    {self.avg_videos_per_channel}",
            f"Daily quota:                  {self.daily_quota} units",
            "",
            "Breakdown:",
        ]
        for est in self.estimates:
            lines.append(f"  {est.stage:40s} {est.api_calls:6d} calls × {est.cost_per_call:3d} = {est.total_units:8,d} units  | {est.description}")
        lines.append("")
        lines.append(f"  Tier A only (channel-level):       {self.total_cost_tier_a_only:>8,d} units")
        lines.append(f"  Full pipeline (A + B, all {self.n_channels_total}):      {self.total_cost_full:>8,d} units")
        lines.append(f"  Days needed (full, single key):    {self.days_needed_for_full:>8d} days")
        lines.append("")
        lines.append("TIERING STRATEGY:")
        lines.append(f"  Tier A — channels.list for ALL {self.n_channels_total} channels: ~{self.total_cost_tier_a_only} units (trivial, do for all)")
        lines.append(f"  Tier B — video-level extraction for {self.n_channels_tier_b} channels: fits in 1 day with headroom")
        lines.append(f"  Future: resume multi-day extraction using manifest checkpoint for remaining channels")
        lines.append("=" * 60)
        return "\n".join(lines)


def compute_quota_plan(
    n_channels: int | None = None,
    n_tier_b: int | None = None,
    avg_videos: int | None = None,
    daily_quota: int | None = None,
) -> QuotaPlan:
    n_channels = n_channels or settings.pipeline.n_channels_total
    n_tier_b = n_tier_b or settings.pipeline.sample_size_tier_b
    avg_videos = avg_videos or 200  # measured estimate, documented in DECISIONS.md
    daily_quota = daily_quota or settings.extraction.quota_daily_default

    max_per_call = settings.extraction.max_ids_per_call

    estimates: list[QuotaEstimate] = []

    # Tier A: channels.list for ALL channels
    calls_a = math.ceil(n_channels / max_per_call)
    estimates.append(QuotaEstimate(
        stage="Tier A — channels.list (all channels)",
        api_calls=calls_a,
        cost_per_call=1,
        total_units=calls_a * 1,
        description=f"{n_channels} channels at {max_per_call}/call",
    ))

    # Tier A: playlistItems for uploads playlist ID fetched via channels.list contentDetails
    # Actually, playlistItems is per-channel to get video IDs; but the uploads playlist ID
    # comes from channels.list contentDetails (included in Tier A call above).
    # The cost is for getting the uploads playlist items.

    # Tier B: playlistItems.list per channel in sample
    calls_pl = n_tier_b * math.ceil(avg_videos / max_per_call)
    estimates.append(QuotaEstimate(
        stage="Tier B — playlistItems.list (sample)",
        api_calls=calls_pl,
        cost_per_call=1,
        total_units=calls_pl * 1,
        description=f"{n_tier_b} channels × {avg_videos} videos, paged at {max_per_call}",
    ))

    # Tier B: videos.list per batch of video IDs in sample
    calls_vid = n_tier_b * math.ceil(avg_videos / max_per_call)
    estimates.append(QuotaEstimate(
        stage="Tier B — videos.list (sample)",
        api_calls=calls_vid,
        cost_per_call=1,
        total_units=calls_vid * 1,
        description=f"{n_tier_b} channels × {avg_videos} videos, batched at {max_per_call}",
    ))

    return QuotaPlan(
        n_channels_total=n_channels,
        n_channels_tier_b=n_tier_b,
        avg_videos_per_channel=avg_videos,
        daily_quota=daily_quota,
        estimates=estimates,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    plan = compute_quota_plan()
    print(plan.summary())
