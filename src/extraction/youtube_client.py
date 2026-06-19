from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any

import requests
from tenacity import (
    before_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings

logger = logging.getLogger(__name__)

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


class QuotaExhausted(Exception):
    """Raised when daily quota is actually exhausted — not retryable."""


class _IsQuotaError:
    def __call__(self, exc: BaseException) -> bool:
        if isinstance(exc, QuotaExhausted):
            return False
        if isinstance(exc, requests.HTTPError):
            status = exc.response.status_code
            if status in (403, 429):
                body = exc.response.json()
                reason = body.get("error", {}).get("errors", [{}])[0].get("reason", "")
                if reason in ("quotaExceeded", "dailyLimitExceeded", "rateLimitExceeded"):
                    return True
            return status >= 500
        return isinstance(exc, (requests.ConnectionError, requests.Timeout))


class YouTubeClient:
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.youtube_api_key
        self.session = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    def _build_url(self, endpoint: str, params: dict[str, Any]) -> str:
        params["key"] = self.api_key
        return f"{YOUTUBE_API_BASE}/{endpoint}"

    @retry(
        retry=retry_if_exception(_IsQuotaError()),
        stop=stop_after_attempt(settings.extraction.retry_max_attempts),
        wait=wait_exponential(
            multiplier=settings.extraction.retry_base_delay_s,
            min=2,
            max=60,
        ),
        before=before_log(logger, logging.DEBUG),
    )
    def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self._build_url(endpoint, params)
        logger.debug("GET %s?%s", endpoint, "&".join(f"{k}={v}" for k, v in params.items()))
        resp = self.session.get(url, params={"key": self.api_key, **params}, timeout=30)

        if resp.status_code == 403:
            body = resp.json()
            reason = body.get("error", {}).get("errors", [{}])[0].get("reason", "")
            if reason in ("quotaExceeded", "dailyLimitExceeded"):
                logger.critical("Daily quota exhausted. Saving checkpoint and stopping.")
                raise QuotaExhausted(reason)

        resp.raise_for_status()
        return resp.json()

    def batch_ids(self, ids: list[str], max_per_call: int | None = None) -> list[list[str]]:
        max_per_call = max_per_call or settings.extraction.max_ids_per_call
        n_batches = math.ceil(len(ids) / max_per_call)
        return [ids[i * max_per_call : (i + 1) * max_per_call] for i in range(n_batches)]

    def get_channels(self, channel_ids: list[str], parts: str = "snippet,statistics,contentDetails") -> list[dict[str, Any]]:
        all_items: list[dict[str, Any]] = []
        for batch in self.batch_ids(channel_ids):
            data = self._request("channels", {"part": parts, "id": ",".join(batch)})
            all_items.extend(data.get("items", []))
        return all_items

    def get_playlist_items(self, playlist_id: str, max_results: int = 50, max_pages: int | None = None) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        page_token: str | None = None
        pages = 0
        while True:
            params: dict[str, Any] = {
                "part": "snippet,contentDetails",
                "playlistId": playlist_id,
                "maxResults": max_results,
            }
            if page_token:
                params["pageToken"] = page_token
            data = self._request("playlistItems", params)
            items.extend(data.get("items", []))
            pages += 1
            page_token = data.get("nextPageToken")
            if not page_token or (max_pages and pages >= max_pages):
                break
        return items

    def get_videos(self, video_ids: list[str], parts: str = "snippet,statistics,contentDetails") -> list[dict[str, Any]]:
        all_items: list[dict[str, Any]] = []
        for batch in self.batch_ids(video_ids):
            data = self._request("videos", {"part": parts, "id": ",".join(batch)})
            all_items.extend(data.get("items", []))
        return all_items


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
