from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

import pandas as pd

logger = logging.getLogger(__name__)


def send_email_alert(
    subject: str,
    body: str,
    to: str | None = None,
) -> bool:
    """Send email alert via SMTP. Configure via env vars SMTP_*."""
    smtp_host = os.getenv("SMTP_HOST", "")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    smtp_from = os.getenv("SMTP_FROM", smtp_user)
    smtp_to = to or os.getenv("ALERT_EMAIL", "")

    if not all([smtp_host, smtp_user, smtp_pass, smtp_to]):
        logger.info(
            "SMTP not configured \u2014 email alert skipped. "
            "To enable, set SMTP_HOST, SMTP_USER, SMTP_PASS, ALERT_EMAIL"
        )
        return False

    try:
        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = smtp_to

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info("Alert email sent to %s: %s", smtp_to, subject)
        return True
    except Exception as e:
        logger.warning("Failed to send alert email: %s", e)
        return False


def alert_new_at_risk(
    report_df: pd.DataFrame,
    previous_report: list[str] | None = None,
) -> tuple[list[str], bool]:
    """Alert if new At-Risk creators appear."""
    current_at_risk = report_df[report_df["risk_flag"] == "At-Risk"]["channel_id"].tolist()
    if previous_report is None:
        previous_at_risk: list[str] = []
        try:
            from src.config import ROOT_DIR
            prev = ROOT_DIR / "reports" / "at_risk_creators.csv"
            if prev.exists():
                prev_df = pd.read_csv(prev)
                previous_at_risk = prev_df[prev_df["risk_flag"] == "At-Risk"]["channel_id"].tolist()
        except Exception:
            pass
    else:
        previous_at_risk = previous_report

    new_at_risk = [cid for cid in current_at_risk if cid not in previous_at_risk]
    if new_at_risk:
        body = f"New At-Risk creators detected: {len(new_at_risk)}\n\n"
        new_rows = report_df[report_df["channel_id"].isin(new_at_risk)]
        for _, r in new_rows.iterrows():
            body += f"  - {r.get('title', 'Unknown')} ({r.get('channel_id', '')})\n"
            body += f"    Days since upload: {r.get('days_since_last_upload', 'N/A')}\n"
            body += f"    Risk score: {r.get('risk_score', 'N/A')}\n\n"
        sent = send_email_alert(
            f"[YouTube Retention] {len(new_at_risk)} new At-Risk creator(s)",
            body,
        )
        return new_at_risk, sent

    return [], False
