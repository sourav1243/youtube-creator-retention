from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

PIPELINE_SCRIPT = Path(__file__).resolve().parent.parent / "src" / "run_pipeline.py"


def run_pipeline():
    """Execute the full pipeline as a subprocess."""
    logger.info("Scheduled pipeline run starting...")
    result = subprocess.run(
        [sys.executable, "-m", "src.run_pipeline"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("Scheduled pipeline run completed successfully")
    else:
        logger.error("Scheduled pipeline run failed:\n%s", result.stderr)
    return result.returncode


def schedule_weekly():
    """Print instructions for setting up weekly cron/scheduled task.

    Windows: Use Task Scheduler
    Linux/Mac: Use crontab

    Example crontab entry (runs every Monday at 9 AM):
        0 9 * * 1 cd /path/to/project && /usr/bin/python3 -m infrastructure.scheduler
    """
    msg = [
        "=" * 60,
        "SCHEDULER SETUP",
        "=" * 60,
        "",
        "To schedule the pipeline to run weekly:",
        "",
        "Linux/Mac (crontab):",
        "  0 9 * * 1 cd /path/to/project && /usr/bin/python3 -m infrastructure.scheduler",
        "",
        "Windows (Task Scheduler):",
        "  1. Open Task Scheduler",
        "  2. Create Basic Task \u2192 Weekly",
        '  3. Action: Start a Program \u2192 "C:\\path\\to\\python.exe"',
        '  4. Arguments: "-m infrastructure.scheduler"',
        '  5. Start in: "C:\\path\\to\\project"',
        "",
        "To run once immediately:",
        "  python -m infrastructure.scheduler",
        "=" * 60,
    ]
    print("\n".join(msg))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--setup" in sys.argv:
        schedule_weekly()
    else:
        sys.exit(run_pipeline())
