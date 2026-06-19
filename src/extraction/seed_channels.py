from __future__ import annotations

import csv
import logging
import re
from pathlib import Path

from src.config import ROOT_DIR

logger = logging.getLogger(__name__)

CHANNEL_ID_PATTERN = re.compile(r"^UC[A-Za-z0-9_-]{22}$")

CURATED_CHANNELS: list[dict[str, str]] = [
    # Tech
    {"channel_id": "UCXuqSBlHAE6Xw-yeJA0Tunw", "name": "Linus Tech Tips", "niche": "tech"},
    {"channel_id": "UCBJycsmduvYEL83R_U4JriQ", "name": "Marques Brownlee", "niche": "tech"},
    {"channel_id": "UCsBjURrPoezykLs9EqgamOA", "name": "Fireship", "niche": "tech"},
    {"channel_id": "UC9x0AN7BWHpCDHSm9NiJFJQ", "name": "NetworkChuck", "niche": "tech"},
    {"channel_id": "UC8butISFwT-Wl7EV0hUK0BQ", "name": "freeCodeCamp", "niche": "tech"},
    {"channel_id": "UC4JX40jDee_tINbkjycV4Sg", "name": "Tech With Tim", "niche": "tech"},
    {"channel_id": "UCbRP3c757lWg9M-U7TyEkXA", "name": "Theo - t3.gg", "niche": "tech"},
    # Gaming / Entertainment
    {"channel_id": "UC-lHJZR3Gqxm24_Vd_AJ5Yw", "name": "PewDiePie", "niche": "gaming"},
    {"channel_id": "UCX6OQ3DkcsbYNE6H8uQQuVA", "name": "MrBeast", "niche": "entertainment"},
    {"channel_id": "UCJ5v_MCY6GNUBTO8-D3XoAg", "name": "WWE", "niche": "entertainment"},
    {"channel_id": "UCvlE5gTbOvjiolFlEm-c_Ow", "name": "Vlad and Niki", "niche": "entertainment"},
    {"channel_id": "UCk8GzjMOrta8yxDcKfylJYw", "name": "Kids Diana Show", "niche": "entertainment"},
    {"channel_id": "UCJplp5SjeGSdVdwsfb9Q7lQ", "name": "Like Nastya", "niche": "entertainment"},
    {"channel_id": "UCTkXRDQl0luXxVQrRQvWS6w", "name": "Dream", "niche": "gaming"},
    {"channel_id": "UC5p_l5ZeB_wGjO_yDXwiqvw", "name": "TommyInnit", "niche": "gaming"},
    # Education / Science
    {"channel_id": "UCHnyfMqiRRG1u-2MsSQLbXA", "name": "Veritasium", "niche": "education"},
    {"channel_id": "UCsXVk37bltHxD1rDPwtNM8Q", "name": "Kurzgesagt", "niche": "education"},
    {"channel_id": "UCYO_jab_esuFRV4b17AJtAw", "name": "3Blue1Brown", "niche": "education"},
    {"channel_id": "UCX6b17PVsYBQ0ip5gyeme-Q", "name": "CrashCourse", "niche": "education"},
    {"channel_id": "UC6nSFpj9HTCZ5t-N3Rm3-HA", "name": "Vsauce", "niche": "education"},
    {"channel_id": "UC295-Dw_tDNtZXFeAPAW6Aw", "name": "5-Minute Crafts", "niche": "education"},
    {"channel_id": "UC16niRr50-MSBwiO3YDb3RA", "name": "BBC News", "niche": "news"},
    # Music
    {"channel_id": "UCq-Fj5jknLsUf-MWSy4_brA", "name": "T-Series", "niche": "music"},
    {"channel_id": "UCOmHUn--16B90oW2L6FRR3A", "name": "BLACKPINK", "niche": "music"},
    {"channel_id": "UCLkAepWjdylmXSltofFvsYQ", "name": "BANGTANTV", "niche": "music"},
    {"channel_id": "UCFFbwnve3yF62-tVXkTyHqg", "name": "Zee Music Company", "niche": "music"},
    {"channel_id": "UCcdwLMPsaU2ezNSJU1nFoBQ", "name": "Pinkfong Kids Songs", "niche": "music"},
    {"channel_id": "UCbCmjCuTUZos6Inko4u57UQ", "name": "Cocomelon", "niche": "music"},
    # Finance
    {"channel_id": "UCV6KDgJskWaEckne5aPA0aQ", "name": "Graham Stephan", "niche": "finance"},
    {"channel_id": "UCGy7SkBjcIAgTiwkXEtPnYg", "name": "Andrei Jikh", "niche": "finance"},
    {"channel_id": "UCUvvj5lwue7PspotMDjk5UA", "name": "Meet Kevin", "niche": "finance"},
    # Entertainment / TV
    {"channel_id": "UCpEhnqL0y41EpW2TvWAHD7Q", "name": "SET India", "niche": "entertainment"},
    {"channel_id": "UCyoXW-Dse7fURq30EWl_CUA", "name": "Goldmines", "niche": "entertainment"},
    {"channel_id": "UC6-F5tO8uklgE9Zy8IvbdFw", "name": "Sony SAB", "niche": "entertainment"},
    {"channel_id": "UC55IWqFLDH1Xp7iu1_xknRA", "name": "Colors TV", "niche": "entertainment"},
    {"channel_id": "UCppHT7SZKKvar4Oc9J4oljQ", "name": "Zee TV", "niche": "entertainment"},
    {"channel_id": "UCupvZG-5ko_eiXAupbDfxWw", "name": "CNN", "niche": "news"},
]


def validate_channel_id(channel_id: str) -> bool:
    return bool(CHANNEL_ID_PATTERN.match(channel_id))


def build_seed_list(output_path: str | Path | None = None) -> list[dict[str, str]]:
    seen: set[str] = set()
    validated: list[dict[str, str]] = []

    for entry in CURATED_CHANNELS:
        cid = entry["channel_id"]
        if cid in seen:
            logger.warning("Duplicate channel_id skipped: %s (%s)", cid, entry["name"])
            continue
        if not validate_channel_id(cid):
            logger.warning("Invalid channel_id format skipped: %s (%s)", cid, entry["name"])
            continue
        seen.add(cid)
        validated.append(entry)

    logger.info("Seed channels: %d total, %d unique, %d validated", len(CURATED_CHANNELS), len(seen), len(validated))

    if output_path is None:
        output_path = ROOT_DIR / "data" / "raw" / "seed_channel_ids.csv"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["channel_id", "name", "niche"])
        writer.writeheader()
        writer.writerows(validated)

    if not validated:
        logger.warning("Seed list is empty — no channels will be extracted")
    else:
        logger.info("Seed list written to %s (%d channels)", output_path, len(validated))
    return validated


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_seed_list()
