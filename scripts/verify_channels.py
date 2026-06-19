import csv
import os
import sys

import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(".env")
key = os.getenv("YOUTUBE_API_KEY")

with open("data/raw/seed_channel_ids.csv", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

valid = []
invalid = []
for row in rows:
    cid = row["channel_id"]
    r = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "snippet", "id": cid, "key": key},
    )
    items = r.json().get("items", [])
    if items:
        name = items[0]["snippet"]["title"]
        print(f"  OK  {cid} -> {name}")
        valid.append(row)
    else:
        print(f"  BAD {cid} ({row['name']})")
        invalid.append(row)

print(f"\nValid: {len(valid)}/{len(rows)}, Invalid: {len(invalid)}")
with open("data/raw/seed_channel_ids.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["channel_id", "name", "niche"])
    writer.writeheader()
    writer.writerows(valid)
print("Wrote verified seed list to data/raw/seed_channel_ids.csv")
