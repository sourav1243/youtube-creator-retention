import os
import sys

import requests
from dotenv import load_dotenv

sys.stdout.reconfigure(encoding="utf-8")

load_dotenv(".env")
key = os.getenv("YOUTUBE_API_KEY")

handles = {
    "MKBHD": "@mkbhd",
    "Fireship": "@Fireship",
    "NetworkChuck": "@NetworkChuck",
    "TechWithTim": "@TechWithTim",
    "Theo - t3.gg": "@t3dotgg",
    "Kurzgesagt": "@Kurzgesagt",
    "Veritasium": "@veritasium",
    "Vsauce": "@Vsauce",
    "3Blue1Brown": "@3blue1brown",
    "CrashCourse": "@crashcourse",
    "Graham Stephan": "@GrahamStephan",
    "Andrei Jikh": "@andreijikh",
    "Meet Kevin": "@MeetKevin",
    "Tommyinnit": "@tommyinnit",
    "Dream": "@dream",
    "Marques Brownlee": "@mkbhd",
}

for name, handle in handles.items():
    r = requests.get(
        "https://www.googleapis.com/youtube/v3/channels",
        params={"part": "id,snippet", "forHandle": handle.lstrip("@"), "key": key},
    )
    items = r.json().get("items", [])
    if items:
        cid = items[0]["id"]
        title = items[0]["snippet"]["title"]
        print(f"  {name}: {cid} ({title})")
    else:
        print(f"  {name}: NOT FOUND")
