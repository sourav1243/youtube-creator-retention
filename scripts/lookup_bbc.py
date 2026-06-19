import os
import requests
from dotenv import load_dotenv
load_dotenv(".env")
key = os.getenv("YOUTUBE_API_KEY")
r = requests.get("https://www.googleapis.com/youtube/v3/channels", params={"part": "id,snippet", "forHandle": "BBCNews", "key": key})
items = r.json().get("items", [])
if items:
    print(items[0]["id"] + " (" + items[0]["snippet"]["title"] + ")")
else:
    print("NOT FOUND for BBCNews")
    r2 = requests.get("https://www.googleapis.com/youtube/v3/channels", params={"part": "id,snippet", "forHandle": "BBC", "key": key})
    items2 = r2.json().get("items", [])
    if items2:
        print(items2[0]["id"] + " (" + items2[0]["snippet"]["title"] + ")")
    else:
        print("NOT FOUND for BBC either")
