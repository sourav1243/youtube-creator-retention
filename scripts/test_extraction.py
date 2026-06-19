import logging
import sys

print("starting...")
sys.stdout.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

from src.extraction.youtube_client import YouTubeClient

print("imported")
sys.stdout.flush()

client = YouTubeClient()
print("got client")
sys.stdout.flush()

channels = client.get_channels(["UCsBjURrPoezykLs9EqgamOA"])
print("got channels: " + str(len(channels)))
sys.stdout.flush()

c = channels[0]
playlist_id = c["contentDetails"]["relatedPlaylists"]["uploads"]
print("Uploads playlist: " + playlist_id)
sys.stdout.flush()

items = client.get_playlist_items(playlist_id, max_results=50, max_pages=1)
print("Got " + str(len(items)) + " playlist items")
sys.stdout.flush()

if items:
    video_ids = []
    for item in items:
        vid = item.get("contentDetails", {}).get("videoId")
        if vid:
            video_ids.append(vid)
    print("Video IDs: " + str(len(video_ids)))
    sys.stdout.flush()
    videos = client.get_videos(video_ids[:5])
    print("Got " + str(len(videos)) + " video details")
    for v in videos:
        title = v.get("snippet", {}).get("title", "?")
        views = v.get("statistics", {}).get("viewCount", "?")
        print("  " + v["id"] + ": " + title + " (" + str(views) + " views)")
