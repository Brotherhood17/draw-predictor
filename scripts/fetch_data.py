"""
Downloads historical match results + odds CSVs from football-data.co.uk
(free, no API key needed) and saves them under data/raw/.

football-data.co.uk organizes files by league code and season, e.g.:
  https://www.football-data.co.uk/mmz4281/2425/E0.csv   (England Premier League, 2024/25)

Edit LEAGUES below to add/remove leagues or seasons.

NOTE: this script needs real internet access to football-data.co.uk.
It will not run inside a network-locked sandbox - run it on your own
machine or inside the GitHub Actions workflow (which has full internet
access).
"""

import os
import requests

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# league code -> human name (see https://www.football-data.co.uk/notes.txt for full list)
LEAGUE_CODES = {
    "E0": "premier_league",
    "E1": "championship",
    "SP1": "la_liga",
    "D1": "bundesliga",
    "I1": "serie_a",
    "F1": "ligue_1",
}

# seasons in football-data.co.uk's folder format: "2324" = 2023/24 season
SEASONS = ["2122", "2223", "2324", "2425"]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def fetch_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for season in SEASONS:
        for code, name in LEAGUE_CODES.items():
            url = f"{BASE_URL}/{season}/{code}.csv"
            out_path = os.path.join(OUTPUT_DIR, f"{name}_{season}.csv")
            try:
                resp = requests.get(url, timeout=20)
                resp.raise_for_status()
                with open(out_path, "wb") as f:
                    f.write(resp.content)
                print(f"saved {out_path} ({len(resp.content)} bytes)")
            except requests.RequestException as e:
                print(f"failed {url}: {e}")


if __name__ == "__main__":
    fetch_all()
