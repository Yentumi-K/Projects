"""Download every player profile available from API-Football.

The ``/players/profiles`` endpoint is paginated; this script walks every page,
saves each page individually under ``data/raw/player_profiles_pages/`` and then
writes a combined master file at ``data/raw/player_profiles_all.json``.

Requires the ``API_FOOTBALL_KEY`` environment variable (see README).
"""

import os
import json

from api_client import safe_get, throttle


def fetch_all_player_profiles():
    """Page through the player-profiles endpoint and save the full set."""
    os.makedirs("data/raw/player_profiles_pages", exist_ok=True)

    page = 1
    all_players = []

    while True:
        print(f"Fetching page {page}...")
        response = safe_get("/players/profiles", {"page": page})
        players = response.get("response", [])

        with open(f"data/raw/player_profiles_pages/page_{page}.json",
                  "w", encoding="utf8") as f:
            json.dump(players, f, indent=2)

        all_players.extend(players)

        paging = response.get("paging", {})
        current = paging.get("current")
        total = paging.get("total")
        print(f"Page {current} of {total}")

        if current >= total:
            print("All pages downloaded.")
            break

        page += 1
        throttle()

    with open("data/raw/player_profiles_all.json", "w", encoding="utf8") as f:
        json.dump(all_players, f, indent=2)

    print("Total players collected:", len(all_players))


if __name__ == "__main__":
    fetch_all_player_profiles()
