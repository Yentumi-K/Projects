"""Shared client for the API-Football (v3) REST API.

Centralises the base URL, authentication header, request throttling and the
JSON-saving helper so the individual fetch scripts (`fetch_data.py`,
`fetch_player_profiles.py`) don't each re-implement them.

The API key is read lazily from the ``API_FOOTBALL_KEY`` environment variable
the first time a request is made, so this module can be imported (for reading
or testing the fetch functions) without a key present.
"""

import os
import json
import time

import requests

BASE_URL = "https://v3.football.api-sports.io"

# Seconds to wait between requests to stay within the API rate limits.
REQUEST_DELAY = 0.25


def _headers():
    """Build the auth header, raising a clear error if the key is missing."""
    key = os.getenv("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError(
            "API_FOOTBALL_KEY environment variable is not set. "
            "Set it before running the fetch scripts (see README / .env.example)."
        )
    return {"x-apisports-key": key}


def safe_get(path, params=None):
    """Send a GET request to ``BASE_URL + path`` and return the parsed JSON.

    Raises ``requests.HTTPError`` for non-2xx responses so callers can decide
    how to handle failures.
    """
    url = f"{BASE_URL}{path}"
    response = requests.get(url, headers=_headers(), params=params or {}, timeout=15)
    response.raise_for_status()
    return response.json()


def save_json(data, path):
    """Write ``data`` to ``path`` as indented JSON, creating parent dirs."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf8") as f:
        json.dump(data, f, indent=2)


def throttle():
    """Pause briefly between requests to respect the API rate limit."""
    time.sleep(REQUEST_DELAY)
