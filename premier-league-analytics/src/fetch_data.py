"""Download raw Premier League data from API-Football for seasons 2020-2024.

For each season this script pulls fixtures, per-fixture statistics and injuries,
team rosters and season statistics, standings, scorer/assist/card leaderboards
and head-to-head records, saving each response as JSON under ``data/raw/``.

Requires the ``API_FOOTBALL_KEY`` environment variable (see README). The data
for this project has already been collected, so this script is only needed to
refresh it.
"""

import itertools

from api_client import safe_get, save_json, throttle

LEAGUE_ID = 39          # Premier League
SEASONS = range(2020, 2025)


# --- Endpoint wrappers -------------------------------------------------------

def fetch_fixtures(season, league):
    """Return all fixtures for a league/season."""
    return safe_get("/fixtures", {"season": season, "league": league})


def fetch_fixture_statistics(fixture_id):
    """Return the box-score statistics for a single fixture."""
    return safe_get("/fixtures/statistics", {"fixture": fixture_id})


def fetch_fixture_injuries(fixture_id):
    """Return the injury list reported for a single fixture."""
    return safe_get("/injuries", {"fixture": fixture_id})


def fetch_teams(league, season):
    """Return the teams competing in a league/season."""
    return safe_get("/teams", {"league": league, "season": season})


def fetch_team_statistics(league, season, team_id):
    """Return aggregated season statistics for one team."""
    return safe_get(
        "/teams/statistics",
        {"league": league, "season": season, "team": team_id},
    )


def fetch_headtohead(team1, team2):
    """Return the head-to-head fixture history between two teams."""
    return safe_get("/fixtures/headtohead", {"h2h": f"{team1}-{team2}"})


def fetch_standings(league, season):
    """Return the final league table for a league/season."""
    return safe_get("/standings", {"league": league, "season": season})


def fetch_topscorers(league, season):
    """Return the top scorers leaderboard for a league/season."""
    return safe_get("/players/topscorers", {"league": league, "season": season})


def fetch_topassists(league, season):
    """Return the top assists leaderboard for a league/season."""
    return safe_get("/players/topassists", {"league": league, "season": season})


def fetch_topyellowcards(league, season):
    """Return the most-booked players leaderboard for a league/season."""
    return safe_get("/players/topyellowcards", {"league": league, "season": season})


def fetch_topredcards(league, season):
    """Return the most-sent-off players leaderboard for a league/season."""
    return safe_get("/players/topredcards", {"league": league, "season": season})


# --- Per-season collection ---------------------------------------------------

def collect_season(season):
    """Fetch and save every raw dataset for a single season."""
    print(f"PROCESSING SEASON {season}")

    # Fixtures
    fixtures = fetch_fixtures(season, LEAGUE_ID)
    fixture_list = fixtures.get("response", [])
    fixture_ids = [f["fixture"]["id"] for f in fixture_list]
    save_json(fixtures, f"data/raw/fixtures_{LEAGUE_ID}_{season}.json")
    print("Total fixtures:", len(fixture_ids))
    throttle()

    # Per-fixture statistics
    all_fixture_stats = {}
    for fid in fixture_ids:
        try:
            all_fixture_stats[fid] = fetch_fixture_statistics(fid)
            throttle()
        except Exception as e:
            print(f"Error fetching stats for fixture {fid}: {e}")
    save_json(all_fixture_stats, f"data/raw/fixture_statistics_{LEAGUE_ID}_{season}.json")

    # Per-fixture injuries
    all_injuries = {}
    for fid in fixture_ids:
        try:
            all_injuries[fid] = fetch_fixture_injuries(fid)
            throttle()
        except Exception as e:
            print(f"Error fetching injuries for fixture {fid}: {e}")
    save_json(all_injuries, f"data/raw/injuries_{LEAGUE_ID}_{season}.json")

    # Teams
    teams_data = fetch_teams(LEAGUE_ID, season)
    team_list = teams_data.get("response", [])
    team_ids = [team["team"]["id"] for team in team_list]
    save_json(teams_data, f"data/raw/teams_{LEAGUE_ID}_{season}.json")
    print("Total teams:", len(team_ids))
    throttle()

    # Team season statistics
    all_team_stats = {}
    for team_id in team_ids:
        try:
            all_team_stats[team_id] = fetch_team_statistics(LEAGUE_ID, season, team_id)
            throttle()
        except Exception as e:
            print(f"Error fetching team stats for {team_id}: {e}")
    save_json(all_team_stats, f"data/raw/team_statistics_{LEAGUE_ID}_{season}.json")

    # Standings
    save_json(fetch_standings(LEAGUE_ID, season),
              f"data/raw/standings_{LEAGUE_ID}_{season}.json")
    throttle()

    # Leaderboards
    for name, fetch_fn in (
        ("topscorers", fetch_topscorers),
        ("topassists", fetch_topassists),
        ("topyellowcards", fetch_topyellowcards),
        ("topredcards", fetch_topredcards),
    ):
        save_json(fetch_fn(LEAGUE_ID, season),
                  f"data/raw/{name}_{LEAGUE_ID}_{season}.json")
        throttle()

    # Head-to-head for every pair of teams
    print("Generating head-to-head pairs")
    all_h2h = {}
    for team1, team2 in itertools.combinations(team_ids, 2):
        try:
            all_h2h[f"{team1}-{team2}"] = fetch_headtohead(team1, team2)
            throttle()
        except Exception as e:
            print(f"Error fetching H2H {team1}-{team2}: {e}")
    save_json(all_h2h, f"data/raw/headtohead_{LEAGUE_ID}_{season}.json")

    print(f"Season {season} complete")


if __name__ == "__main__":
    for SEASON in SEASONS:
        collect_season(SEASON)
    print("\nALL SEASONS COMPLETE.")
