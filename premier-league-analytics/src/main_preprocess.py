"""Convert the raw API-Football JSON into tidy, analysis-ready CSV files.

Every ``preprocess_*`` function reads the relevant raw JSON from ``data/raw``
and writes a flat CSV to ``data/processed`` with a fixed schema that the
feature-engineering, model-training and dashboard stages depend on. Run this
script (from the project root) after the raw data has been collected.
"""

import json
from pathlib import Path

import pandas as pd

SEASONS = [2020, 2021, 2022, 2023, 2024]
LEAGUE_ID = 39

# Anchor all paths to the project root so the script works regardless of the
# directory it is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --- Generic helpers ---------------------------------------------------------

def load_json(path):
    """Load and return JSON from ``path``.

    Returns ``None`` (after printing a notice) if the file is missing or cannot
    be parsed, so callers can skip it instead of crashing.
    """
    path = Path(path)
    if not path.exists():
        print(f"Missing file: {path.name}")
        return None
    try:
        with open(path, "r", encoding="utf8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Could not read {path.name}, skipping.")
        return None


def per_90(counts, minutes):
    """Convert a per-match count series into a per-90-minutes rate."""
    return (counts / minutes.replace(0, 1)) * 90


def clean_percentage(value):
    """Convert a percentage string (e.g. ``'55%'``) to a float, or 0.0 if empty."""
    if value is None or value == "":
        return 0.0
    return float(str(value).replace("%", ""))


def extract_stat(stat_list, stat_name):
    """Return the value of ``stat_name`` from a fixture stat list, or 0 if absent."""
    for s in stat_list:
        if s["type"] == stat_name:
            val = s["value"]
            return val if val is not None else 0
    return 0


def sum_card_totals(card_dict):
    """Sum the per-minute-bucket ``total`` values in a cards dictionary."""
    total = 0
    for _minute, data in card_dict.items():
        if data["total"] is not None:
            total += data["total"]
    return total


def build_player_stat_csv(raw_template, output_name, parse_item, add_metrics, label):
    """Build a per-season player leaderboard CSV.

    Loops over every season, loads the raw leaderboard JSON, runs ``parse_item``
    on each player entry to build one record, then applies ``add_metrics`` to
    derive per-90 columns before saving.

    Parameters
    ----------
    raw_template : str
        Filename template with a ``{season}`` placeholder, e.g.
        ``"topscorers_39_{season}.json"``.
    output_name : str
        Name of the CSV written to ``data/processed``.
    parse_item : callable(player_dict, stats_dict, season) -> dict
        Builds one flat record from a single player entry.
    add_metrics : callable(DataFrame) -> DataFrame
        Adds derived columns to the assembled frame.
    label : str
        Human-readable name used in the completion message.
    """
    records = []
    for season in SEASONS:
        data = load_json(RAW_DIR / raw_template.format(season=season))
        if data is None:
            continue
        for item in data["response"]:
            records.append(parse_item(item["player"], item["statistics"][0], season))

    df = add_metrics(pd.DataFrame(records))
    df.to_csv(PROCESSED_DIR / output_name, index=False)
    print(f"{label} complete. Rows: {len(df)}")
    return df


# --- Player leaderboards -----------------------------------------------------

def preprocess_player_yellow():
    """Build the player discipline table from the top-yellow-cards leaderboard."""
    def parse(player, stats, season):
        return {
            "season": season,
            "player_id": player["id"],
            "name": player["name"],
            "team": stats["team"]["name"],
            "position": stats["games"]["position"],
            "appearances": stats["games"]["appearences"] or 0,
            "minutes": stats["games"]["minutes"] or 0,
            "yellow_cards": stats["cards"]["yellow"] or 0,
            "red_cards": stats["cards"]["red"] or 0,
            "fouls_committed": stats["fouls"]["committed"] or 0,
            "tackles": stats["tackles"]["total"] or 0,
            "duels_won": stats["duels"]["won"] or 0,
        }

    def metrics(df):
        df["yellow_per_90"] = per_90(df["yellow_cards"], df["minutes"])
        df["fouls_per_90"] = per_90(df["fouls_committed"], df["minutes"])
        df["tackles_per_90"] = per_90(df["tackles"], df["minutes"])
        return df

    return build_player_stat_csv(
        "topyellowcards_39_{season}.json",
        "player_discipline_39_2020_2024.csv",
        parse, metrics, "Player discipline",
    )


def preprocess_topscorers():
    """Build the top-scorers table with per-90 attacking metrics."""
    def parse(player, stats, season):
        rating = stats["games"]["rating"]
        return {
            "season": season,
            "player_id": player["id"],
            "name": player["name"],
            "nationality": player["nationality"],
            "age": player["age"],
            "team": stats["team"]["name"],
            "position": stats["games"]["position"],
            "appearances": stats["games"]["appearences"] or 0,
            "minutes": stats["games"]["minutes"] or 0,
            "goals": stats["goals"]["total"] or 0,
            "assists": stats["goals"]["assists"] or 0,
            "shots": stats["shots"]["total"] or 0,
            "shots_on_target": stats["shots"]["on"] or 0,
            "rating": float(rating) if rating else None,
            "penalty_goals": stats["penalty"]["scored"] or 0,
            "yellow_cards": stats["cards"]["yellow"] or 0,
            "red_cards": stats["cards"]["red"] or 0,
        }

    def metrics(df):
        df["goals_per_90"] = per_90(df["goals"], df["minutes"])
        df["assists_per_90"] = per_90(df["assists"], df["minutes"])
        df["shots_per_90"] = per_90(df["shots"], df["minutes"])
        df["shot_accuracy"] = df["shots_on_target"] / df["shots"].replace(0, 1)
        return df

    return build_player_stat_csv(
        "topscorers_39_{season}.json",
        "player_topscorers_39_2020_2024.csv",
        parse, metrics, "Top scorers",
    )


def preprocess_topredcards():
    """Build the most-sent-off players table with per-90 discipline metrics."""
    def parse(player, stats, season):
        rating = stats["games"]["rating"]
        return {
            "season": season,
            "player_id": player["id"],
            "name": player["name"],
            "nationality": player["nationality"],
            "age": player["age"],
            "team": stats["team"]["name"],
            "position": stats["games"]["position"],
            "appearances": stats["games"]["appearences"] or 0,
            "minutes": stats["games"]["minutes"] or 0,
            "red_cards": stats["cards"]["red"] or 0,
            "yellow_cards": stats["cards"]["yellow"] or 0,
            "fouls_committed": stats["fouls"]["committed"] or 0,
            "tackles": stats["tackles"]["total"] or 0,
            "rating": float(rating) if rating else None,
        }

    def metrics(df):
        df["red_per_90"] = per_90(df["red_cards"], df["minutes"])
        df["yellow_per_90"] = per_90(df["yellow_cards"], df["minutes"])
        df["fouls_per_90"] = per_90(df["fouls_committed"], df["minutes"])
        df["tackles_per_90"] = per_90(df["tackles"], df["minutes"])
        return df

    return build_player_stat_csv(
        "topredcards_39_{season}.json",
        "player_topredcards_39_2020_2024.csv",
        parse, metrics, "Top red cards",
    )


def preprocess_topassists():
    """Build the top-assists table with per-90 creativity metrics."""
    def parse(player, stats, season):
        rating = stats["games"]["rating"]
        return {
            "season": season,
            "player_id": player["id"],
            "name": player["name"],
            "nationality": player["nationality"],
            "age": player["age"],
            "team": stats["team"]["name"],
            "position": stats["games"]["position"],
            "appearances": stats["games"]["appearences"] or 0,
            "minutes": stats["games"]["minutes"] or 0,
            "assists": stats["goals"]["assists"] or 0,
            "goals": stats["goals"]["total"] or 0,
            "key_passes": stats["passes"]["key"] or 0,
            "shots": stats["shots"]["total"] or 0,
            "rating": float(rating) if rating else None,
        }

    def metrics(df):
        df["assists_per_90"] = per_90(df["assists"], df["minutes"])
        df["goals_per_90"] = per_90(df["goals"], df["minutes"])
        df["key_passes_per_90"] = per_90(df["key_passes"], df["minutes"])
        df["shot_involvement_per_90"] = per_90(df["goals"] + df["assists"], df["minutes"])
        return df

    return build_player_stat_csv(
        "topassists_39_{season}.json",
        "player_topassists_39_2020_2024.csv",
        parse, metrics, "Top assists",
    )


# --- Teams -------------------------------------------------------------------

def preprocess_teams():
    """Build the teams reference table (identity + venue details)."""
    records = []
    for season in SEASONS:
        data = load_json(RAW_DIR / f"teams_39_{season}.json")
        if data is None:
            continue
        for item in data["response"]:
            team = item["team"]
            venue = item["venue"]
            records.append({
                "season": season,
                "team_id": team["id"],
                "team_name": team["name"],
                "team_code": team["code"],
                "country": team["country"],
                "founded": team["founded"],
                "national_team": team["national"],
                "venue_id": venue["id"],
                "venue_name": venue["name"],
                "venue_city": venue["city"],
                "venue_capacity": venue["capacity"],
                "venue_surface": venue["surface"],
            })

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "teams_39_2020_2024.csv", index=False)
    print("Teams complete. Rows:", len(df))


def preprocess_team_statistics():
    """Build the aggregated team-season statistics table."""
    records = []
    for season in SEASONS:
        data = load_json(RAW_DIR / f"team_statistics_39_{season}.json")
        if data is None:
            continue
        for _team_id, team_data in data.items():
            response = team_data["response"]
            team = response["team"]
            fixtures = response["fixtures"]
            goals = response["goals"]
            cards = response["cards"]
            records.append({
                "season": season,
                "team_id": team["id"],
                "team_name": team["name"],
                "matches_played": fixtures["played"]["total"],
                "wins": fixtures["wins"]["total"],
                "draws": fixtures["draws"]["total"],
                "losses": fixtures["loses"]["total"],
                "goals_for": goals["for"]["total"]["total"],
                "goals_against": goals["against"]["total"]["total"],
                "clean_sheets": response["clean_sheet"]["total"],
                "yellow_cards": sum_card_totals(cards["yellow"]),
                "red_cards": sum_card_totals(cards["red"]),
            })

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "team_stat_39_2020_2024.csv", index=False)
    print("Team statistics complete. Rows:", len(df))


def preprocess_standings():
    """Build the final league-table (standings) data for every season."""
    records = []
    for season in SEASONS:
        data = load_json(RAW_DIR / f"standings_39_{season}.json")
        if data is None:
            continue
        standings = data["response"][0]["league"]["standings"][0]
        for team_data in standings:
            team = team_data["team"]
            all_stats = team_data["all"]
            rank = team_data["rank"]
            records.append({
                "season": season,
                "team_id": team["id"],
                "team_name": team["name"],
                "rank": rank,
                "points": team_data["points"],
                "goal_difference": team_data["goalsDiff"],
                "matches_played": all_stats["played"],
                "wins": all_stats["win"],
                "draws": all_stats["draw"],
                "losses": all_stats["lose"],
                "goals_for": all_stats["goals"]["for"],
                "goals_against": all_stats["goals"]["against"],
                "top4": 1 if rank <= 4 else 0,
                "relegated": 1 if rank >= 18 else 0,
            })

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "standings_39_2020_2024.csv", index=False)
    print("Standings complete. Rows:", len(df))


# --- Players, injuries, fixtures ---------------------------------------------

def preprocess_player_profiles():
    """Build the player-profiles reference table (biographical details)."""
    data = load_json(RAW_DIR / "player_profiles_all.json")
    if data is None:
        return

    # The master file is a list of players; tolerate a wrapped {"response": [...]} too.
    players = data if isinstance(data, list) else data.get("response", [])

    records = []
    for item in players:
        player_data = item.get("player", {})
        birth = player_data.get("birth", {})
        records.append({
            "player_id": player_data.get("id"),
            "name": player_data.get("name"),
            "firstname": player_data.get("firstname"),
            "lastname": player_data.get("lastname"),
            "age": player_data.get("age"),
            "birth_date": birth.get("date"),
            "birth_place": birth.get("place"),
            "birth_country": birth.get("country"),
            "nationality": player_data.get("nationality"),
            "height": player_data.get("height"),
            "weight": player_data.get("weight"),
            "position": player_data.get("position"),
        })

    df = pd.DataFrame(records)
    if "height" in df.columns:
        df["height"] = df["height"].astype(str).str.replace(" cm", "", regex=False)
    if "weight" in df.columns:
        df["weight"] = df["weight"].astype(str).str.replace(" kg", "", regex=False)

    df.to_csv(PROCESSED_DIR / "player_profiles_all.csv", index=False)
    print("Player profiles complete. Rows:", len(df))


def preprocess_injuries():
    """Build the per-fixture injuries table across all seasons."""
    records = []
    for season in SEASONS:
        data = load_json(RAW_DIR / f"injuries_39_{season}.json")
        if data is None:
            continue
        for _fixture_id, fixture_data in data.items():
            for injury in fixture_data.get("response", []):
                player = injury.get("player", {})
                team = injury.get("team", {})
                fixture = injury.get("fixture", {})
                records.append({
                    "season": season,
                    "fixture_id": fixture.get("id"),
                    "fixture_date": fixture.get("date"),
                    "team_id": team.get("id"),
                    "team_name": team.get("name"),
                    "player_id": player.get("id"),
                    "player_name": player.get("name"),
                    "type": injury.get("type"),
                    "reason": injury.get("reason"),
                })

    if not records:
        print("Empty injuries dataset")
        return

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "injuries_39_2020_2024.csv", index=False)
    print(f"Injuries complete. Rows: {len(df)}")


def preprocess_fixtures():
    """Build the match-results table, including the home/draw/away target."""
    records = []
    for season in SEASONS:
        data = load_json(RAW_DIR / f"fixtures_39_{season}.json")
        if data is None:
            continue
        for match in data["response"]:
            fixture = match["fixture"]
            teams = match["teams"]
            goals = match["goals"]
            home_goals = goals["home"]
            away_goals = goals["away"]

            # Skip unfinished matches (no score yet).
            if home_goals is None or away_goals is None:
                continue

            if home_goals > away_goals:
                result = "H"
            elif home_goals < away_goals:
                result = "A"
            else:
                result = "D"

            records.append({
                "season": season,
                "fixture_id": fixture["id"],
                "date": fixture["date"],
                "home_team_id": teams["home"]["id"],
                "home_team_name": teams["home"]["name"],
                "away_team_id": teams["away"]["id"],
                "away_team_name": teams["away"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
                "total_goals": home_goals + away_goals,
                "result": result,
                "home_win": 1 if result == "H" else 0,
                "draw": 1 if result == "D" else 0,
                "away_win": 1 if result == "A" else 0,
                "over_2_5": 1 if (home_goals + away_goals) > 2.5 else 0,
            })

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "fixtures_39_2020_2024.csv", index=False)
    print("Fixtures complete. Total matches:", len(df))


def preprocess_fixture_statistics():
    """Build the per-fixture team statistics table (shots, possession, etc.)."""
    records = []
    for season in SEASONS:
        data = load_json(RAW_DIR / f"fixture_statistics_39_{season}.json")
        if data is None:
            continue
        for fixture_id, fixture_data in data.items():
            response = fixture_data.get("response", [])
            if len(response) != 2:
                continue

            home, away = response[0], response[1]
            home_stats = home.get("statistics", [])
            away_stats = away.get("statistics", [])

            records.append({
                "season": season,
                "fixture_id": int(fixture_id),
                "home_team": home["team"]["name"],
                "away_team": away["team"]["name"],

                # Shots
                "home_total_shots": extract_stat(home_stats, "Total Shots"),
                "away_total_shots": extract_stat(away_stats, "Total Shots"),
                "home_shots_on_target": extract_stat(home_stats, "Shots on Goal"),
                "away_shots_on_target": extract_stat(away_stats, "Shots on Goal"),

                # Possession
                "home_possession": clean_percentage(extract_stat(home_stats, "Ball Possession")),
                "away_possession": clean_percentage(extract_stat(away_stats, "Ball Possession")),

                # Passing
                "home_total_passes": extract_stat(home_stats, "Total passes"),
                "away_total_passes": extract_stat(away_stats, "Total passes"),
                "home_pass_accuracy": clean_percentage(extract_stat(home_stats, "Passes %")),
                "away_pass_accuracy": clean_percentage(extract_stat(away_stats, "Passes %")),

                # Discipline
                "home_fouls": extract_stat(home_stats, "Fouls"),
                "away_fouls": extract_stat(away_stats, "Fouls"),
                "home_yellow": extract_stat(home_stats, "Yellow Cards"),
                "away_yellow": extract_stat(away_stats, "Yellow Cards"),
                "home_red": extract_stat(home_stats, "Red Cards"),
                "away_red": extract_stat(away_stats, "Red Cards"),

                # Corners
                "home_corners": extract_stat(home_stats, "Corner Kicks"),
                "away_corners": extract_stat(away_stats, "Corner Kicks"),
            })

    if not records:
        print("No fixture-statistics records found")
        return

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "fixture_statistics_39_2020_2024.csv", index=False)
    print(f"Fixture statistics complete. Rows: {len(df)}")


def preprocess_headtohead():
    """Aggregate head-to-head records (Premier League, finished matches only)."""
    max_season = 2024
    data = load_json(RAW_DIR / "headtohead_39_2024.json")
    if data is None:
        return

    records = []
    for matchup_key, matchup_data in data.items():
        team1_id, team2_id = map(int, matchup_key.split("-"))

        team1_wins = team2_wins = draws = 0
        goals_team1 = goals_team2 = valid_matches = 0

        for match in matchup_data.get("response", []):
            league = match["league"]
            if league["id"] != LEAGUE_ID or league["season"] > max_season:
                continue
            if match["fixture"]["status"]["short"] != "FT":
                continue

            home_id = match["teams"]["home"]["id"]
            away_id = match["teams"]["away"]["id"]
            home_goals = match["goals"]["home"]
            away_goals = match["goals"]["away"]
            if home_goals is None or away_goals is None:
                continue

            valid_matches += 1

            # Orient goals/results relative to team1.
            if home_id == team1_id:
                goals_team1 += home_goals
                goals_team2 += away_goals
                t1_goals, t2_goals = home_goals, away_goals
            elif away_id == team1_id:
                goals_team1 += away_goals
                goals_team2 += home_goals
                t1_goals, t2_goals = away_goals, home_goals
            else:
                continue

            if t1_goals > t2_goals:
                team1_wins += 1
            elif t1_goals < t2_goals:
                team2_wins += 1
            else:
                draws += 1

        if valid_matches == 0:
            continue

        records.append({
            "team1_id": team1_id,
            "team2_id": team2_id,
            "h2h_matches": valid_matches,
            "team1_wins": team1_wins,
            "team2_wins": team2_wins,
            "draws": draws,
            "avg_goals_team1": goals_team1 / valid_matches,
            "avg_goals_team2": goals_team2 / valid_matches,
            "avg_total_goals": (goals_team1 + goals_team2) / valid_matches,
        })

    df = pd.DataFrame(records)
    df.to_csv(PROCESSED_DIR / "headtohead_fixtures_39_2020_2024.csv", index=False)
    print("Head-to-head complete. Total matchups:", len(df))


# --- Execution ---------------------------------------------------------------

def run_all():
    """Run every preprocessing step in order."""
    preprocess_player_yellow()
    preprocess_topscorers()
    preprocess_topredcards()
    preprocess_topassists()
    preprocess_teams()
    preprocess_team_statistics()
    preprocess_standings()
    preprocess_player_profiles()
    preprocess_injuries()
    preprocess_fixtures()
    preprocess_fixture_statistics()
    preprocess_headtohead()
    print("\nCOMPLETE.")


if __name__ == "__main__":
    run_all()
