"""Join the processed CSVs into a single supervised training table.

Starting from the fixtures table, this merges fixture statistics, previous- and
current-season standings, aggregated team statistics, head-to-head records and
injury counts, derives a set of difference features, encodes the match result
as the target, and writes ``model_training_data.csv`` plus the saved feature
list and target mapping.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Anchor paths to the project root so the script runs from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_MODEL_DATA = PROCESSED / "model_training_data.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

FILES = {
    "fixtures": PROCESSED / "fixtures_39_2020_2024.csv",
    "fixture_stats": PROCESSED / "fixture_statistics_39_2020_2024.csv",
    "team_stats": PROCESSED / "team_stat_39_2020_2024.csv",
    "standings": PROCESSED / "standings_39_2020_2024.csv",
    "headtohead": PROCESSED / "headtohead_fixtures_39_2020_2024.csv",
    "injuries": PROCESSED / "injuries_39_2020_2024.csv",
}


def safe_load(path):
    """Load a CSV into a DataFrame, returning an empty frame if missing/unreadable."""
    if not path.exists():
        print(f"Missing: {path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"Failed to load {path}: {e}")
        return pd.DataFrame()


def safe_col(df, col):
    """Return ``df[col]`` if present, otherwise 0 (so diffs degrade gracefully)."""
    return df[col] if col in df.columns else 0


def get_h2h_features(row, h2h):
    """Return the head-to-head features for one fixture as a Series.

    Looks up the home/away pairing in the ``h2h`` table in both orderings
    (swapping the win columns when the stored pairing is reversed) and falls
    back to zeros when no record exists.
    """
    empty = pd.Series({
        "h2h_matches": 0, "h2h_team1_wins": 0, "h2h_team2_wins": 0,
        "h2h_draws": 0, "h2h_avg_total_goals": 0,
    })
    if h2h.empty:
        return empty

    home, away = int(row["home_team_id"]), int(row["away_team_id"])

    rec = h2h[(h2h["team1_id"] == home) & (h2h["team2_id"] == away)]
    if not rec.empty:
        r = rec.iloc[0]
        return pd.Series({
            "h2h_matches": r.get("h2h_matches", 0),
            "h2h_team1_wins": r.get("team1_wins", 0),
            "h2h_team2_wins": r.get("team2_wins", 0),
            "h2h_draws": r.get("draws", 0),
            "h2h_avg_total_goals": r.get("avg_total_goals", 0),
        })

    # Stored in reverse order: swap team1/team2 win columns.
    rec = h2h[(h2h["team1_id"] == away) & (h2h["team2_id"] == home)]
    if not rec.empty:
        r = rec.iloc[0]
        return pd.Series({
            "h2h_matches": r.get("h2h_matches", 0),
            "h2h_team1_wins": r.get("team2_wins", 0),
            "h2h_team2_wins": r.get("team1_wins", 0),
            "h2h_draws": r.get("draws", 0),
            "h2h_avg_total_goals": r.get("avg_total_goals", 0),
        })

    return empty


def _merge_team_season_stats(fixtures, team_stats):
    """Merge aggregated team-season stats onto both the home and away sides."""
    ts = team_stats.copy()
    ts["team_id"] = ts["team_id"].astype(int)
    ts_cols = ["season", "team_id", "matches_played", "wins", "draws", "losses",
               "goals_for", "goals_against", "yellow_cards", "red_cards"]
    ts = ts[[c for c in ts_cols if c in ts.columns]]

    rename_for = lambda side: {
        "team_id": f"{side}_team_id",
        "matches_played": f"{side}_matches_played",
        "wins": f"{side}_wins",
        "draws": f"{side}_draws",
        "losses": f"{side}_losses",
        "goals_for": f"{side}_goals_for",
        "goals_against": f"{side}_goals_against",
        "yellow_cards": f"{side}_yellow_cards",
        "red_cards": f"{side}_red_cards",
    }

    fixtures = fixtures.merge(ts.rename(columns=rename_for("home")),
                              on=["season", "home_team_id"], how="left")
    fixtures = fixtures.merge(ts.rename(columns=rename_for("away")),
                              on=["season", "away_team_id"], how="left")
    return fixtures


def f_engineer():
    """Build and save the full supervised training dataset."""
    fixtures = safe_load(FILES["fixtures"])
    fixture_stats = safe_load(FILES["fixture_stats"])
    team_stats = safe_load(FILES["team_stats"])
    standings = safe_load(FILES["standings"])
    h2h = safe_load(FILES["headtohead"])
    injuries = safe_load(FILES["injuries"])

    if fixtures.empty:
        raise SystemExit("fixtures file missing")

    fixtures = fixtures.copy()
    fixtures["fixture_id"] = fixtures["fixture_id"].astype(int)

    # Merge per-fixture statistics.
    if not fixture_stats.empty:
        fixture_stats["fixture_id"] = fixture_stats["fixture_id"].astype(int)
        fixtures = fixtures.merge(fixture_stats, on="fixture_id", how="left", suffixes=("", "_fs"))

    if not h2h.empty:
        h2h["team1_id"] = h2h["team1_id"].astype(int)
        h2h["team2_id"] = h2h["team2_id"].astype(int)

    # Previous-season standings (rank + points) for each side.
    fixtures["prev_season"] = fixtures["season"] - 1
    if not standings.empty:
        standings_subset = standings[["season", "team_id", "rank", "points"]].copy()
        fixtures = fixtures.merge(
            standings_subset.rename(columns={"season": "prev_season", "team_id": "home_team_id",
                                             "rank": "home_prev_rank", "points": "home_prev_points"}),
            on=["prev_season", "home_team_id"], how="left",
        )
        fixtures = fixtures.merge(
            standings_subset.rename(columns={"season": "prev_season", "team_id": "away_team_id",
                                             "rank": "away_prev_rank", "points": "away_prev_points"}),
            on=["prev_season", "away_team_id"], how="left",
        )

    # Aggregated team-season statistics.
    if not team_stats.empty:
        fixtures = _merge_team_season_stats(fixtures, team_stats)

    # Head-to-head features per fixture.
    h2h_feats = fixtures.apply(lambda r: get_h2h_features(r, h2h), axis=1)
    fixtures = pd.concat([fixtures.reset_index(drop=True), h2h_feats.reset_index(drop=True)], axis=1)

    # Injury counts per team-season.
    if not injuries.empty:
        inj_agg = injuries.groupby(["season", "team_id"]).size().reset_index(name="injury_count")
        fixtures = fixtures.merge(
            inj_agg.rename(columns={"team_id": "home_team_id", "injury_count": "home_injury_count"}),
            on=["season", "home_team_id"], how="left",
        )
        fixtures = fixtures.merge(
            inj_agg.rename(columns={"team_id": "away_team_id", "injury_count": "away_injury_count"}),
            on=["season", "away_team_id"], how="left",
        )
    else:
        fixtures["home_injury_count"] = 0
        fixtures["away_injury_count"] = 0

    # Current-season standings (rank + points).
    if not standings.empty:
        st_df = standings[["season", "team_id", "rank", "points"]].copy()
        fixtures = fixtures.merge(
            st_df.rename(columns={"team_id": "home_team_id", "rank": "home_rank", "points": "home_points"}),
            on=["season", "home_team_id"], how="left",
        )
        fixtures = fixtures.merge(
            st_df.rename(columns={"team_id": "away_team_id", "rank": "away_rank", "points": "away_points"}),
            on=["season", "away_team_id"], how="left",
        )
    else:
        fixtures["home_rank"] = np.nan
        fixtures["away_rank"] = np.nan
        fixtures["home_points"] = np.nan
        fixtures["away_points"] = np.nan

    # Derived difference features.
    fixtures["possession_diff"] = safe_col(fixtures, "home_possession") - safe_col(fixtures, "away_possession")
    fixtures["shots_diff"] = safe_col(fixtures, "home_total_shots") - safe_col(fixtures, "away_total_shots")
    fixtures["shots_on_target_diff"] = safe_col(fixtures, "home_shots_on_target") - safe_col(fixtures, "away_shots_on_target")
    fixtures["goals_for_diff"] = safe_col(fixtures, "home_goals_for") - safe_col(fixtures, "away_goals_for")
    fixtures["goals_against_diff"] = safe_col(fixtures, "home_goals_against") - safe_col(fixtures, "away_goals_against")

    fixtures["prev_rank_diff"] = fixtures["home_prev_rank"].fillna(20) - fixtures["away_prev_rank"].fillna(20)
    fixtures["prev_points_diff"] = fixtures["home_prev_points"].fillna(0) - fixtures["away_prev_points"].fillna(0)
    fixtures["injury_diff"] = fixtures["home_injury_count"].fillna(0) - fixtures["away_injury_count"].fillna(0)

    fixtures["h2h_win_rate_home"] = fixtures.apply(
        lambda r: (r["h2h_team1_wins"] / r["h2h_matches"]) if r["h2h_matches"] > 0 else 0, axis=1)
    fixtures["h2h_win_rate_away"] = fixtures.apply(
        lambda r: (r["h2h_team2_wins"] / r["h2h_matches"]) if r["h2h_matches"] > 0 else 0, axis=1)

    # Encode the target: H -> 0, D -> 1, A -> 2.
    if "result" not in fixtures.columns:
        raise SystemExit("No 'result' column in fixtures; cannot create supervised dataset.")
    target_map = {"H": 0, "D": 1, "A": 2}
    fixtures["target"] = fixtures["result"].map(target_map)

    chosen_features = [
        # Fixture-level stats
        "home_total_shots", "away_total_shots", "home_shots_on_target", "away_shots_on_target",
        "home_possession", "away_possession",
        "home_total_passes", "away_total_passes", "home_pass_accuracy", "away_pass_accuracy",
        "home_fouls", "away_fouls", "home_yellow", "away_yellow", "home_red", "away_red",
        "home_corners", "away_corners",
        # Team season stats
        "home_matches_played", "home_wins", "home_draws", "home_losses", "home_goals_for", "home_goals_against",
        "away_matches_played", "away_wins", "away_draws", "away_losses", "away_goals_for", "away_goals_against",
        # Previous-season info
        "home_prev_rank", "away_prev_rank", "prev_rank_diff", "home_prev_points", "away_prev_points", "prev_points_diff",
        # Head-to-head & derived
        "h2h_matches", "h2h_team1_wins", "h2h_team2_wins", "h2h_draws", "h2h_avg_total_goals",
        "h2h_win_rate_home", "h2h_win_rate_away",
        # Injuries & current standings
        "home_injury_count", "away_injury_count", "injury_diff",
        "home_rank", "away_rank", "home_points", "away_points",
        # Diffs and extras
        "possession_diff", "shots_diff", "shots_on_target_diff", "goals_for_diff", "goals_against_diff",
    ]

    # Ensure every chosen feature exists (fill missing ones with 0).
    final_features = []
    for f in chosen_features:
        if f not in fixtures.columns:
            fixtures[f] = 0
        final_features.append(f)

    model_df = fixtures.dropna(subset=["target"]).copy()
    X = model_df[final_features].astype(float).fillna(0)
    y = model_df["target"].astype(int)

    if "season" not in model_df.columns:
        raise SystemExit("ERROR: season column missing before save.")

    meta_cols = ["fixture_id", "date", "season", "home_team_name", "away_team_name", "result"]
    save_df = pd.concat(
        [model_df[meta_cols].reset_index(drop=True), X.reset_index(drop=True), y.reset_index(drop=True)],
        axis=1,
    )

    OUT_MODEL_DATA.parent.mkdir(parents=True, exist_ok=True)
    save_df.to_csv(OUT_MODEL_DATA, index=False)
    joblib.dump(final_features, MODELS_DIR / "features_list.joblib")
    with open(MODELS_DIR / "target_mapping.json", "w") as fh:
        json.dump(target_map, fh)

    print(f"training data: {OUT_MODEL_DATA} ({len(save_df)} rows)")
    print(f"feature list to {MODELS_DIR / 'features_list.joblib'}")
    print(f"target mapping to {MODELS_DIR / 'target_mapping.json'}")

    return OUT_MODEL_DATA, final_features, (save_df, X, y)


if __name__ == "__main__":
    f_engineer()
