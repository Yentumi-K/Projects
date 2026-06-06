"""Streamlit dashboard for Premier League analytics (2020/21 - 2024/25).

Four pages share a season selector in the sidebar:

* **League Dashboard** - season overview, standings and summary charts.
* **Match Prediction** - backtest results and a future-fixture predictor.
* **Player Dashboard** - scorer/assist leaderboards with per-90 trend charts.
* **Injury & Discipline Monitor** - card leaders and an injury calendar.

Run from the project root: ``streamlit run src/Pl_dashboard.py``.
"""

import calendar
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from sklearn.metrics import accuracy_score, confusion_matrix

# Suppress library deprecation/usage warnings so they don't clutter the UI.
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Football Analytics", layout="wide")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

BACKTEST_MODEL_FILE = MODEL_DIR / "rf_backtest_model.joblib"
FUTURE_MODEL_FILE = MODEL_DIR / "rf_future_model.joblib"
FEATURE_FILE = MODEL_DIR / "feature_order.joblib"
TRAIN_DATA = DATA_DIR / "model_training_data.csv"


# --- Data and model loading --------------------------------------------------

@st.cache_data
def load_data():
    """Load every processed CSV the dashboard needs, keyed by short name."""
    return {
        "fixtures": pd.read_csv(DATA_DIR / "fixtures_39_2020_2024.csv"),
        "standings": pd.read_csv(DATA_DIR / "standings_39_2020_2024.csv"),
        "topscorers": pd.read_csv(DATA_DIR / "player_topscorers_39_2020_2024.csv"),
        "topassists": pd.read_csv(DATA_DIR / "player_topassists_39_2020_2024.csv"),
        "topyellow": pd.read_csv(DATA_DIR / "player_discipline_39_2020_2024.csv"),
        "topred": pd.read_csv(DATA_DIR / "player_topredcards_39_2020_2024.csv"),
        "profiles": pd.read_csv(DATA_DIR / "player_profiles_all.csv"),
        "injuries": pd.read_csv(DATA_DIR / "injuries_39_2020_2024.csv"),
    }


@st.cache_resource
def load_models():
    """Load the backtest model, future model and feature order (None if absent)."""
    backtest_model = joblib.load(BACKTEST_MODEL_FILE) if BACKTEST_MODEL_FILE.exists() else None
    future_model = joblib.load(FUTURE_MODEL_FILE) if FUTURE_MODEL_FILE.exists() else None
    features = joblib.load(FEATURE_FILE) if FEATURE_FILE.exists() else None
    return backtest_model, future_model, features


@st.cache_data
def load_training_data():
    """Load the engineered training table, or an empty frame if not yet built."""
    if TRAIN_DATA.exists():
        return pd.read_csv(TRAIN_DATA)
    return pd.DataFrame()


data = load_data()
fixtures = data["fixtures"]
standings = data["standings"]
topscorers = data["topscorers"]
topassists = data["topassists"]
topyellow = data["topyellow"]
topred = data["topred"]
profiles = data["profiles"]
injuries = data["injuries"]

injuries["fixture_date"] = pd.to_datetime(injuries["fixture_date"], errors="coerce")

SEASONS = sorted(fixtures["season"].unique(), reverse=True)
ALL_SEASONS = sorted(fixtures["season"].unique())
YEARS = list(range(2021, 2025))

backtest_model, future_model, features = load_models()
train_df = load_training_data()


# --- Shared table helpers ----------------------------------------------------

def simple_table(df):
    """Render a DataFrame as a plain, full-width Streamlit table."""
    st.dataframe(df, use_container_width=True, hide_index=True)


def get_profile_tooltip(player_id):
    """Return a single-line summary of a player's profile for use as a tooltip."""
    prof = profiles[profiles["player_id"] == player_id]
    if not prof.empty:
        row = prof.iloc[0].drop("player_id", errors="ignore")
        return " | ".join(
            f"{str(k).replace('_', ' ').title()}: {v}"
            for k, v in row.items() if pd.notna(v)
        )
    return "Profile not available"


def tooltip_table(df, name_col="name", id_col="player_id"):
    """Render a DataFrame as an HTML table with profile tooltips on player names."""
    html = """
    <style>
    table.custom-table { width:100%; border-collapse:collapse;
        font-family:Arial, sans-serif; font-size:14px; }
    table.custom-table th { text-align:left; padding:8px; border-bottom:2px solid #ddd; }
    table.custom-table td { padding:8px; border-bottom:1px solid #eee; }
    table.custom-table tr:hover { background-color:rgba(128,128,128,0.08); }
    </style>
    <table class="custom-table"><thead><tr>
    """

    for col in df.columns:
        if col != id_col:
            html += f"<th>{col.replace('_', ' ').title()}</th>"
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            if col == id_col:
                continue
            if col == name_col:
                tooltip = get_profile_tooltip(row[id_col])
                html += (f'<td><span title="{tooltip}" style="cursor:help;">'
                         f'{row[col]}</span></td>')
            else:
                html += f"<td>{row[col]}</td>"
        html += "</tr>"

    html += "</tbody></table>"
    st.markdown(html, unsafe_allow_html=True)


def render_per90_section(heading, leaderboard, source_df, metric_col,
                         metric_label, picker_label, radio_label, key):
    """Render a leaderboard plus an optional per-90 trend chart.

    Shows the top-15 leaderboard with tooltips, lets the user pick up to three
    players, and draws their ``metric_col`` (e.g. goals/assists per 90) over all
    seasons either as separate charts or one combined comparison chart. Used by
    both the scorers and assists panels, which differ only in the metric.
    """
    st.markdown(heading)
    tooltip_table(leaderboard.head(15))

    names = leaderboard["name"].unique()
    selected = st.multiselect(picker_label, names, max_selections=3, key=f"{key}_pick")
    if not selected:
        return

    graph_type = st.radio(radio_label, ["Separate Graphs", "Combined Graph"], key=f"{key}_type")

    if graph_type == "Separate Graphs":
        for player in selected:
            pid = leaderboard[leaderboard["name"] == player]["player_id"].iloc[0]
            stats = source_df[source_df["player_id"] == pid].sort_values("season")
            seasons = stats["season"].astype(int)

            fig, ax = plt.subplots(figsize=(5, 2.5))
            ax.plot(seasons, stats[metric_col], marker="o")
            ax.set_title(f"{player} {metric_label} per 90")
            ax.set_xticks(ALL_SEASONS)
            ax.set_xlim(min(ALL_SEASONS) - 0.5, max(ALL_SEASONS) + 0.5)
            ax.set_xlabel("Season")
            st.pyplot(fig)
    else:  # Combined Graph
        fig, ax = plt.subplots(figsize=(6, 3))
        for player in selected:
            pid = leaderboard[leaderboard["name"] == player]["player_id"].iloc[0]
            stats = source_df[source_df["player_id"] == pid]
            y = []
            for s in ALL_SEASONS:
                val = stats[stats["season"] == s][metric_col]
                y.append(val.iloc[0] if not val.empty else None)
            ax.plot(ALL_SEASONS, y, marker="o", label=player)

        ax.set_title(f"{metric_label} per 90 Comparison")
        ax.set_xticks(ALL_SEASONS)
        ax.set_xlim(min(ALL_SEASONS) - 0.5, max(ALL_SEASONS) + 0.5)
        ax.set_xlabel("Season")
        ax.legend()
        st.pyplot(fig)


# --- Pages -------------------------------------------------------------------

def render_league_dashboard(season_selected):
    """Season overview: headline metrics, standings and summary charts."""
    st.title("League Dashboard Overview")

    season_df = fixtures[fixtures["season"] == season_selected]

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(season_df))
    col2.metric("Avg Goals / Match", round(season_df["total_goals"].mean(), 2))
    col3.metric("Home Win %",
                round((season_df["home_goals"] > season_df["away_goals"]).mean() * 100, 1))

    st.markdown("---")

    st.subheader("League Standings")
    table = standings[standings["season"] == season_selected].sort_values("rank")
    table = table.drop(columns=["season", "team_id", "top4", "relegated"], errors="ignore")
    table = table[[
        "rank", "team_name", "points", "goal_difference",
        "matches_played", "wins", "draws", "losses",
        "goals_for", "goals_against",
    ]]
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Match Outcomes")
        home_wins = len(season_df[season_df["home_goals"] > season_df["away_goals"]])
        draws = len(season_df[season_df["home_goals"] == season_df["away_goals"]])
        away_wins = len(season_df[season_df["home_goals"] < season_df["away_goals"]])

        fig_outcomes = px.pie(
            names=["Home Wins", "Draws", "Away Wins"],
            values=[home_wins, draws, away_wins],
            hole=0.4,
            color=["Home Wins", "Draws", "Away Wins"],
            color_discrete_map={"Home Wins": "green", "Draws": "yellow", "Away Wins": "red"},
        )
        fig_outcomes.update_layout(margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig_outcomes, use_container_width=True)

    with col_b:
        st.subheader("Goals Trend (5-Match Rolling Avg)")
        rolling = season_df["total_goals"].rolling(5).mean().reset_index(drop=True)
        fig_trend = px.line(
            x=rolling.index + 1, y=rolling,
            labels={"x": "Match Number", "y": "Goals (Avg)"},
        )
        fig_trend.update_traces(line_color="blue", line_width=3)
        fig_trend.update_layout(margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")

    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Attack vs. Defense")
        fig_scatter = px.scatter(
            table, x="goals_for", y="goals_against",
            hover_name="team_name",
            hover_data={"goals_for": True, "goals_against": True},
            labels={"goals_for": "Goals For (Attack)", "goals_against": "Goals Against (Defense)"},
        )
        # Lower goals-against (better defense) at the top.
        fig_scatter.update_yaxes(autorange="reversed")
        fig_scatter.add_vline(x=table["goals_for"].mean(), line_dash="dash", line_color="gray", opacity=0.7)
        fig_scatter.add_hline(y=table["goals_against"].mean(), line_dash="dash", line_color="gray", opacity=0.7)
        fig_scatter.update_traces(marker=dict(size=12, color="royalblue", opacity=0.8,
                                              line=dict(width=1, color="DarkSlateGrey")))
        fig_scatter.update_layout(margin=dict(t=20, b=20, l=0, r=0))
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col_d:
        st.subheader("Goal Difference by Team")
        gd_table = table.sort_values("goal_difference", ascending=True)
        gd_table["Color"] = gd_table["goal_difference"].apply(lambda x: "red" if x < 0 else "green")
        fig_gd = px.bar(
            gd_table, x="goal_difference", y="team_name", orientation="h",
            hover_name="team_name",
            labels={"goal_difference": "Goal Difference", "team_name": ""},
        )
        fig_gd.update_traces(marker_color=gd_table["Color"])
        fig_gd.update_layout(margin=dict(t=20, b=20, l=0, r=0), showlegend=False)
        st.plotly_chart(fig_gd, use_container_width=True)


def render_match_prediction(season_selected):
    """Backtest evaluation (2024) and an interactive future-fixture predictor."""
    st.title("Match Outcome Prediction")
    st.info("Season filter does not affect predictions.")

    if backtest_model is None or future_model is None:
        st.error("Models not trained yet. Run model_train.py first.")
        st.stop()
    if train_df.empty:
        st.error("Training dataset missing.")
        st.stop()

    tab1, tab2 = st.tabs(["Backtest (2024 Season)", "Future Prediction (2025/26)"])

    with tab1:
        st.subheader("Backtest: Train 2020-2023 -> Predict 2024")
        df_2024 = train_df[train_df["season"] == 2024]

        if df_2024.empty:
            st.warning("No 2024 data available.")
        else:
            X = df_2024[features].astype(float).fillna(0)
            y_true = df_2024["target"]
            preds = backtest_model.predict(X)
            probs = backtest_model.predict_proba(X)

            st.metric("Backtest Accuracy (2024)", f"{accuracy_score(y_true, preds) * 100:.2f}%")

            inv_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
            results = df_2024[["home_team_name", "away_team_name", "result"]].copy()
            results["Prediction"] = [inv_map[p] for p in preds]
            results["Prob Home"] = probs[:, 0]
            results["Prob Draw"] = probs[:, 1]
            results["Prob Away"] = probs[:, 2]
            st.dataframe(results.head(50), use_container_width=True, hide_index=True)

            st.markdown("Prediction Distribution")
            pred_counts = results["Prediction"].value_counts()
            fig = px.bar(pred_counts, labels={"value": "Matches", "index": "Outcome"},
                         title="Predicted Match Outcomes")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("Confusion Matrix")
            cm = confusion_matrix(y_true, preds)
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Home", "Actual Draw", "Actual Away"],
                columns=["Pred Home", "Pred Draw", "Pred Away"],
            )
            fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues")
            st.plotly_chart(fig_cm, use_container_width=True)

    with tab2:
        st.subheader("Predict 2025/26 Season")
        teams = sorted(train_df["home_team_name"].unique())

        col1, col2 = st.columns(2)
        home_team = col1.selectbox("Home Team", teams)
        away_team = col2.selectbox("Away Team", teams)

        if home_team == away_team:
            st.warning("Select two different teams.")
        elif st.button("Predict Match Outcome"):
            hist = train_df[
                (train_df["home_team_name"] == home_team)
                & (train_df["away_team_name"] == away_team)
            ]
            if hist.empty:
                st.info("No historical matchup found. Using league average.")
                X_input = np.zeros((1, len(features)))
            else:
                X_input = hist[features].iloc[-1:].values

            probs = future_model.predict_proba(X_input)[0]
            pred_idx = np.argmax(probs)
            outcomes = ["Home Win", "Draw", "Away Win"]
            st.success(f"Predicted Result: **{outcomes[pred_idx]}**")

            col1, col2, col3 = st.columns(3)
            col1.metric("Home Win", f"{probs[0] * 100:.1f}%")
            col2.metric("Draw", f"{probs[1] * 100:.1f}%")
            col3.metric("Away Win", f"{probs[2] * 100:.1f}%")

            fig = px.bar(x=["Home", "Draw", "Away"], y=probs,
                         labels={"x": "Outcome", "y": "Probability"},
                         title="Prediction Probabilities")
            st.plotly_chart(fig, use_container_width=True)


def render_player_dashboard(season_selected):
    """Top scorers and assists for the season, each with per-90 trend charts."""
    st.title("👤 Player Dashboard")
    left, right = st.columns(2)

    with left:
        scorers = topscorers[topscorers["season"] == season_selected] \
            .sort_values("goals", ascending=False)[["player_id", "name", "goals"]]
        render_per90_section(
            "### Top Scorers Per Season", scorers, topscorers,
            "goals_per_90", "Goals", "Pick Scorers (max 3)", "Scorer Graph Type", "scorers",
        )

    with right:
        assists = topassists[topassists["season"] == season_selected] \
            .sort_values("assists", ascending=False)[["player_id", "name", "assists"]]
        render_per90_section(
            "### Top Assists Per Season", assists, topassists,
            "assists_per_90", "Assists", "Pick Assisters (max 3)", "Assist Graph Type", "assists",
        )


def render_injury_monitor(season_selected):
    """Card leaders for the season and a filterable monthly injury calendar."""
    st.title("🚑 Injury & Discipline Monitor")

    st.subheader("Top Yellow Cards")
    yellows = topyellow[topyellow["season"] == season_selected] \
        .sort_values("yellow_cards", ascending=False)[["player_id", "name", "yellow_cards"]]
    tooltip_table(yellows.head(10))

    st.subheader("Top Red Cards")
    reds = topred[topred["season"] == season_selected] \
        .sort_values("red_cards", ascending=False)[["player_id", "name", "red_cards"]]
    tooltip_table(reds.head(10))

    st.subheader("Injury Calendar")
    teams = sorted(injuries["team_name"].dropna().unique())

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_team = st.selectbox("Filter by Team", ["All Teams"] + teams)

    if selected_team != "All Teams":
        team_players = injuries[injuries["team_name"] == selected_team]["player_name"].dropna().unique()
    else:
        team_players = injuries["player_name"].dropna().unique()

    with filter_col2:
        selected_players = st.multiselect("Filter by Player", sorted(team_players))

    year = st.selectbox("Select Year", YEARS)
    month_name = st.selectbox("Select Month", list(calendar.month_name)[1:])
    month = list(calendar.month_name).index(month_name)
    cal = calendar.monthcalendar(year, month)

    headers = st.columns(7)
    for i, day in enumerate(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]):
        headers[i].markdown(f"**{day}**")

    month_inj = injuries[
        (injuries["fixture_date"].dt.year == year)
        & (injuries["fixture_date"].dt.month == month)
    ]
    if selected_team != "All Teams":
        month_inj = month_inj[month_inj["team_name"] == selected_team]
    if selected_players:
        month_inj = month_inj[month_inj["player_name"].isin(selected_players)]

    for week in cal:
        cols = st.columns(7)
        for i, day in enumerate(week):
            if day == 0:
                cols[i].write("")
                continue

            current_date = pd.Timestamp(year=year, month=month, day=day)
            day_inj = month_inj[month_inj["fixture_date"].dt.date == current_date.date()]

            if day_inj.empty:
                cols[i].markdown(str(day))
                continue

            text = f"**{day}**<br>"
            for _, row in day_inj.iterrows():
                pid = row.get("player_id", None)
                if pd.notna(pid):
                    prof_tooltip = get_profile_tooltip(pid)
                    text += (f'🔴 <span title="{prof_tooltip}" '
                             f'style="cursor:help; border-bottom:1px dotted #888;">'
                             f'{row["player_name"]}</span> ({row["team_name"]})<br>')
                else:
                    text += f"🔴 {row['player_name']} ({row['team_name']})<br>"
            cols[i].markdown(text, unsafe_allow_html=True)


# --- Sidebar navigation and dispatch -----------------------------------------

PAGES = {
    "League Dashboard": render_league_dashboard,
    "Match Prediction": render_match_prediction,
    "Player Dashboard": render_player_dashboard,
    "Injury & Discipline Monitor": render_injury_monitor,
}

st.sidebar.title("Global Controls")
season_selected = st.sidebar.selectbox("Season", SEASONS)
page = st.sidebar.radio("Navigate", list(PAGES.keys()))

PAGES[page](season_selected)
