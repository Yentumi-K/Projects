# Premier League Data Analytics (2020/21 – 2024/25)

An end-to-end football analytics project covering data cleaning, feature
engineering, match-outcome modelling and an interactive Streamlit dashboard.
Built on five seasons (2020/21 – 2024/25) of English Premier League data from the
[API-Football](https://www.api-football.com/) service.

## What it does

- **Cleans** the collected raw JSON into tidy, analysis-ready CSV tables.
- **Engineers** a supervised dataset by joining fixtures with team form,
  previous-season standings, head-to-head records and injury counts, plus a
  set of difference features.
- **Trains** Random Forest classifiers to predict match outcomes
  (Home / Draw / Away), with a backtest (train 2020–2023, test 2024 — ~64%
  accuracy) and a future-prediction model.
- **Visualises** everything in a four-page Streamlit dashboard: league
  overview, match prediction, player leaderboards and an injury/discipline
  monitor.

> The raw data was collected via the API-Football scripts (`fetch_data.py`,
> `fetch_player_profiles.py`), which are included for reference. **The data has
> already been fetched**, so you don't need an API key — just download it from
> the link in [Get the data](#get-the-data) and run the pipeline from
> preprocessing onward.

## Tech stack

Python · pandas · NumPy · scikit-learn · Streamlit · Plotly · Matplotlib · joblib

## Project structure

```
premier-league-analytics/
├── README.md
├── requirements.txt
├── .gitignore
├── Screenshots/                 # dashboard screenshots
└── src/
    ├── api_client.py            # Shared API-Football client (used only for fetching)
    ├── fetch_data.py            # [optional] Re-download fixtures/stats/standings/H2H
    ├── fetch_player_profiles.py # [optional] Re-download paginated player profiles
    ├── main_preprocess.py       # Raw JSON -> processed CSVs
    ├── feature_engineering.py   # Processed CSVs -> single training table
    ├── model_train.py           # Train backtest / future Random Forest models
    └── Pl_dashboard.py          # Streamlit dashboard
```

After you download the data and run the pipeline, two more folders appear:

```
├── data/                        # downloaded — raw JSON + processed CSVs
└── models/                      # created automatically when you run the pipeline
```

## Setup

```bash
git clone <your-repo-url>
cd premier-league-analytics

python -m venv venv

# macOS / Linux:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

> **Note (managed Windows machines):** if importing scikit-learn fails with a
> DLL / "Application Control policy" error, install via Anaconda instead:
> ```
> conda create -n pl_env python=3.11 pandas numpy scikit-learn matplotlib joblib -y
> conda activate pl_env
> pip install streamlit plotly
> ```

## Get the data

The data has already been collected. **Download it here:**

https://drive.google.com/drive/folders/1FeCmTFP_Urq9AVKJQ64XtuCXIWrICe_H?usp=sharing

Extract the archive into the project root so you end up with a `data/` folder
containing `raw/` and `processed/`, i.e. `data/raw/fixtures_39_2020.json`,
`data/processed/...`, etc.

No API key is needed for any step below — the fetch scripts are only there in
case you ever want to re-collect the data with your own key.

## How to run

Run all commands **from the project root**, after the `data/` folder is in place.

```bash
# 1. Clean the raw JSON into processed CSVs
python src/main_preprocess.py

# 2. Build the single training table
python src/feature_engineering.py

# 3. Train the models (run twice: choose 1 = backtest, then 2 = future)
python src/model_train.py

# 4. Launch the dashboard
streamlit run src/Pl_dashboard.py
```

The `models/` folder is created automatically during steps 2–3. The dashboard
opens in your browser at `http://localhost:8501`.

## Modelling notes

- Target encoding: Home win = 0, Draw = 1, Away win = 2.
- Leakage columns (final scores, goal difference, identifiers, metadata) are
  excluded from the feature set in `model_train.py`.
- Models are class-balanced Random Forests (`n_estimators=400`) inside a
  `StandardScaler` pipeline; the feature order is persisted alongside the model
  so the dashboard scores fixtures with the same columns.

