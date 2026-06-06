"""Train Random Forest match-outcome models from the engineered dataset.

Two modes are provided:

* **Backtest** - train on seasons 2020-2023 and evaluate on 2024, printing
  accuracy, a classification report and a confusion matrix.
* **Future** - train on all seasons (2020-2024) to predict upcoming fixtures.

Both pipelines standardise features and fit a class-balanced Random Forest, and
save the trained pipeline plus the feature order used at training time.
"""

import joblib
from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data" / "processed"
TRAIN_CSV = PROCESSED / "model_training_data.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

BACKTEST_MODEL = MODELS_DIR / "rf_backtest_model.joblib"
FUTURE_MODEL = MODELS_DIR / "rf_future_model.joblib"
FEATURE_FILE = MODELS_DIR / "feature_order.joblib"

# Columns that must never be used as features: identifiers, metadata, the target
# itself, and any post-match outcome that would leak the result.
LEAKAGE_COLUMNS = {
    "fixture_id", "date", "home_team_name", "away_team_name", "season",
    "result", "target", "home_goals", "away_goals", "total_goals", "goal_difference",
}


def load_training():
    """Load the engineered dataset and return ``(df, feature_columns)``.

    Drops leakage columns to derive the feature list, fills missing values,
    and persists the feature order for the dashboard to reuse.
    """
    if not TRAIN_CSV.exists():
        raise SystemExit(f"{TRAIN_CSV} missing - run feature_engineering.py first.")

    df = pd.read_csv(TRAIN_CSV)
    if "season" not in df.columns:
        raise SystemExit("Training CSV must contain a 'season' column.")

    features = [c for c in df.columns if c not in LEAKAGE_COLUMNS]
    df[features] = df[features].astype(float).fillna(0)
    joblib.dump(features, FEATURE_FILE)
    return df, features


def _build_pipeline():
    """Return the standard scaler + Random Forest pipeline used by both modes."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])


def train_backtest():
    """Train on 2020-2023, evaluate on 2024, and save the backtest model."""
    df, features = load_training()

    train_df = df[df["season"] <= 2023]
    test_df = df[df["season"] == 2024]

    X_train, y_train = train_df[features], train_df["target"]
    X_test, y_test = test_df[features], test_df["target"]

    pipe = _build_pipeline()
    print("\nTraining backtest model...")
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print("\nBacktest Results (2024)")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(pipe, BACKTEST_MODEL)
    print("\nBacktest model saved.")


def train_future():
    """Train on all seasons (2020-2024) and save the future-prediction model."""
    df, features = load_training()

    train_df = df[df["season"] <= 2024]
    X_train, y_train = train_df[features], train_df["target"]

    pipe = _build_pipeline()
    print("\nTraining future model...")
    pipe.fit(X_train, y_train)

    joblib.dump(pipe, FUTURE_MODEL)
    print("\nFuture prediction model saved.")


if __name__ == "__main__":
    print("Select Mode:")
    print("1 - Train Backtest Model (2020-2023 -> Test 2024)")
    print("2 - Train Future Model (2020-2024 -> Predict 2025/26)")

    choice = input("Enter 1 or 2: ")
    if choice == "1":
        train_backtest()
    elif choice == "2":
        train_future()
