"""
Ericsson Hackathon 2026 - Underwater Welding Drone Safety Classifier
---------------------------------------------------------------------
Trains a decision tree on synthetic data, then classifies marine forecast
rows into one of three states for the drone:

    0 = CONTINUE              (conditions OK, keep welding)
    1 = GO DOCK               (weather unsafe, return to dock)
    2 = HUMAN INTERVENTION    (strange geometry detected, takeover needed)

The model is applied to two CSVs:
    * the real Open-Meteo forecast
    * a separate testing_data.csv for validation / demo variety
"""

import glob
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


# -----------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------
FEATURE_COLS = ["wave_height", "current", "wave_period", "sst", "strange_geometry"]
STATUS_MAP   = {0: "CONTINUE", 1: "GO DOCK", 2: "HUMAN INTERVENTION"}


def load_csv(path):
    """
    Load an Open-Meteo-style CSV, clean it, rename columns to short names,
    and add the strange_geometry column if it's missing.

    Handles BOTH CSV variants:
      - Raw Open-Meteo download (3 metadata lines on top -> skip them)
      - Stripped CSV where the first line is the column header directly
    """
    # Sniff the first line so we know whether to skip metadata rows
    with open(path) as f:
        first_line = f.readline().strip()
    skip = 3 if first_line.startswith("latitude") else 0

    df = pd.read_csv(path, skiprows=skip)
    df = df.dropna().reset_index(drop=True)

    # Rename ugly units-in-the-name columns to something typeable
    df = df.rename(columns={
        "wave_height (m)":               "wave_height",
        "ocean_current_velocity (km/h)": "current",
        "sea_surface_temperature (°C)":  "sst",
        "wave_period (s)":               "wave_period",
    })

    # time is only for display, not prediction - skip it if missing
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

    # Fallback for CSVs that don't have a strange_geometry column yet
    if "strange_geometry" not in df.columns:
        print(f"Note: no 'strange_geometry' column in {path}, injecting defaults.")
        df["strange_geometry"] = 0
        if len(df) > 3:   df.loc[3,   "strange_geometry"] = 1
        if len(df) > 100: df.loc[100, "strange_geometry"] = 1

    # Make sure all feature columns the model needs are present
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"\n{path} is missing required columns: {missing}\n"
            f"Expected: {FEATURE_COLS}\n"
            f"Found:    {list(df.columns)}"
        )

    return df


def classify(df, clf):
    """Run the trained model on a DataFrame; adds 'prediction' and 'status'."""
    X = df[FEATURE_COLS].values
    df["prediction"] = clf.predict(X)
    df["status"]     = df["prediction"].map(STATUS_MAP)
    return df


def summarize(name, df):
    """Print a tidy summary of predictions for one dataset."""
    print(f"\n=== {name} ===")
    print(f"Total rows: {len(df)}")
    print(df["status"].value_counts())
    print("\nFirst 20 rows:")
    # Only display columns that exist in this DataFrame
    preferred = ["time", "wave_height", "current", "wave_period",
                 "strange_geometry", "status"]
    cols = [c for c in preferred if c in df.columns]
    print(df[cols].head(20).to_string(index=False))


# -----------------------------------------------------------------------
# 1. TRAIN THE MODEL (synthetic labeled data + physics rules)
# -----------------------------------------------------------------------
WAVE_MAX_M      = 1.5
CURRENT_MAX_KMH = 3.6
PERIOD_MIN_S    = 6.0

def label_row(wave, current, period, geom):
    """Return the true class given the physics thresholds."""
    if geom == 1:
        return 2
    if wave > WAVE_MAX_M or current > CURRENT_MAX_KMH or period < PERIOD_MIN_S:
        return 1
    return 0

rng = np.random.default_rng(42)
n_train = 2000

train_wave    = rng.uniform(0, 4,  n_train)
train_current = rng.uniform(0, 10, n_train)
train_period  = rng.uniform(3, 15, n_train)
train_sst     = rng.uniform(5, 25, n_train)
train_geom    = rng.choice([0, 1], n_train, p=[0.9, 0.1])

train_labels = np.zeros(n_train, dtype=int)
for i in range(n_train):
    train_labels[i] = label_row(
        train_wave[i], train_current[i],
        train_period[i], train_geom[i]
    )

X_train = np.column_stack([
    train_wave, train_current, train_period, train_sst, train_geom
])

clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_train, train_labels)
print(f"Training accuracy: {clf.score(X_train, train_labels):.3f}")


# -----------------------------------------------------------------------
# 2. CLASSIFY THE OPEN-METEO FORECAST
# -----------------------------------------------------------------------
# glob lets this work whether your filename uses dots or underscores
forecast_path = glob.glob("open-meteo-*.csv")[0]
forecast_df   = load_csv(forecast_path)
forecast_df   = classify(forecast_df, clf)
summarize("Open-Meteo forecast", forecast_df)


# -----------------------------------------------------------------------
# 3. CLASSIFY THE TESTING DATA
# -----------------------------------------------------------------------
test_df = load_csv("testing_data.csv")
test_df = classify(test_df, clf)
summarize("testing_data.csv", test_df)