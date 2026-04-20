"""Marine-weather replay for two simulated North Sea oil rigs.

Two CSVs back the simulation:

  * ``NORMAL_CSV``  — real Open-Meteo marine data, played in a loop for
    "steady state" operations.
  * ``STORM_CSV``   — a hand-crafted 30-hour North Sea storm arc (build-up,
    peak, subsidence). Pulled in only while the Disaster button is active.

Station B is derived from Station A with a time offset and small Gaussian
noise so the two rigs show visibly different telemetry without us needing a
second pair of CSVs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
NORMAL_CSV = REPO_ROOT / "open-meteo-37.79N122.46W18m.csv"
STORM_CSV = REPO_ROOT / "storm.csv"

STATIONS = ("A", "B")

# Station B lags Station A by this many CSV rows (hours) so the two
# timeseries visibly diverge on the dashboard.
STATION_B_OFFSET_HOURS = 6
STATION_B_NOISE_SIGMA = {
    "wave_height": 0.08,
    "current_velocity": 0.15,
    "wave_period": 0.25,
}

# Tick dict keys (plain dict; documented here for readers):
#   station, t_index, source, wave_height, current_velocity, wave_period,
#   sea_surface_temp, disaster_active.

Tick = dict[str, Any]


def _read_marine_csv(path: Path) -> pd.DataFrame:
    """Shared loader for the Open-Meteo CSV schema.

    Both files use the same two-line metadata header followed by a blank
    line, so we skip the first 3 rows.
    """

    if not path.exists():
        raise FileNotFoundError(f"Marine CSV not found at {path}")

    df = pd.read_csv(path, skiprows=3)
    df.columns = [
        "time",
        "wave_height",
        "current_velocity_kmh",
        "sea_surface_temp",
        "wave_period",
    ]
    df["time"] = pd.to_datetime(df["time"])
    # Convert km/h -> m/s so the threshold "> 1 m/s" from the spec works directly.
    df["current_velocity"] = df["current_velocity_kmh"] / 3.6
    return df.reset_index(drop=True)


def load_normal_csv(path: Path = NORMAL_CSV) -> pd.DataFrame:
    """Real Open-Meteo marine data — benign operations."""
    return _read_marine_csv(path)


def load_storm_csv(path: Path = STORM_CSV) -> pd.DataFrame:
    """Hand-crafted storm timeline — replayed during Disaster mode."""
    return _read_marine_csv(path)


# Backwards-compatible alias.
load_marine_csv = load_normal_csv


def _station_row(df: pd.DataFrame, i: int, station: str) -> dict:
    """Raw weather row for one station at row index ``i``.

    Station A uses the given df directly. Station B samples an offset row and
    adds zero-mean Gaussian noise seeded by the tick index so the series is
    both varied and reproducible.
    """

    n = len(df)
    if station == STATIONS[0]:
        row = df.iloc[i % n]
        return {
            "wave_height": float(row["wave_height"]),
            "current_velocity": float(row["current_velocity"]),
            "wave_period": float(row["wave_period"]),
            "sea_surface_temp": float(row["sea_surface_temp"]),
        }

    j = (i + STATION_B_OFFSET_HOURS) % n
    row = df.iloc[j]
    rng = np.random.default_rng(seed=1000 + i)
    return {
        "wave_height": max(
            0.0,
            float(row["wave_height"])
            + rng.normal(0, STATION_B_NOISE_SIGMA["wave_height"]),
        ),
        "current_velocity": max(
            0.0,
            float(row["current_velocity"])
            + rng.normal(0, STATION_B_NOISE_SIGMA["current_velocity"]),
        ),
        "wave_period": max(
            0.1,
            float(row["wave_period"])
            + rng.normal(0, STATION_B_NOISE_SIGMA["wave_period"]),
        ),
        "sea_surface_temp": float(row["sea_surface_temp"]),
    }


def make_tick(
    normal_df: pd.DataFrame,
    storm_df: pd.DataFrame,
    t_index: int,
    station: str,
    *,
    stress_multiplier: float = 1.0,
    disaster_active: bool = False,
    disaster_elapsed: int = 0,
) -> Tick:
    """Produce one tick dict for the given station.

    When ``disaster_active`` is True we pull from ``storm_df`` indexed by
    ``disaster_elapsed`` (0, 1, 2, …) so the storm arc plays from the start
    each time the operator triggers Disaster. Otherwise we pull from
    ``normal_df`` indexed by the global ``t_index``.

    ``stress_multiplier`` still applies on top — a great way to push a
    benign day into CAUTION on stage without triggering a full storm.
    """

    if disaster_active:
        raw = _station_row(storm_df, disaster_elapsed, station)
        source = "storm"
    else:
        raw = _station_row(normal_df, t_index, station)
        source = "normal"

    wave = raw["wave_height"] * stress_multiplier
    current = raw["current_velocity"] * stress_multiplier

    return {
        "station": station,
        "t_index": t_index,
        "source": source,
        "wave_height": wave,
        "current_velocity": current,
        "wave_period": raw["wave_period"],
        "sea_surface_temp": raw["sea_surface_temp"],
        "disaster_active": disaster_active,
    }
