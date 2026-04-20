"""Marine-weather replay for two independent oil rigs.

Three CSVs back the simulation:

  * ``NORMAL_CSV``  — real Open-Meteo marine data for **Rig A** (North Sea).
  * ``LA_CSV``      — real Open-Meteo marine data for **Rig B** (LA basin).
    Entirely independent from Rig A, so the two dashboards diverge naturally
    without us having to synthesise noise.
  * ``STORM_CSV``   — hand-crafted 30-hour storm arc shared by both rigs
    while the Disaster button is active.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
NORMAL_CSV = REPO_ROOT / "open-meteo-37.79N122.46W18m.csv"
LA_CSV = REPO_ROOT / "LA_underwater_data.csv"
STORM_CSV = REPO_ROOT / "storm.csv"

STATIONS = ("A", "B")

# Which CSV backs each rig under steady-state operations.
STATION_SOURCES = {
    "A": NORMAL_CSV,
    "B": LA_CSV,
}

# Tick dict keys (plain dict; documented here for readers):
#   station, t_index, source, wave_height, current_velocity, wave_period,
#   sea_surface_temp, strange_geometry, disaster_active.
#
# ``strange_geometry`` (0/1) is a data-driven anomaly flag that, when raised,
# should trigger a Human-in-the-Loop takeover request on the dashboard.

Tick = dict[str, Any]


def _read_marine_csv(path: Path) -> pd.DataFrame:
    """Shared loader for the Open-Meteo-style marine CSVs.

    Two header formats are supported transparently:
      * Trimmed form — single header row starting with ``time,`` (used by the
        main Open-Meteo CSV after the repo cleanup and by ``testing_data.csv``).
        May include an extra ``strange_geometry`` column.
      * Raw Open-Meteo export — two lines of metadata, one blank line, then
        the real header (used by ``LA_underwater_data.csv`` and ``storm.csv``).
        No anomaly column.
    """

    if not path.exists():
        raise FileNotFoundError(f"Marine CSV not found at {path}")

    with path.open("r", encoding="utf-8") as fh:
        first = fh.readline().strip().lower()
    skiprows = 0 if first.startswith("time,") else 3

    df = pd.read_csv(path, skiprows=skiprows)

    rename = {
        "time": "time",
        "wave_height (m)": "wave_height",
        "ocean_current_velocity (km/h)": "current_velocity_kmh",
        "sea_surface_temperature (°C)": "sea_surface_temp",
        "wave_period (s)": "wave_period",
        "strange_geometry": "strange_geometry",
    }
    df = df.rename(columns=rename)

    df["time"] = pd.to_datetime(df["time"])
    # Convert km/h -> m/s so the spec threshold "> 1 m/s" works directly.
    df["current_velocity"] = df["current_velocity_kmh"] / 3.6

    if "strange_geometry" not in df.columns:
        df["strange_geometry"] = 0
    df["strange_geometry"] = df["strange_geometry"].fillna(0).astype(int)

    return df.reset_index(drop=True)


def load_normal_csv(path: Path = NORMAL_CSV) -> pd.DataFrame:
    """Rig A steady-state feed (North Sea)."""
    return _read_marine_csv(path)


def load_la_csv(path: Path = LA_CSV) -> pd.DataFrame:
    """Rig B steady-state feed (LA basin)."""
    return _read_marine_csv(path)


def load_storm_csv(path: Path = STORM_CSV) -> pd.DataFrame:
    """Hand-crafted storm timeline — shared by both rigs during Disaster mode."""
    return _read_marine_csv(path)


def load_station_feeds() -> dict[str, pd.DataFrame]:
    """Map each station id to its dedicated steady-state dataframe."""
    return {
        "A": load_normal_csv(),
        "B": load_la_csv(),
    }


# Backwards-compatible alias.
load_marine_csv = load_normal_csv


def _row(df: pd.DataFrame, i: int) -> dict:
    """Plain row accessor — no per-station noise, no offset."""
    row = df.iloc[i % len(df)]
    return {
        "wave_height": float(row["wave_height"]),
        "current_velocity": float(row["current_velocity"]),
        "wave_period": float(row["wave_period"]),
        "sea_surface_temp": float(row["sea_surface_temp"]),
        "strange_geometry": int(row.get("strange_geometry", 0)),
    }


def make_tick(
    station_feeds: dict[str, pd.DataFrame],
    storm_df: pd.DataFrame,
    t_index: int,
    station: str,
    *,
    stress_multiplier: float = 1.0,
    disaster_active: bool = False,
    disaster_elapsed: int = 0,
) -> Tick:
    """Produce one tick dict for the given station.

    Under steady-state we pull from ``station_feeds[station]`` at
    ``t_index``. When ``disaster_active`` is True every station switches
    to ``storm_df`` indexed by ``disaster_elapsed`` so the storm arc plays
    from the start each time the operator triggers Disaster.

    ``stress_multiplier`` still applies on top — a great way to push a
    benign day into CAUTION on stage without triggering a full storm.
    """

    if disaster_active:
        raw = _row(storm_df, disaster_elapsed)
        source = "storm"
    else:
        feed = station_feeds[station]
        raw = _row(feed, t_index)
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
        "strange_geometry": int(raw.get("strange_geometry", 0)),
        "disaster_active": disaster_active,
    }
