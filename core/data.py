"""Marine-weather replay for two independent Pacific-coast rigs.

Four CSVs back the simulation:

  * ``NORMAL_CSV``   — real Open-Meteo marine data for **Rig A** (SF offshore).
  * ``LA_CSV``       — real Open-Meteo marine data for **Rig B** (LA basin).
    Entirely independent from Rig A, so the two dashboards diverge naturally
    without us having to synthesise noise.
  * ``STORM_SF_CSV`` — hand-crafted 30-hour oscillating storm arc for Rig A
    while the Disaster button is active.
  * ``STORM_LA_CSV`` — the same storm system as seen from Rig B: phase-lagged
    roughly one hour, with slightly lower peaks to reflect the partial
    Channel-Islands shelter. Different enough that the two dashboards tell
    subtly different stories during Disaster, similar enough that it reads
    as one weather event.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
NORMAL_CSV = REPO_ROOT / "open-meteo-37.79N122.46W18m.csv"
LA_CSV = REPO_ROOT / "LA_underwater_data.csv"
STORM_SF_CSV = REPO_ROOT / "storm_sf.csv"
STORM_LA_CSV = REPO_ROOT / "storm_la.csv"

STATIONS = ("A", "B")

# Which CSV backs each rig under steady-state operations.
STATION_SOURCES = {
    "A": NORMAL_CSV,
    "B": LA_CSV,
}

# Per-station storm feed for Disaster mode. Each rig gets its own storm CSV
# so the fleet map / KPI cards / chat diverge realistically even under the
# same weather event.
STATION_STORM_SOURCES = {
    "A": STORM_SF_CSV,
    "B": STORM_LA_CSV,
}

# Per-station replay cursor offset, in rows. The LA basin CSV spends its
# first ~20 rows almost perfectly flat (wave ~0.74 m, current ~0.11 m/s),
# which makes Rig B look inert for the opening minute of the demo. We fast-
# forward its cursor a bit so the graphs move from tick 0 and the first
# ``strange_geometry`` trip hits early enough to tell a story.
STATION_START_OFFSET = {
    "A": 0,
    "B": 20,
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
        SF feed, the hand-crafted storm CSVs, and ``testing_data.csv``). May
        include an extra ``strange_geometry`` column.
      * Raw Open-Meteo export — two lines of metadata, one blank line, then
        the real header (used by ``LA_underwater_data.csv``). No anomaly
        column.
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


def load_station_feeds() -> dict[str, pd.DataFrame]:
    """Map each station id to its dedicated steady-state dataframe."""
    return {
        "A": load_normal_csv(),
        "B": load_la_csv(),
    }


def load_storm_feeds() -> dict[str, pd.DataFrame]:
    """Map each station id to its dedicated Disaster-mode storm dataframe."""
    return {
        station: _read_marine_csv(path)
        for station, path in STATION_STORM_SOURCES.items()
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
    storm_feeds: dict[str, pd.DataFrame],
    t_index: int,
    station: str,
    *,
    stress_multiplier: float = 1.0,
    disaster_active: bool = False,
    disaster_elapsed: int = 0,
) -> Tick:
    """Produce one tick dict for the given station.

    Under steady-state we pull from ``station_feeds[station]`` at
    ``t_index``. When ``disaster_active`` is True each station switches to
    its own ``storm_feeds[station]`` indexed by ``disaster_elapsed`` so the
    storm arc plays from the start each time the operator triggers Disaster
    — and Rig A / Rig B see related-but-distinct weather.

    ``stress_multiplier`` still applies on top — a great way to push a
    benign day into CAUTION on stage without triggering a full storm.

    ``source`` on the returned tick is ``"normal"`` during steady-state and
    ``"storm_sf"`` / ``"storm_la"`` during Disaster so the UI badges can
    show which file is driving each rig.
    """

    if disaster_active:
        raw = _row(storm_feeds[station], disaster_elapsed)
        source = "storm_sf" if station == "A" else "storm_la"
    else:
        feed = station_feeds[station]
        offset = STATION_START_OFFSET.get(station, 0)
        raw = _row(feed, t_index + offset)
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
