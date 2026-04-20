"""Streamlit dashboard for the 6G subsea welding PoC.

Two offshore rigs (A, B) on the California Pacific coast, each with a
welding drone tethered by fibre to a 6G edge buoy. Marine weather is
replayed from a local Open-Meteo CSV. A decision tree in ``core.ml``
picks the mission state; ``core.llm`` narrates transitions into a chat
log so the buoy feels like it's actually talking to the drone.

Run from repo root:
    ollama pull qwen2.5:3b          # optional, for real narration
    streamlit run frontend/app.py
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Make the repo root importable so ``core`` resolves regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import data as data_mod
from core import llm as llm_mod
from core import ml as ml_mod

# ---------------------------------------------------------------------------
# Page / theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="6G Subsea Welding Ops",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Instrument+Serif:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
    :root {
        --bg: #1a1a1a; --bg2: #111317; --bg3: #181a1f;
        --border: rgba(255,255,255,0.07); --border2: rgba(255,255,255,0.12);
        --text: #e8e6e0; --muted: #b0b0b0;
        --accent: #c8f135; --accent2: #a8d420;
        --red: #ff4d4d; --orange: #ff8c42; --yellow: #ffd166;
        --green: #06d6a0; --blue: #4cc9f0;
        --mono: 'DM Mono', monospace;
        --serif: 'Instrument Serif', serif;
        --sans: 'DM Sans', sans-serif;
    }

    html, body, .stApp, [class*="st-"] { font-family: var(--sans) !important; }
    .stApp { background: var(--bg) !important; color: var(--text) !important; }
    section[data-testid="stSidebar"] {
        background: var(--bg2) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* Headings: serif for emphasis, sans for the rest. */
    h1, h2, h3 { font-family: var(--serif); letter-spacing: -0.01em; font-weight: 400; }
    h4, h5, h6 { font-family: var(--sans); font-weight: 500; letter-spacing: -0.01em; }

    /* Monospace for technical labels, metric deltas, code. */
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
        font-family: var(--mono) !important;
        color: var(--muted) !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        font-size: 11px !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--serif) !important;
        color: var(--text) !important;
        font-weight: 400;
    }

    /* Sliders + toggles accent. */
    [data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }
    [data-baseweb="slider"] div[aria-valuenow] + div { background: var(--accent) !important; }

    /* Buttons. */
    div[data-testid="stButton"] > button {
        background: var(--bg3); color: var(--text);
        border: 1px solid var(--border2); border-radius: 10px;
        font-family: var(--mono); font-size: 12px;
        transition: border-color .15s, color .15s, background .15s;
    }
    div[data-testid="stButton"] > button:hover {
        border-color: var(--accent); color: var(--accent);
    }
    div[data-testid="stButton"] > button[kind="primary"],
    div[data-testid="stButton"] > button:focus:not(:active) {
        background: var(--accent); color: #0b0c0e; border-color: var(--accent);
    }

    /* Card surface used for rigs. */
    .rig-card {
        background: var(--bg3);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 18px 20px;
    }

    /* Big state word — bold serif, no italics. */
    .big-metric {
        font-family: var(--serif);
        font-size: 2.4rem;
        font-weight: 600;
        letter-spacing: -0.01em;
    }
    .state-NORMAL  { color: var(--green); }
    .state-CAUTION { color: var(--yellow); }
    .state-ABORT   { color: var(--red); }

    /* Pill badges. */
    .badge {
        display: inline-block;
        font-family: var(--mono); font-size: 11px;
        color: var(--muted);
        padding: 3px 10px; border-radius: 999px;
        border: 1px solid var(--border2);
        letter-spacing: 0.02em;
    }
    .badge.state-NORMAL   { color: var(--green);  border-color: rgba(6,214,160,0.35); }
    .badge.state-CAUTION  { color: var(--yellow); border-color: rgba(255,209,102,0.35); }
    .badge.state-ABORT    { color: var(--red);    border-color: rgba(255,77,77,0.40); }
    .badge.state-TAKEOVER {
        color: #ffd9d9; background: rgba(255,77,77,0.18);
        border-color: rgba(255,77,77,0.6); font-weight: 600;
    }
    .state-TAKEOVER { color: var(--red); }

    /* Session ops-counter strip. */
    .ops-counters {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 6px 0 14px;
        padding: 8px 14px;
        background: var(--bg3);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-family: var(--mono);
        font-size: 11px;
        letter-spacing: 0.06em;
        color: var(--muted);
        text-transform: uppercase;
    }
    .ops-counters .label { color: var(--muted); }
    .ops-counters .count {
        color: var(--text);
        font-weight: 600;
        margin-left: 4px;
    }
    .ops-counters .sep { opacity: 0.35; }
    .ops-counters .count.hil     { color: var(--red); }
    .ops-counters .count.abort   { color: var(--red); }
    .ops-counters .count.caution { color: var(--yellow); }
    .ops-counters .count.normal  { color: var(--green); }
    .ops-counters .count.recov   { color: var(--accent); }

    /* Human-in-the-loop escalation banner. */
    .hil-banner {
        background: linear-gradient(90deg, rgba(255,77,77,0.18), rgba(255,77,77,0.05));
        border: 1px solid rgba(255,77,77,0.55);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 6px 0 14px;
        font-family: var(--mono);
        font-size: 13px;
        color: #ffd9d9;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        animation: hilPulse 1.4s ease-in-out infinite;
    }
    @keyframes hilPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255,77,77,0.0); }
        50%      { box-shadow: 0 0 0 6px rgba(255,77,77,0.18); }
    }
    .hil-banner b { color: #fff; font-weight: 600; }

    /* Chat bubbles. */
    .chat-drone, .chat-buoy {
        background: var(--bg3);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 10px 14px;
        margin: 8px 0;
        font-family: var(--sans);
        font-size: 14px;
    }
    .chat-drone { border-left: 3px solid var(--blue); }
    .chat-buoy  { border-left: 3px solid var(--accent); }
    .chat-drone.takeover, .chat-buoy.takeover {
        border-left-color: var(--red);
        background: rgba(255,77,77,0.06);
    }
    .chat-drone b, .chat-buoy b {
        font-family: var(--mono); font-size: 11px;
        color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em;
        font-weight: 500;
    }

    /* Scrollable live-panel wrapper (Streamlit container with height). */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--bg2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 14px !important;
    }

    /* Live indicator dot. */
    .live-dot {
        display:inline-block; width:8px; height:8px; border-radius:50%;
        background: var(--accent); margin-right:8px; vertical-align:middle;
        box-shadow: 0 0 0 0 rgba(200,241,53,0.6);
        animation: livePulse 1.6s infinite;
    }
    @keyframes livePulse {
        0%   { box-shadow: 0 0 0 0 rgba(200,241,53,0.55); }
        70%  { box-shadow: 0 0 0 10px rgba(200,241,53,0); }
        100% { box-shadow: 0 0 0 0 rgba(200,241,53,0); }
    }

    /* Caption + general muted text. */
    .stMarkdown p, .stCaption, small, [data-testid="stCaptionContainer"] {
        color: var(--muted);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session-state bootstrap
# ---------------------------------------------------------------------------

def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("running", True)
    ss.setdefault("tick", 0)
    ss.setdefault("disaster_until_tick", -1)
    ss.setdefault("history", {s: [] for s in data_mod.STATIONS})
    ss.setdefault("chat", {s: [] for s in data_mod.STATIONS})
    ss.setdefault("last_state", {s: "NORMAL" for s in data_mod.STATIONS})
    # Latched human-in-the-loop takeover flag per rig. Set when the data stream
    # reports ``strange_geometry=1``; cleared only by an explicit operator
    # acknowledgment on the dashboard.
    ss.setdefault("takeover", {s: False for s in data_mod.STATIONS})
    ss.setdefault("takeover_tick", {s: None for s in data_mod.STATIONS})
    # Session-wide event tallies for the ops counter strip. ``recoveries``
    # counts ABORT -> NORMAL transitions only — the system coming back from
    # an emergency-retreat on its own. CAUTION -> NORMAL is too routine to
    # be a meaningful pitch metric.
    ss.setdefault(
        "counters",
        {"HIL": 0, "ABORT": 0, "CAUTION": 0, "NORMAL": 0, "recoveries": 0},
    )
    ss.setdefault("rng", np.random.default_rng(7))

_init_state()


@st.cache_data
def _load_datasets() -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Per-rig steady-state feeds + per-rig storm arcs."""
    return data_mod.load_station_feeds(), data_mod.load_storm_feeds()


station_feeds, storm_feeds = _load_datasets()
clf = ml_mod.get_classifier()


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Controls")
    tick_ms = st.slider(
        "Tick interval (ms)", 2000, 8000, 5000, step=250,
        help="Slower ticks give the Qwen LLM room to narrate every event.",
    )
    stress = st.slider("Stress test — wave multiplier", 1.0, 3.0, 1.0, step=0.1)
    use_llm = st.toggle(
        "Use Ollama Qwen 2.5 3B for narration",
        value=True,
        help="On = local LLM narration. Off = minimal template fallback.",
    )

    st.divider()
    c1, c2 = st.columns(2)
    if c1.button(
        "Pause" if st.session_state.running else "Resume",
        width="stretch",
    ):
        st.session_state.running = not st.session_state.running
    if c2.button("Reset", width="stretch"):
        for key in (
            "tick",
            "history",
            "chat",
            "last_state",
            "disaster_until_tick",
            "takeover",
            "takeover_tick",
            "counters",
        ):
            st.session_state.pop(key, None)
        _init_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Chart helpers (module-level; rebuilt inside the fragment on every tick)
# ---------------------------------------------------------------------------

_CHART_BG = "#181a1f"
_CHART_FG = "#e8e6e0"
_CHART_MUTED = "#b0b0b0"
_CHART_GRID = "rgba(255,255,255,0.06)"
_STATION_COLORS = {"A": "#c8f135", "B": "#4cc9f0"}

# Display coords for each rig on the fleet map. Deliberately offshore rather
# than the Open-Meteo sample points in the CSV metadata — those points snap
# onshore and would put the pins on land at this zoom. The weather data is
# still sampled from the real coords; these just position the rigs.
#   Rig A: ~110 km SW of the Golden Gate, well past the Farallones.
#   Rig B: ~65 km SW of Long Beach, past San Clemente Island in open Pacific.
RIG_COORDS: dict[str, dict[str, float | str]] = {
    "A": {"lat": 37.50, "lon": -124.00, "label": "San Francisco offshore"},
    "B": {"lat": 33.30, "lon": -118.90, "label": "Los Angeles offshore"},
}

# Map state -> hex colour for the fleet-map pins. TAKEOVER piggybacks on
# ABORT's red because operationally they're both "do not proceed" signals.
_STATE_MAP_COLOR = {
    "NORMAL": "#06d6a0",
    "CAUTION": "#ffd166",
    "ABORT": "#ff4d4d",
    "TAKEOVER": "#ff4d4d",
}


def _combined_df() -> pd.DataFrame:
    frames = []
    for s, rows in st.session_state.history.items():
        if rows:
            frame = pd.DataFrame(rows)
            frame["station"] = s
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _style_fig(fig: go.Figure, title: str, y_range=None) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(family="DM Mono, monospace",
                                          size=12, color=_CHART_MUTED)),
        paper_bgcolor=_CHART_BG,
        plot_bgcolor=_CHART_BG,
        font=dict(family="DM Sans, sans-serif", color=_CHART_FG, size=12),
        height=280,
        margin=dict(l=20, r=20, t=48, b=24),
        legend=dict(orientation="h", y=-0.2, font=dict(family="DM Mono, monospace", size=11)),
        xaxis=dict(gridcolor=_CHART_GRID, linecolor=_CHART_GRID, zerolinecolor=_CHART_GRID),
        yaxis=dict(gridcolor=_CHART_GRID, linecolor=_CHART_GRID,
                    zerolinecolor=_CHART_GRID,
                    range=y_range),
    )


def _build_fleet_map(snapshots: dict[str, dict]) -> go.Figure:
    """Return a scoped Pacific-coast map with one pin per rig.

    ``snapshots`` is keyed by station id and each value is the most recent
    history row plus a ``takeover`` flag. Missing stations are rendered in a
    neutral gray so the map still makes sense on tick 0 before any data has
    flowed through the pipeline.
    """

    lats: list[float] = []
    lons: list[float] = []
    colors: list[str] = []
    labels: list[str] = []
    hover: list[str] = []

    takeover_lats: list[float] = []
    takeover_lons: list[float] = []

    for station in data_mod.STATIONS:
        coords = RIG_COORDS[station]
        snap = snapshots.get(station, {}) or {}
        raw_state = snap.get("state") or "NORMAL"
        in_takeover = bool(snap.get("takeover"))
        display_state = "TAKEOVER" if in_takeover else raw_state

        lats.append(float(coords["lat"]))
        lons.append(float(coords["lon"]))
        # Fall back to a muted gray when we genuinely have no telemetry yet
        # so the pin doesn't lie about a healthy state.
        if not snap:
            colors.append("#6b6b6b")
        else:
            colors.append(_STATE_MAP_COLOR.get(display_state, "#6b6b6b"))
        labels.append(f"  {station}")
        hover.append(
            f"<b>Rig {station}</b> — {coords['label']}<br>"
            f"state: {display_state}<br>"
            f"R(t): {snap.get('R', 0.0):.2f}<br>"
            f"wave: {snap.get('wave', 0.0):.2f} m<br>"
            f"current: {snap.get('current', 0.0):.2f} m/s<br>"
            f"source: {snap.get('source', '—')}"
        )

        if in_takeover:
            takeover_lats.append(float(coords["lat"]))
            takeover_lons.append(float(coords["lon"]))

    fig = go.Figure()

    # Backhaul line — a dotted connector between the two buoys. Pure visual
    # shorthand for "these rigs share a 6G edge mesh", no routing intent.
    fig.add_trace(
        go.Scattergeo(
            lat=[RIG_COORDS["A"]["lat"], RIG_COORDS["B"]["lat"]],
            lon=[RIG_COORDS["A"]["lon"], RIG_COORDS["B"]["lon"]],
            mode="lines",
            line=dict(width=1, color="rgba(200, 241, 53, 0.22)", dash="dot"),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # HIL emphasis ring — drawn before pins so the coloured pin sits on top.
    if takeover_lats:
        fig.add_trace(
            go.Scattergeo(
                lat=takeover_lats,
                lon=takeover_lons,
                mode="markers",
                marker=dict(
                    size=52,
                    color="rgba(255, 77, 77, 0.12)",
                    line=dict(color="#ff4d4d", width=2),
                ),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    # Main rig pins.
    fig.add_trace(
        go.Scattergeo(
            lat=lats,
            lon=lons,
            mode="markers+text",
            text=labels,
            textposition="middle right",
            textfont=dict(family="DM Mono, monospace", size=14, color=_CHART_FG),
            marker=dict(
                size=26,
                color=colors,
                line=dict(color="rgba(0,0,0,0.55)", width=1.5),
            ),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
        )
    )

    fig.update_geos(
        projection_type="mercator",
        lonaxis_range=[-128, -114],
        lataxis_range=[30.5, 41.0],
        showland=True, landcolor="#181a1f",
        showocean=True, oceancolor="#0e1013",
        showlakes=True, lakecolor="#0e1013",
        showcoastlines=True, coastlinecolor="rgba(255,255,255,0.18)",
        coastlinewidth=0.6,
        showcountries=False,
        showframe=False,
        showsubunits=True, subunitcolor="rgba(255,255,255,0.08)",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=540,
        hoverlabel=dict(
            bgcolor="#181a1f",
            bordercolor="rgba(255,255,255,0.12)",
            font=dict(family="DM Mono, monospace", size=11, color=_CHART_FG),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Live dashboard — a single Streamlit fragment that reruns at ``tick_ms``
# without triggering a full-page rerun. Everything below the sidebar lives
# inside here: top bar, HIL banner, KPI cards, charts, chat panels.
# ---------------------------------------------------------------------------

_run_every = (tick_ms / 1000.0) if st.session_state.running else None


@st.fragment(run_every=_run_every)
def live_dashboard() -> None:
    # --- Advance one tick (only when running) ---------------------------
    if st.session_state.running:
        st.session_state.tick += 1
    t = st.session_state.tick

    # 6G URLLC target: sub-millisecond. We sample 100-500 µs to stay well
    # under 1 ms while looking plausibly noisy tick-to-tick. The value is
    # displayed in microseconds on the KPI card for pitch readability and
    # passed to the LLM prompts in milliseconds so narration still reads
    # "0.3 ms" rather than "0.0003 s".
    latency_us = random.randint(100, 500)
    latency_ms = latency_us / 1000.0

    # --- Top bar --------------------------------------------------------
    top_cols = st.columns([3, 2, 2, 3])
    top_cols[0].markdown(
        "<h2 style='margin:0; font-family:var(--serif);'>Subsea Welding Ops</h2>"
        "<div style='font-family:var(--mono); font-size:11px; color:var(--muted); "
        "text-transform:uppercase; letter-spacing:0.08em; margin-top:4px;'>"
        "6G edge · pacific coast fleet · v0.1"
        "</div>",
        unsafe_allow_html=True,
    )
    top_cols[1].metric("6G URLLC latency", f"{latency_us} µs")
    top_cols[2].metric("Tick", t)

    disaster_active = t < st.session_state.disaster_until_tick
    if top_cols[3].button(
        "Trigger Disaster (30 ticks)" if not disaster_active else "Disaster active…",
        width="stretch",
        disabled=disaster_active,
        key="disaster_btn",
    ):
        st.session_state.disaster_until_tick = t + 30
        disaster_active = True

    # --- Compute this tick for every station ---------------------------
    if st.session_state.running:
        disaster_window = 30
        disaster_start = st.session_state.disaster_until_tick - disaster_window
        disaster_elapsed = max(0, t - disaster_start) if disaster_active else 0

        for station in data_mod.STATIONS:
            tick = data_mod.make_tick(
                station_feeds, storm_feeds, t, station,
                stress_multiplier=stress,
                disaster_active=disaster_active,
                disaster_elapsed=disaster_elapsed,
            )

            lam = ml_mod.compute_lambda(
                tick["wave_height"], tick["current_velocity"]
            )
            R = ml_mod.reliability(lam, t_seconds=10.0)
            state = clf.predict(
                tick["wave_height"], tick["current_velocity"], tick["wave_period"]
            )

            st.session_state.history[station].append(
                {
                    "tick": t,
                    "wave": tick["wave_height"],
                    "current": tick["current_velocity"],
                    "period": tick["wave_period"],
                    "R": R,
                    "state": state,
                    "source": tick["source"],
                    "strange_geometry": tick.get("strange_geometry", 0),
                }
            )

            # Narrate on classifier state change.
            prev_state = st.session_state.last_state[station]
            if state != prev_state:
                st.session_state.counters[state] = (
                    st.session_state.counters.get(state, 0) + 1
                )
                if state == "NORMAL" and prev_state == "ABORT":
                    st.session_state.counters["recoveries"] += 1

                drone_line = llm_mod.narrate(
                    "drone", state, tick["wave_height"], tick["current_velocity"],
                    R, latency_ms, use_llm=use_llm, rig=station,
                )
                buoy_line = llm_mod.narrate(
                    "buoy", state, tick["wave_height"], tick["current_velocity"],
                    R, latency_ms, use_llm=use_llm, rig=station,
                )
                st.session_state.chat[station].append(
                    {"tick": t, "state": state, "drone": drone_line, "buoy": buoy_line}
                )
                st.session_state.last_state[station] = state

            # Latched human-in-the-loop escalation on ``strange_geometry`` flag.
            if tick.get("strange_geometry", 0) == 1 and not st.session_state.takeover[station]:
                st.session_state.takeover[station] = True
                st.session_state.takeover_tick[station] = t
                st.session_state.counters["HIL"] += 1
                drone_line = llm_mod.narrate(
                    "drone", "TAKEOVER", tick["wave_height"], tick["current_velocity"],
                    R, latency_ms, use_llm=use_llm, rig=station,
                )
                buoy_line = llm_mod.narrate(
                    "buoy", "TAKEOVER", tick["wave_height"], tick["current_velocity"],
                    R, latency_ms, use_llm=use_llm, rig=station,
                )
                st.session_state.chat[station].append(
                    {"tick": t, "state": "TAKEOVER", "drone": drone_line, "buoy": buoy_line}
                )

    # --- HIL escalation banner -----------------------------------------
    pending_takeovers = [
        s for s in data_mod.STATIONS if st.session_state.takeover.get(s)
    ]
    if pending_takeovers:
        rig_list = ", ".join(f"Rig {s}" for s in pending_takeovers)
        st.markdown(
            f"<div class='hil-banner'>"
            f"<b>Human takeover requested</b> — {rig_list} · anomalous weld geometry · "
            f"escalated over 6G URLLC slice"
            f"</div>",
            unsafe_allow_html=True,
        )

    # --- KPI cards per rig ---------------------------------------------
    # --- Session ops counters ------------------------------------------
    counters = st.session_state.counters
    st.markdown(
        "<div class='ops-counters'>"
        "<span class='label'>session events</span>"
        "<span class='sep'>·</span>"
        f"<span>HIL <span class='count hil'>{counters['HIL']}</span></span>"
        "<span class='sep'>·</span>"
        f"<span>abort <span class='count abort'>{counters['ABORT']}</span></span>"
        "<span class='sep'>·</span>"
        f"<span>caution <span class='count caution'>{counters['CAUTION']}</span></span>"
        "<span class='sep'>·</span>"
        f"<span>normal <span class='count normal'>{counters['NORMAL']}</span></span>"
        "<span class='sep'>→</span>"
        f"<span>recoveries <span class='count recov'>{counters['recoveries']}</span></span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # --- Fleet map ------------------------------------------------------
    # Both rigs on a single Pacific-coast map. Pin colour tracks the live
    # classifier state, and an outer red ring marks any rig currently in
    # Human-in-the-Loop takeover. Rebuilt each fragment tick so the colours
    # stay in sync with the KPI cards below.
    map_snapshots: dict[str, dict] = {}
    for station in data_mod.STATIONS:
        hist = st.session_state.history.get(station, [])
        if not hist:
            continue
        last = hist[-1]
        map_snapshots[station] = {
            "state": last["state"],
            "R": last["R"],
            "wave": last["wave"],
            "current": last["current"],
            "source": last.get("source", "—"),
            "takeover": bool(st.session_state.takeover.get(station, False)),
        }
    st.markdown("#### Fleet map")
    st.plotly_chart(
        _build_fleet_map(map_snapshots),
        use_container_width=True,
        key=f"fleet_map_{t}",
        config={"displayModeBar": False, "scrollZoom": False},
    )

    st.markdown("#### Live rig telemetry")
    rig_cols = st.columns(2)
    rig_locations = {"A": "san francisco", "B": "los angeles basin"}
    for col, station in zip(rig_cols, data_mod.STATIONS):
        hist = st.session_state.history[station]
        if not hist:
            continue
        last = hist[-1]
        state = last["state"]
        in_takeover = st.session_state.takeover.get(station, False)
        display_state = "TAKEOVER" if in_takeover else state
        src = last.get("source", "")
        if src.startswith("storm"):
            source_badge = f"{src}.csv"
        else:
            source_badge = "sf_underwater.csv" if station == "A" else "la_underwater.csv"
        with col:
            takeover_badge = (
                f"<span class='badge state-TAKEOVER'>HIL · takeover · "
                f"t={st.session_state.takeover_tick[station]}</span> "
                if in_takeover else ""
            )
            st.markdown(
                f"<div class='rig-card'>"
                f"<div style='display:flex; justify-content:space-between; "
                f"align-items:center; margin-bottom:10px;'>"
                f"  <div>"
                f"    <span class='badge'>rig · {station} · {rig_locations[station]}</span> "
                f"    <span class='badge'>data · {source_badge}</span> "
                f"    {takeover_badge}"
                f"  </div>"
                f"  <span class='state-{display_state} big-metric'>"
                f"{display_state.lower()}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
            k1, k2, k3 = st.columns(3)
            k1.metric("Reliability R", f"{last['R']:.2f}")
            k3.metric("Wave", f"{last['wave']:.2f} m")
            st.markdown("</div>", unsafe_allow_html=True)

            if in_takeover:
                if st.button(
                    f"Acknowledge takeover — Rig {station}",
                    key=f"ack_{station}",
                    width="stretch",
                    type="primary",
                ):
                    st.session_state.takeover[station] = False
                    st.session_state.takeover_tick[station] = None
                    st.rerun(scope="fragment")

    # --- Charts --------------------------------------------------------
    combined = _combined_df()
    if not combined.empty:
        ch1, ch2 = st.columns(2)

        fig_wave = go.Figure()
        for s in data_mod.STATIONS:
            sub = combined[combined.station == s]
            fig_wave.add_trace(go.Scatter(
                x=sub["tick"], y=sub["wave"], mode="lines", name=f"Rig {s}",
                line=dict(color=_STATION_COLORS.get(s, "#c8f135"), width=2),
            ))
        _style_fig(fig_wave, "WAVE HEIGHT (m)")
        ch1.plotly_chart(fig_wave, width="stretch", key=f"wave_chart_{t}")

        fig_R = go.Figure()
        for s in data_mod.STATIONS:
            sub = combined[combined.station == s]
            color = _STATION_COLORS.get(s, "#c8f135")
            fig_R.add_trace(go.Scatter(
                x=sub["tick"], y=sub["R"], mode="lines", name=f"Rig {s} · R",
                line=dict(color=color, width=2),
            ))
        _style_fig(fig_R, "RELIABILITY R(t)", y_range=[0, 1.05])
        ch2.plotly_chart(fig_R, width="stretch", key=f"r_chart_{t}")

    # --- Chat panels ---------------------------------------------------
    st.markdown("#### Agent chat log — state transitions only")
    chat_cols = st.columns(2)
    for col, station in zip(chat_cols, data_mod.STATIONS):
        with col:
            st.markdown(
                f"<span class='live-dot'></span> **Rig {station} — drone ⇄ buoy**",
                unsafe_allow_html=True,
            )
            panel = st.container(height=420, border=True)
            msgs = st.session_state.chat[station]
            if not msgs:
                panel.caption("Awaiting first state transition…")
            else:
                for m in reversed(msgs):
                    extra = " takeover" if m["state"] == "TAKEOVER" else ""
                    panel.markdown(
                        f"<div class='chat-drone{extra}'><b>drone</b> "
                        f"<span class='badge state-{m['state']}'>"
                        f"t={m['tick']} · {m['state']}</span><br/>"
                        f"{m['drone']}</div>",
                        unsafe_allow_html=True,
                    )
                    panel.markdown(
                        f"<div class='chat-buoy{extra}'><b>buoy</b><br/>{m['buoy']}</div>",
                        unsafe_allow_html=True,
                    )


live_dashboard()

st.caption(
    "PoC · Ericsson Hackathon 2026 · Decisions by scikit-learn + R=e^(-λt) · "
    "Narration by Qwen 2.5 3B (optional)"
)
