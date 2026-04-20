"""Streamlit dashboard for the 5G/6G underwater welding PoC.

Two simulated North Sea oil rigs (A, B), each with one welding drone
tethered by fibre to a 5G edge buoy. Marine weather is replayed from a
local Open-Meteo CSV. A decision tree in ``core.ml`` picks the mission state;
``core.llm`` narrates transitions into a chat log.

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
from streamlit_autorefresh import st_autorefresh

from core import data as data_mod
from core import llm as llm_mod
from core import ml as ml_mod

# ---------------------------------------------------------------------------
# Page / theme
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="5G/6G Subsea Welding Ops",
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

    /* Big state word — serif italic for drama. */
    .big-metric {
        font-family: var(--serif);
        font-size: 2.4rem;
        font-weight: 400;
        font-style: italic;
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
    ss.setdefault("weld", {s: 92.0 for s in data_mod.STATIONS})
    ss.setdefault("last_state", {s: "NORMAL" for s in data_mod.STATIONS})
    # Latched human-in-the-loop takeover flag per rig. Set when the data stream
    # reports ``strange_geometry=1``; cleared only by an explicit operator
    # acknowledgment on the dashboard.
    ss.setdefault("takeover", {s: False for s in data_mod.STATIONS})
    ss.setdefault("takeover_tick", {s: None for s in data_mod.STATIONS})
    ss.setdefault("rng", np.random.default_rng(7))

_init_state()


@st.cache_data
def _load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    return data_mod.load_normal_csv(), data_mod.load_storm_csv()


normal_df, storm_df = _load_datasets()
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
    st.subheader("Model insights")
    st.caption("Decision tree — feature importances")
    imp = clf.feature_importances()
    st.bar_chart(pd.Series(imp).sort_values(ascending=False))

    st.caption("Reliability model")
    st.latex(r"R(t) = e^{-\lambda \, t}")
    st.latex(
        r"\lambda = 0.05\,H_s + 0.03\,U_c + 0.02\,\frac{100-W}{100}"
    )
    st.caption("Hs = wave height (m), Uc = current (m/s), W = weld integrity (%)")

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
            "weld",
            "last_state",
            "disaster_until_tick",
            "takeover",
            "takeover_tick",
        ):
            st.session_state.pop(key, None)
        _init_state()
        st.rerun()


# ---------------------------------------------------------------------------
# Top bar
# ---------------------------------------------------------------------------

top_cols = st.columns([3, 2, 2, 3])
top_cols[0].markdown(
    "<h2 style='margin:0; font-family:var(--serif);'>Subsea Welding Ops</h2>"
    "<div style='font-family:var(--mono); font-size:11px; color:var(--muted); "
    "text-transform:uppercase; letter-spacing:0.08em; margin-top:4px;'>"
    "5G/6G edge · north sea · v0.1"
    "</div>",
    unsafe_allow_html=True,
)

latency_5g = round(random.uniform(1.0, 2.0), 2)
latency_4g = round(random.uniform(40.0, 60.0), 1)
top_cols[1].metric(
    "5G URLLC latency",
    f"{latency_5g} ms",
    delta=f"vs 4G {latency_4g} ms",
    delta_color="inverse",
)

top_cols[2].metric("Tick", st.session_state.tick)

disaster_active = st.session_state.tick < st.session_state.disaster_until_tick
if top_cols[3].button(
    "Trigger Disaster (30 ticks)" if not disaster_active else "Disaster active…",
    width="stretch",
    disabled=disaster_active,
):
    st.session_state.disaster_until_tick = st.session_state.tick + 30
    disaster_active = True


# ---------------------------------------------------------------------------
# Tick loop: advance once per autorefresh
# ---------------------------------------------------------------------------

if st.session_state.running:
    st_autorefresh(interval=tick_ms, key="tick_refresh")

    st.session_state.tick += 1
    t = st.session_state.tick

    # Elapsed ticks inside the current disaster window (0 on first storm tick).
    disaster_window = 30
    disaster_start = st.session_state.disaster_until_tick - disaster_window
    disaster_elapsed = max(0, t - disaster_start) if disaster_active else 0

    for station in data_mod.STATIONS:
        tick = data_mod.make_tick(
            normal_df, storm_df, t, station,
            stress_multiplier=stress,
            disaster_active=disaster_active,
            disaster_elapsed=disaster_elapsed,
        )

        prev_weld = st.session_state.weld[station]
        weld = ml_mod.next_weld_integrity(
            prev_weld, tick["wave_height"], st.session_state.rng
        )
        st.session_state.weld[station] = weld

        lam = ml_mod.compute_lambda(
            tick["wave_height"], tick["current_velocity"], weld
        )
        R = ml_mod.reliability(lam, t_seconds=10.0)  # 10 s display horizon
        state = clf.predict(
            tick["wave_height"], tick["current_velocity"],
            tick["wave_period"], weld,
        )

        st.session_state.history[station].append(
            {
                "tick": t,
                "wave": tick["wave_height"],
                "current": tick["current_velocity"],
                "period": tick["wave_period"],
                "weld": weld,
                "lambda": lam,
                "R": R,
                "state": state,
                "source": tick["source"],
                "strange_geometry": tick.get("strange_geometry", 0),
            }
        )

        # Narrate only on state change (keeps chat readable + fast).
        prev_state = st.session_state.last_state[station]
        if state != prev_state:
            drone_line = llm_mod.narrate(
                "drone", state, tick["wave_height"], tick["current_velocity"],
                weld, R, latency_5g, use_llm=use_llm, rig=station,
            )
            buoy_line = llm_mod.narrate(
                "buoy", state, tick["wave_height"], tick["current_velocity"],
                weld, R, latency_5g, use_llm=use_llm, rig=station,
            )
            st.session_state.chat[station].append(
                {"tick": t, "state": state, "drone": drone_line, "buoy": buoy_line}
            )
            st.session_state.last_state[station] = state

        # Human-in-the-loop: raise a latched takeover request the first time
        # the data stream reports anomalous weld geometry. The flag is cleared
        # only when the operator hits "Acknowledge" on the rig card.
        if tick.get("strange_geometry", 0) == 1 and not st.session_state.takeover[station]:
            st.session_state.takeover[station] = True
            st.session_state.takeover_tick[station] = t
            drone_line = llm_mod.narrate(
                "drone", "TAKEOVER", tick["wave_height"], tick["current_velocity"],
                weld, R, latency_5g, use_llm=use_llm, rig=station,
            )
            buoy_line = llm_mod.narrate(
                "buoy", "TAKEOVER", tick["wave_height"], tick["current_velocity"],
                weld, R, latency_5g, use_llm=use_llm, rig=station,
            )
            st.session_state.chat[station].append(
                {"tick": t, "state": "TAKEOVER", "drone": drone_line, "buoy": buoy_line}
            )


# ---------------------------------------------------------------------------
# Human-in-the-loop escalation banner
# ---------------------------------------------------------------------------

pending_takeovers = [
    s for s in data_mod.STATIONS if st.session_state.takeover.get(s)
]
if pending_takeovers:
    rig_list = ", ".join(f"Rig {s}" for s in pending_takeovers)
    st.markdown(
        f"<div class='hil-banner'>"
        f"<b>Human takeover requested</b> — {rig_list} · anomalous weld geometry · "
        f"escalated over URLLC slice"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# KPI cards per rig
# ---------------------------------------------------------------------------

st.markdown("#### Live rig telemetry")
rig_cols = st.columns(2)

for col, station in zip(rig_cols, data_mod.STATIONS):
    hist = st.session_state.history[station]
    if not hist:
        continue
    last = hist[-1]
    state = last["state"]
    in_takeover = st.session_state.takeover.get(station, False)
    # TAKEOVER visually supersedes the weather-driven state on the card.
    display_state = "TAKEOVER" if in_takeover else state
    source_badge = "storm.csv" if last.get("source") == "storm" else "normal.csv"
    with col:
        takeover_badge = (
            f"<span class='badge state-TAKEOVER'>HIL · takeover · t={st.session_state.takeover_tick[station]}</span> "
            if in_takeover else ""
        )
        st.markdown(
            f"<div class='rig-card'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;'>"
            f"  <div>"
            f"    <span class='badge'>rig · {station}</span> "
            f"    <span class='badge'>data · {source_badge}</span> "
            f"    {takeover_badge}"
            f"  </div>"
            f"  <span class='state-{display_state} big-metric'>{display_state.lower()}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Reliability R", f"{last['R']:.2f}")
        k2.metric("λ", f"{last['lambda']:.3f}")
        k3.metric("Weld", f"{last['weld']:.0f}%")
        k4.metric("Wave", f"{last['wave']:.2f} m")
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
                st.rerun()


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

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


combined = _combined_df()

# Shared chart styling for the LaunchSafe palette.
_CHART_BG = "#181a1f"
_CHART_FG = "#e8e6e0"
_CHART_MUTED = "#b0b0b0"
_CHART_GRID = "rgba(255,255,255,0.06)"
_STATION_COLORS = {"A": "#c8f135", "B": "#4cc9f0"}


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
    ch1.plotly_chart(fig_wave, width="stretch")

    fig_R = go.Figure()
    for s in data_mod.STATIONS:
        sub = combined[combined.station == s]
        color = _STATION_COLORS.get(s, "#c8f135")
        fig_R.add_trace(go.Scatter(
            x=sub["tick"], y=sub["R"], mode="lines", name=f"Rig {s} · R",
            line=dict(color=color, width=2),
        ))
        fig_R.add_trace(go.Scatter(
            x=sub["tick"], y=sub["weld"] / 100.0, mode="lines",
            name=f"Rig {s} · weld",
            line=dict(color=color, width=1, dash="dot"),
            opacity=0.6,
        ))
    _style_fig(fig_R, "RELIABILITY R(t) & WELD INTEGRITY", y_range=[0, 1.05])
    ch2.plotly_chart(fig_R, width="stretch")


# ---------------------------------------------------------------------------
# Chat columns
# ---------------------------------------------------------------------------

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
                    f"<span class='badge state-{m['state']}'>t={m['tick']} · {m['state']}</span><br/>"
                    f"{m['drone']}</div>",
                    unsafe_allow_html=True,
                )
                panel.markdown(
                    f"<div class='chat-buoy{extra}'><b>buoy</b><br/>{m['buoy']}</div>",
                    unsafe_allow_html=True,
                )

st.caption(
    "PoC · Ericsson Hackathon 2026 · Decisions by scikit-learn + R=e^(-λt) · "
    "Narration by Qwen 2.5 3B (optional)"
)
