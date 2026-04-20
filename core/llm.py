"""Short narrations for the chat log.

The ML layer in ``ml.py`` makes every decision. This module only turns a
decision into one human-readable sentence so the dashboard's chat column
feels like "agents are talking".

We try a local Qwen 2.5 3B model through Ollama first (fast on an M3, no
network needed). If Ollama is not running, not installed, or too slow,
we fall back to deterministic templates so the demo never stalls on stage.
"""

from __future__ import annotations

from typing import Literal

Role = Literal["drone", "buoy"]

OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_TIMEOUT_S = 3.0

_ROLE_SYSTEM = {
    "drone": (
        "You are an autonomous underwater welding drone on a North Sea oil rig, "
        "tethered by fibre to a 5G edge buoy. "
        "Reply in ONE sentence, max 18 words, crisp technical radio tone. "
        "No preamble, no pleasantries. Reference the telemetry numbers."
    ),
    "buoy": (
        "You are a 5G edge buoy coordinating a welding drone on the rig beneath you. "
        "Reply in ONE sentence, max 18 words, dispatcher tone. "
        "No preamble, no pleasantries. Mention sea state or network slice priority."
    ),
}



# ---------------------------------------------------------------------------
# Minimal fallback templates.
#
# Used only when the user toggles the LLM off, or when Ollama is unreachable.
# Kept deliberately terse and uniform so the audience can tell real LLM output
# apart from the fallback (the "[fallback]" prefix is intentional).
# ---------------------------------------------------------------------------

_TEMPLATES: dict[Role, dict[str, str]] = {
    "drone": {
        "NORMAL":  "[fallback] drone nominal · weld {weld:.0f}% · R={R:.2f}",
        "CAUTION": "[fallback] drone caution · weld {weld:.0f}% · R={R:.2f}",
        "ABORT":   "[fallback] drone abort · weld {weld:.0f}% · R={R:.2f}",
    },
    "buoy": {
        "NORMAL":  "[fallback] link {lat:.1f} ms · wave {wave:.1f} m · current {cur:.1f} m/s",
        "CAUTION": "[fallback] advisory · wave {wave:.1f} m · current {cur:.1f} m/s",
        "ABORT":   "[fallback] emergency · wave {wave:.1f} m · current {cur:.1f} m/s",
    },
}


def _template(role: Role, state: str, ctx: dict) -> str:
    return _TEMPLATES[role].get(state, _TEMPLATES[role]["NORMAL"]).format(**ctx)



# Ollama path



def _try_ollama(role: Role, state: str, ctx: dict) -> str | None:
    """Return a model-generated sentence, or None on any failure."""

    try:
        import ollama  
    except Exception:
        return None

    user = (
        f"State={state}. Wave={ctx['wave']:.1f}m, current={ctx['cur']:.1f}m/s, "
        f"weld={ctx['weld']:.0f}%, R={ctx['R']:.2f}, 5G_latency={ctx['lat']:.1f}ms. "
        "Explain your next action."
    )

    try:
        resp = ollama.chat(  
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": _ROLE_SYSTEM[role]},
                {"role": "user", "content": user},
            ],
            options={"num_predict": 60, "temperature": 0.5},
        )
    except Exception:
        return None

    text = (resp.get("message", {}) or {}).get("content", "").strip()
    if not text:
        return None
    return text.splitlines()[0].strip()



# Public API

def narrate(
    role: Role,
    state: str,
    wave: float,
    current: float,
    weld: float,
    R: float,
    latency_ms: float,
    use_llm: bool = True,
) -> str:
    """One-sentence narration for the chat log.

    ``use_llm=False`` forces the template path (handy for offline demos or to
    cut latency when ticking fast).
    """

    ctx = {
        "wave": float(wave),
        "cur": float(current),
        "weld": float(weld),
        "R": float(R),
        "lat": float(latency_ms),
    }

    if use_llm:
        out = _try_ollama(role, state, ctx)
        if out:
            return out
    return _template(role, state, ctx)
