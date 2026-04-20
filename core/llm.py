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

# When a TAKEOVER is active we replace the base system prompt so the LLM's
# tone matches a formal human-in-the-loop handover instead of routine chatter.
_TAKEOVER_SYSTEM = {
    "drone": (
        "You are an autonomous welding drone that has just detected anomalous "
        "weld geometry it cannot resolve safely. "
        "Issue a ONE-sentence handover request to the human operator over the "
        "5G control slice. Max 20 words, urgent but calm, reference the rig ID "
        "and the anomaly. No preamble."
    ),
    "buoy": (
        "You are a 5G edge buoy escalating a welding anomaly to the on-shore "
        "operations centre. "
        "Announce the human-in-the-loop takeover in ONE sentence, max 20 words, "
        "dispatcher tone, mention URLLC slice priority. No preamble."
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
        "NORMAL":   "[fallback] drone nominal · wave {wave:.1f} m · R={R:.2f}",
        "CAUTION":  "[fallback] drone caution · wave {wave:.1f} m · R={R:.2f}",
        "ABORT":    "[fallback] drone abort · wave {wave:.1f} m · R={R:.2f}",
        "TAKEOVER": "[fallback] Rig {rig}: anomalous weld geometry — requesting human takeover.",
    },
    "buoy": {
        "NORMAL":   "[fallback] link {lat:.1f} ms · wave {wave:.1f} m · current {cur:.1f} m/s",
        "CAUTION":  "[fallback] advisory · wave {wave:.1f} m · current {cur:.1f} m/s",
        "ABORT":    "[fallback] emergency · wave {wave:.1f} m · current {cur:.1f} m/s",
        "TAKEOVER": "[fallback] Rig {rig}: escalating to URLLC takeover slice · operator handover.",
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

    is_takeover = state == "TAKEOVER"
    system_prompt = (_TAKEOVER_SYSTEM if is_takeover else _ROLE_SYSTEM)[role]

    if is_takeover:
        user = (
            f"Rig {ctx['rig']}: anomalous weld geometry detected. "
            f"Telemetry — wave={ctx['wave']:.1f}m, current={ctx['cur']:.1f}m/s, "
            f"R={ctx['R']:.2f}, 5G_latency={ctx['lat']:.1f}ms. "
            "Issue the handover."
        )
    else:
        user = (
            f"State={state}. Wave={ctx['wave']:.1f}m, current={ctx['cur']:.1f}m/s, "
            f"R={ctx['R']:.2f}, 5G_latency={ctx['lat']:.1f}ms. "
            "Explain your next action."
        )

    try:
        resp = ollama.chat(  
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
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
    R: float,
    latency_ms: float,
    use_llm: bool = True,
    rig: str = "",
) -> str:
    """One-sentence narration for the chat log.

    ``state`` accepts ``NORMAL`` / ``CAUTION`` / ``ABORT`` (from the classifier)
    and the special ``TAKEOVER`` pseudo-state, which is triggered when the
    ``strange_geometry`` anomaly flag is raised and requires a human-in-the-loop
    handover. ``use_llm=False`` forces the template path.
    """

    ctx = {
        "wave": float(wave),
        "cur": float(current),
        "R": float(R),
        "lat": float(latency_ms),
        "rig": rig,
    }

    if use_llm:
        out = _try_ollama(role, state, ctx)
        if out:
            return out
    return _template(role, state, ctx)
