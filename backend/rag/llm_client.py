"""
Thin client for a local Ollama server (OpenAI-ish REST API).

Ollama's /api/generate endpoint can take a long time to return on the very
first call after the server starts, or after the model has been evicted
from memory once its keep_alive window expires, because it has to load the
full model weights into RAM before it can generate a single token. On a
CPU-only machine (no GPU) this routinely takes well past a short timeout —
that's what produces a "Read timed out" error even though Ollama is running
completely fine.

This module fixes that by:

  1. Splitting the timeout into a short CONNECT timeout (is the server even
     there? should be near-instant for a local process) and a long READ
     timeout (generation, including a cold model load, can be slow).
  2. Checking /api/tags before generating, so we can tell the difference
     between "Ollama isn't running" (fix: start Ollama), "the model isn't
     pulled" (fix: `ollama pull <model>`), and "Ollama is just slow to
     respond" (fix: nothing, just wait) instead of always blaming the model.
  3. Passing `keep_alive` on every request so Ollama keeps the model resident
     in memory afterward instead of unloading it and paying the load cost
     again on the very next question.
  4. Reusing one requests.Session for connection pooling across calls.
"""
import json
import os
from typing import Iterator, Optional

import requests

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "codellama:7b")

# How long Ollama keeps the model loaded in memory after a request, so a slow
# "cold start" load only has to happen once in a while instead of on every
# question. Accepts a duration string like "30m", "1h", or "-1" (forever).
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "30m")

# connect_timeout: how long to wait for a TCP connection to Ollama itself.
#   Ollama is a local process, so if it's actually up this should be
#   near-instant — keeping this short lets us fail fast and report an
#   accurate "Ollama isn't running" message instead of hanging.
# read_timeout: how long to wait for a response once connected. This has to
#   cover a worst-case cold model load PLUS generation time, so it's
#   intentionally generous. Bump OLLAMA_READ_TIMEOUT even higher if a 7B
#   model on your hardware needs longer than 600s.
CONNECT_TIMEOUT = float(os.environ.get("OLLAMA_CONNECT_TIMEOUT", "5"))
READ_TIMEOUT = float(os.environ.get("OLLAMA_READ_TIMEOUT", "600"))
HEALTH_CHECK_TIMEOUT = (CONNECT_TIMEOUT, 10)  # /api/tags should always be fast

_session: Optional[requests.Session] = None


def _get_session() -> requests.Session:
    """One reused, connection-pooled session instead of a new connection per call."""
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


# ---------------------------------------------------------------------------
# Health / model-availability checks
# ---------------------------------------------------------------------------

def check_health() -> bool:
    """True if the Ollama server itself is reachable (says nothing about the model)."""
    try:
        resp = _get_session().get(f"{OLLAMA_BASE_URL}/api/tags", timeout=HEALTH_CHECK_TIMEOUT)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def list_available_models() -> list[str]:
    """Names of models Ollama currently has pulled. Empty list if unreachable."""
    try:
        resp = _get_session().get(f"{OLLAMA_BASE_URL}/api/tags", timeout=HEALTH_CHECK_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name", "") for m in data.get("models", [])]
    except (requests.RequestException, ValueError):
        return []


def check_model_available(model: str = None) -> bool:
    """True if `model` (default: OLLAMA_MODEL) is pulled. Tolerant of a differing ':tag' suffix."""
    model = model or OLLAMA_MODEL
    available = list_available_models()
    if model in available:
        return True
    # Ollama tags look like "codellama:7b" — also accept a bare-name match in
    # case the pulled tag differs (e.g. "codellama:latest" vs "codellama:7b").
    base_name = model.split(":")[0]
    return any(a.split(":")[0] == base_name for a in available)


def get_status() -> dict:
    """Full diagnostic snapshot, used by /health and the Streamlit sidebar."""
    reachable = check_health()
    models = list_available_models() if reachable else []
    model_available = check_model_available(OLLAMA_MODEL) if reachable else False
    return {
        "ollama_reachable": reachable,
        "model": OLLAMA_MODEL,
        "model_available": model_available,
        "available_models": models,
    }


def _unreachable_message() -> str:
    return (
        f"⚠️ Can't reach Ollama at {OLLAMA_BASE_URL}. "
        "Make sure it's running — start it with `ollama serve`. "
        "If you saw \"bind: Only one usage of each socket address...\" when trying that, "
        "Ollama is already running as a background service — you do NOT need to run "
        "`ollama serve` again. That error instead means something else is blocking the "
        "connection (a firewall, antivirus, or the service is still starting up) — check "
        "Task Manager for an `ollama.exe` / `ollama app.exe` process, or run `ollama list` "
        "in a terminal to confirm it responds."
    )


def _model_missing_message(model: str, available: list[str]) -> str:
    available_str = ", ".join(available) if available else "(none)"
    return (
        f"⚠️ Ollama is running, but the model '{model}' isn't pulled yet. Run:\n\n"
        f"    ollama pull {model}\n\n"
        f"Currently available models: {available_str}"
    )


def _slow_response_message(read_timeout: float) -> str:
    return (
        f"⏳ Ollama is reachable but took longer than {int(read_timeout)}s to respond. "
        "This is normal on the *first* request after Ollama starts (or after the model "
        "was idle and got unloaded) — loading a 7B model into memory on CPU can take a "
        "few minutes. Please try your question again; it should be much faster now that "
        "the model is warm. If it keeps timing out, run `ollama ps` to confirm the model "
        "is actually loading, or set OLLAMA_READ_TIMEOUT to a higher value."
    )


def _preflight(model: str) -> Optional[str]:
    """Run before every generate call. Returns an error message, or None if OK to proceed."""
    if not check_health():
        return _unreachable_message()
    if not check_model_available(model):
        return _model_missing_message(model, list_available_models())
    return None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(prompt: str, system: str = "", temperature: float = 0.2, model: str = None) -> str:
    """Non-streaming generation, with a pre-flight check and specific, actionable error messages."""
    model = model or OLLAMA_MODEL

    preflight_error = _preflight(model)
    if preflight_error:
        return preflight_error

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature},
    }
    try:
        resp = _get_session().post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectTimeout:
        return _unreachable_message()
    except requests.exceptions.ReadTimeout:
        return _slow_response_message(READ_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return _unreachable_message()
    except requests.exceptions.HTTPError as e:
        return f"⚠️ Ollama returned an error: {e}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ Unexpected error talking to Ollama: {e}"

    try:
        data = resp.json()
    except ValueError:
        return "⚠️ Ollama returned a response that wasn't valid JSON. Please try again."

    text = (data.get("response") or "").strip()
    if not text:
        return "⚠️ Ollama returned an empty response. Try rephrasing your question, or try again."
    return text


def generate_stream(prompt: str, system: str = "", temperature: float = 0.2, model: str = None) -> Iterator[str]:
    """Streaming generation. Yields tokens as they arrive; yields one clear message and stops on failure."""
    model = model or OLLAMA_MODEL

    preflight_error = _preflight(model)
    if preflight_error:
        yield preflight_error
        return

    payload = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": True,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": temperature},
    }

    got_any_token = False
    try:
        with _get_session().post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
            stream=True,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue  # skip a malformed line rather than failing the whole stream
                token = chunk.get("response", "")
                if token:
                    got_any_token = True
                    yield token
                if chunk.get("done"):
                    break
            if not got_any_token:
                yield "⚠️ Ollama returned an empty response. Try rephrasing your question, or try again."
    except requests.exceptions.ConnectTimeout:
        yield _unreachable_message()
    except requests.exceptions.ReadTimeout:
        if got_any_token:
            yield (
                "\n\n⚠️ [Connection to Ollama timed out partway through — the answer above is "
                f"what was generated before the timeout. Increase OLLAMA_READ_TIMEOUT (currently "
                f"{int(READ_TIMEOUT)}s) if this keeps happening.]"
            )
        else:
            yield _slow_response_message(READ_TIMEOUT)
    except requests.exceptions.ConnectionError:
        yield _unreachable_message()
    except requests.exceptions.HTTPError as e:
        yield f"⚠️ Ollama returned an error: {e}"
    except requests.exceptions.RequestException as e:
        yield f"⚠️ Unexpected error talking to Ollama: {e}"


def warm_up_model(model: str = None) -> None:
    """
    Fire a trivial generate call to force Ollama to load the model into memory
    ahead of the user's first real question. Best-effort — failures are
    swallowed since this is a background optimization, not a correctness
    requirement (generate()/generate_stream() still run their own checks
    regardless). Intended to be called from a background thread at app
    startup so it doesn't block the server while the model loads.
    """
    model = model or OLLAMA_MODEL
    if not check_health():
        return
    try:
        _get_session().post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": model, "prompt": "", "stream": False, "keep_alive": OLLAMA_KEEP_ALIVE},
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
        )
    except requests.RequestException:
        pass