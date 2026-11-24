"""Model clients for Ordo 3B: Ollama (default) or any OpenAI-compatible server.

The OpenAI-compatible backend covers llama.cpp (`llama-server`), LM Studio, vLLM, LocalAI and Ollama's
own `/v1` endpoint. It is selected with `ORDO_BASE_URL` (or `--base-url`).
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import httpx

from .contract import GcMode, build_messages

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("ORDO_MODEL", "ordo:3b")
DEFAULT_BASE_URL = os.environ.get("ORDO_BASE_URL", "")
# names the published model is usually installed under (local Modelfile or the Hugging Face quickstart)
KNOWN_NAMES = ("ordo:3b", "hf.co/devlucascfarias/ordo-3b")


@dataclass
class OrdoResult:
    text: str
    prompt_tokens: int
    output_tokens: int
    seconds: float
    truncated: bool = False  # stopped at the num_predict limit
    model: str = ""


def normalize_host(host: str) -> str:
    """Accept the server-style OLLAMA_HOST format: `0.0.0.0`, `0.0.0.0:11434`, `:11434`, `host`, URL."""
    h = (host or "").strip().rstrip("/")
    scheme = "http"
    if "://" in h:
        scheme, h = h.split("://", 1)
    if h.startswith("["):  # bracketed IPv6, with or without a port
        name, _, port = h.partition("]")
        name, port = name + "]", port.lstrip(":")
    elif h.count(":") == 1:
        name, port = h.split(":")
    else:
        name, port = h, ""
    if name in ("", "0.0.0.0", "*"):
        name = "127.0.0.1"
    elif name == "[::]":
        name = "[::1]"
    return f"{scheme}://{name}:{port or '11434'}"


def installed_models(host: str, timeout: float = 5.0) -> list[str]:
    r = httpx.get(f"{normalize_host(host)}/api/tags", timeout=timeout)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def find_ordo_model(host: str) -> str | None:
    """Look for Ordo among the installed models, in case it was installed under another name."""
    try:
        names = installed_models(host)
    except (httpx.HTTPError, ValueError):
        return None
    return match_ordo(names)


def match_ordo(names: list[str]) -> str | None:
    """Pick the Ordo model from a list of served or installed model names."""
    for known in KNOWN_NAMES:  # "hf.co/devlucascfarias/ordo-3b" matches ".../ordo-3b:latest"
        for n in names:
            if n == known or (":" not in known and n.rsplit(":", 1)[0] == known):
                return n
    return next((n for n in names if "ordo" in n.lower()), None)


_resolved: dict[tuple[str, str], str] = {}


def call_ollama_ordo(
    mode: GcMode | str,
    intent: str,
    raw_input: str,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    num_ctx: int = 16384,
    num_predict: int = 1024,  # the Modelfile uses 512, which cuts long CODE_SLICE snippets in half
    timeout: float = 300.0,
) -> OrdoResult:
    base = normalize_host(host)
    name = _resolved.get((base, model), model)
    payload = {
        "model": name,
        "messages": build_messages(mode, intent, raw_input),
        "stream": False,
        "keep_alive": os.environ.get("ORDO_KEEP_ALIVE", "30m"),  # avoid reloading the model on every call
        "options": {"temperature": 0.0, "top_p": 0.9, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    r = httpx.post(f"{base}/api/chat", json=payload, timeout=timeout)
    if r.status_code == 404 and name == model:  # model not installed under this name
        alt = find_ordo_model(base)
        if alt and alt != model:
            print(f"ordo: model {model!r} not found; using {alt!r} (set ORDO_MODEL={alt} to silence)", file=sys.stderr)
            _resolved[(base, model)] = alt
            payload["model"] = name = alt
            r = httpx.post(f"{base}/api/chat", json=payload, timeout=timeout)
        else:
            raise RuntimeError(f"model {model!r} is not installed in Ollama; run `ollama pull "
                               f"hf.co/devlucascfarias/ordo-3b` or set ORDO_MODEL")
    r.raise_for_status()
    data = r.json()
    return OrdoResult(
        text=data["message"]["content"],
        prompt_tokens=data.get("prompt_eval_count", 0),
        output_tokens=data.get("eval_count", 0),
        seconds=data.get("total_duration", 0) / 1e9,
        truncated=data.get("done_reason") == "length",
        model=name,
    )


# ---------------------------------------------------------------- OpenAI-compatible servers


def normalize_base_url(url: str) -> str:
    """`localhost:8080`, `http://h:8080` or `http://h:8080/v1` -> `http://h:8080/v1`."""
    u = url.strip().rstrip("/")
    if "://" not in u:
        u = f"http://{u}"
    if urlsplit(u).path in ("", "/"):
        u += "/v1"
    return u


def _headers() -> dict[str, str]:
    key = os.environ.get("ORDO_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def served_models(base_url: str, timeout: float = 5.0) -> list[str]:
    r = httpx.get(f"{normalize_base_url(base_url)}/models", headers=_headers(), timeout=timeout)
    r.raise_for_status()
    return [m["id"] for m in r.json().get("data", []) if "id" in m]


def _resolve_served(base: str, model: str) -> str:
    """The name to send: as configured if served; else the Ordo model; else the only model served."""
    if (base, model) in _resolved:
        return _resolved[(base, model)]
    try:
        names = served_models(base)
    except (httpx.HTTPError, ValueError, KeyError):
        names = []  # some servers do not list models; send the configured name as is
    name = model
    if names and model not in names:
        alt = match_ordo(names) or (names[0] if len(names) == 1 else None)
        if alt is None:
            raise RuntimeError(f"model {model!r} is not served at {base}; served: {', '.join(names[:8])}. "
                               f"Set ORDO_MODEL to one of them")
        if model != DEFAULT_MODEL or os.environ.get("ORDO_MODEL"):
            print(f"ordo: model {model!r} not served; using {alt!r} (set ORDO_MODEL={alt} to silence)",
                  file=sys.stderr)
        name = alt
    _resolved[(base, model)] = name
    return name


def call_openai_ordo(
    mode: GcMode | str,
    intent: str,
    raw_input: str,
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    num_predict: int = 1024,
    timeout: float = 300.0,
) -> OrdoResult:
    """Chat completion on an OpenAI-compatible server. The context size is the server's setting
    (for llama.cpp: `llama-server -c 16384`)."""
    base = normalize_base_url(base_url)
    name = _resolve_served(base, model)
    payload = {
        "model": name,
        "messages": build_messages(mode, intent, raw_input),
        "stream": False,
        "temperature": 0.0,
        "top_p": 0.9,
        "max_tokens": num_predict,
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{base}/chat/completions", json=payload, headers=_headers(), timeout=timeout)
    r.raise_for_status()
    data = r.json()
    choice = data["choices"][0]
    usage = data.get("usage") or {}
    return OrdoResult(
        text=choice["message"]["content"] or "",
        prompt_tokens=usage.get("prompt_tokens", 0),
        output_tokens=usage.get("completion_tokens", 0),
        seconds=time.perf_counter() - t0,
        truncated=choice.get("finish_reason") == "length",
        model=name,
    )


def call_ordo(
    mode: GcMode | str,
    intent: str,
    raw_input: str,
    *,
    model: str = DEFAULT_MODEL,
    host: str = DEFAULT_HOST,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 300.0,
) -> OrdoResult:
    """Call Ordo on the configured backend: an OpenAI-compatible server when `base_url` is set, else Ollama."""
    if base_url:
        return call_openai_ordo(mode, intent, raw_input, model=model, base_url=base_url, timeout=timeout)
    return call_ollama_ordo(mode, intent, raw_input, model=model, host=host, timeout=timeout)
