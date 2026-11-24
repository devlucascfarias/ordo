"""Ollama client for Ordo 3B."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import httpx

from .contract import GcMode, build_messages

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
DEFAULT_MODEL = os.environ.get("ORDO_MODEL", "ordo:3b")
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
