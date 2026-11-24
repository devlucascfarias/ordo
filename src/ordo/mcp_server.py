"""Servidor MCP (stdio) do Ordo: as mesmas ferramentas do CLI como ferramentas nativas de agentes.

    ordo mcp              # todas as ferramentas
    ordo mcp --no-run     # sem execução de comandos

Cada ferramenta chama o comando equivalente do CLI com --json via `cli.invoke` (mesma lógica, sem
segunda implementação) e devolve o envelope estruturado. Chamadas são serializadas: o modelo local
atende uma de cada vez e a captura de stderr é global.
"""
from __future__ import annotations

import json
import os
import shlex
import tempfile
import threading
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer

from . import __version__
from .cli import invoke

_LOCK = threading.Lock()
_SHELL_META = ("|", "&&", "||", ";", ">", "<", "$(", "`")

INSTRUCTIONS = """Ordo filters noisy tool output for you. Prefer ordo_run over running tests, builds or linters
directly when the output may be long: it returns the failure cause (file, line, symbol, error) as structured
data plus a raw_id. If the summary is not enough, ordo_show(raw_id, grep=...) returns the exact original lines.
Warnings starting with 'unverified:' mark citations not found in the input; check them with ordo_show.
Record failed approaches with ordo_dead_end and facts with ordo_note; ordo_compact returns the session state."""


def _call(argv: list[str]) -> dict[str, Any]:
    with _LOCK:
        code, out, err = invoke(argv)
    out = out.strip()
    try:
        data = json.loads(out) if out.startswith("{") else None
    except json.JSONDecodeError:
        data = None
    if data is None:
        data = {"text": out}
    data.setdefault("exit", code)
    if err.strip():
        data["stderr"] = err.strip()[-2000:]
    if code not in (0, None) and not out:
        data["error"] = err.strip() or f"exit {code}"
    return data


async def _acall(argv: list[str]) -> dict[str, Any]:
    return await anyio.to_thread.run_sync(_call, argv)


def _command_argv(command: str) -> list[str]:
    """String do agente -> argv. Com pipes/redirecionamentos, roda pelo shell do sistema."""
    if any(m in command for m in _SHELL_META):
        return ["cmd", "/d", "/s", "/c", command] if os.name == "nt" else ["sh", "-c", command]
    if os.name != "nt":
        return shlex.split(command)
    # no Windows o shlex não-POSIX mantém as aspas: "C:\Program Files\x.exe" precisa sair sem elas
    return [t[1:-1] if len(t) > 1 and t[0] == t[-1] and t[0] in "\"'" else t for t in shlex.split(command, posix=False)]


def build_server(allow_run: bool = True) -> MCPServer:
    server = MCPServer(name="ordo", version=__version__, instructions=INSTRUCTIONS)

    if allow_run:
        @server.tool(description=(
            "Run a test/build/lint command and get its output filtered to what matters for `intent`: status, "
            "causal trace (file, line, symbol, error) and exact snippets, as structured data. Use instead of "
            "running the command directly when output may be long. `exit` is the command's own exit code. "
            "Short output is returned unchanged in `text`. Keep `raw_id` to fetch exact lines with ordo_show."))
        async def ordo_run(command: str, intent: str, cwd: str | None = None,
                           timeout_seconds: float | None = None) -> dict[str, Any]:
            argv = ["run", "--json", "-i", intent]
            if cwd:
                argv += ["--cwd", cwd]
            if timeout_seconds:
                argv += ["--cmd-timeout", str(timeout_seconds)]
            return await _acall(argv + ["--"] + _command_argv(command))

    @server.tool(description=(
        "Filter text you already have (a long log, a CI output, a pasted error) for `intent`. mode: TOOL_OUTPUT "
        "(default) or CODE_SLICE (treat the text as a source file). Returns the same structure as ordo_run."))
    async def ordo_filter(text: str, intent: str, mode: str = "TOOL_OUTPUT") -> dict[str, Any]:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
            f.write(text)
            path = f.name
        try:
            return await _acall(["filter", "--json", "--mode", mode, "-i", intent, path])
        finally:
            os.unlink(path)

    @server.tool(description=(
        "Locate code in a large source file: returns the relevant symbols with exact 1-based line ranges and a short "
        "snippet. `intent` can name the symbol or describe the behavior. Cheaper than reading the whole file."))
    async def ordo_slice(path: str, intent: str) -> dict[str, Any]:
        return await _acall(["slice", "--json", "-i", intent, path])

    @server.tool(description=(
        "Search the repository with a regex (ripgrep or built-in) and get the matches filtered for `intent` "
        "(definitions vs. callers, the relevant ones only)."))
    async def ordo_search(pattern: str, intent: str, paths: list[str] | None = None, context: int = 2) -> dict[str, Any]:
        return await _acall(["search", "--json", "-C", str(context), "-i", intent, pattern, *(paths or [])])

    @server.tool(description=(
        "Print exact lines of a saved ordo_run/ordo_filter/ordo_search output (nothing is lost by filtering). "
        "raw_id defaults to the last one. Use grep (regex) or lines ('120-160') or tail to select."))
    async def ordo_show(raw_id: str = "last", grep: str | None = None, lines: str | None = None,
                        tail: int | None = None, context: int = 2) -> dict[str, Any]:
        argv = ["show", raw_id, "-C", str(context)]
        if grep:
            argv += ["--grep", grep]
        if lines:
            argv += ["--lines", lines]
        if tail:
            argv += ["--tail", str(tail)]
        return await _acall(argv)

    @server.tool(description="Record an approach that failed and why, so it is not tried again in this session.")
    async def ordo_dead_end(attempt: str, reason: str) -> dict[str, Any]:
        return await _acall(["dead-end", attempt, reason])

    @server.tool(description="Record a confirmed fact (with path:line when possible) for the session ledger.")
    async def ordo_note(text: str) -> dict[str, Any]:
        return await _acall(["note", text])

    @server.tool(description=(
        "Return the session state ledger: goal, confirmed facts, dead ends not to repeat, modified files with their "
        "last test status. Use before a long task, after many steps, or when resuming work."))
    async def ordo_compact(goal: str) -> dict[str, Any]:
        return await _acall(["compact", "-i", goal])

    @server.tool(description="Token savings of this session (calls, input vs. output tokens, routes).")
    async def ordo_stats() -> dict[str, Any]:
        return await _acall(["stats", "--json"])

    return server


def serve(allow_run: bool = True) -> None:
    build_server(allow_run).run("stdio")
