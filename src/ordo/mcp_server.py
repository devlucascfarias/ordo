"""Ordo MCP server (stdio): the same tools as the CLI, exposed as native agent tools.

    ordo mcp              # all tools
    ordo mcp --no-run     # no command execution

Each tool calls the matching CLI command with --json through `cli.invoke` (same logic, no second
implementation) and returns the structured envelope. Calls are serialized: the local model serves
one request at a time and stderr capture is global.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import tempfile
import threading
from typing import Any

import anyio
from mcp.server.mcpserver import MCPServer

from . import __version__
from .cli import invoke

_LOCK = threading.Lock()
_SHELL_META = ("|", "&", ";", ">", "<", "$(", "`")

INSTRUCTIONS = """Ordo condenses tool output into causal data (status, file, line, symbol, error) with a local model.
- Use ordo_run for tests, builds and linters whose output may be long; `exit` is the command's own exit code.
  Short output is returned unchanged in `text` (route "bypass").
- `result` holds the parsed output (route "model"). Route "fallback" means the model was unavailable and
  `text` is the original output, windowed around errors: read it as a normal log.
- Warnings starting with "unverified:" mark citations not found in the input; confirm them with ordo_show
  before acting on them.
- Every run, filter and search returns a raw_id; ordo_show(raw_id, grep=..., lines=...) returns the exact
  original lines. Prefer it to re-running a command.
- Record failed approaches with ordo_dead_end and verified facts with ordo_note. Keep the memory true:
  ordo_memory lists items with stable ids, ordo_edit corrects one, ordo_forget retracts one. Read ordo_compact before a
  long task or when resuming work, and do not repeat any listed dead end."""


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


def _with_id(data: dict[str, Any]) -> dict[str, Any]:
    """note/dead-end print the new id in text; expose it as a field so the agent can pass it to ordo_edit."""
    m = re.search(r"\b(?:note|dead-end) ([nd][0-9a-f]{6})\b", data.get("text", ""))
    if m:
        data["id"] = m.group(1)
    return data


def _command_argv(command: str) -> list[str]:
    """Agent command string -> argv. With pipes or redirections, run it through the system shell."""
    if any(m in command for m in _SHELL_META):
        return ["cmd", "/d", "/s", "/c", command] if os.name == "nt" else ["sh", "-c", command]
    if os.name != "nt":
        return shlex.split(command)
    # on Windows, non-POSIX shlex keeps the quotes: "C:\Program Files\x.exe" must come out without them
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
        return _with_id(await _acall(["dead-end", attempt, reason]))

    @server.tool(description="Record a confirmed fact (with path:line when possible) for the session ledger.")
    async def ordo_note(text: str) -> dict[str, Any]:
        return _with_id(await _acall(["note", text]))

    @server.tool(description=(
        "List the session memory: notes and dead ends with stable ids (n..., d...). `grep` filters by regex. "
        "Use it to find what to edit or retract in a long-running project."))
    async def ordo_memory(grep: str | None = None) -> dict[str, Any]:
        return await _acall(["memory", "--json"] + (["--grep", grep] if grep else []))

    @server.tool(description=(
        "Rewrite a memory item, keeping its id: `text` replaces a note, or the attempt of a dead end; "
        "`reason` replaces the reason of a dead end. Ids come from ordo_memory."))
    async def ordo_edit(id: str, text: str | None = None, reason: str | None = None) -> dict[str, Any]:
        return await _acall(["edit", id] + ([text] if text else []) + (["--reason", reason] if reason else []))

    @server.tool(description=(
        "Retract a note or dead end that is wrong or obsolete. `item` is an id from ordo_memory, or text "
        "contained in exactly one item."))
    async def ordo_forget(item: str) -> dict[str, Any]:
        return await _acall(["forget", item])

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
