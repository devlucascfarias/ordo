"""MCP server: tools called directly, and a real MCP client talking to `ordo mcp` over stdio."""
import json
import shutil
import sys

import anyio
import pytest

pytest.importorskip("mcp")

from ordo import cli  # noqa: E402
from ordo.mcp_server import _command_argv, build_server  # noqa: E402
from test_cli import FakeModel, model, ordo_home  # noqa: E402,F401


def _names(server) -> set[str]:
    return {t.name for t in anyio.run(server.list_tools)}


def test_tools_listed_and_no_run_mode():
    assert {"ordo_run", "ordo_filter", "ordo_slice", "ordo_search", "ordo_show", "ordo_dead_end", "ordo_note",
            "ordo_compact", "ordo_stats"} <= _names(build_server())
    assert "ordo_run" not in _names(build_server(allow_run=False))


def _call(server, name, args):
    res = anyio.run(server.call_tool, name, args)  # SDK 2.x: CallToolResult
    if getattr(res, "structured_content", None):
        sc = res.structured_content
        return sc.get("result", sc) if isinstance(sc, dict) and set(sc) == {"result"} else sc
    return json.loads(res.content[0].text)


def test_run_tool_returns_envelope_and_show_recovers_original(model):
    srv = build_server()
    env = _call(srv, "ordo_run", {"command": f'"{sys.executable}" -c "print(42)"', "intent": "x"})
    assert env["exit"] == 0 and env["route"] == "bypass" and env["text"].strip() == "42" and env["raw_id"]
    shown = _call(srv, "ordo_show", {"raw_id": env["raw_id"]})
    assert "1 | 42" in shown["text"] and not model.calls


def test_filter_tool_calls_model(model):
    model.reply = "status: FAIL\nsummary: boom at a.py:1\ncausal_trace:\n- file: a.py\n  lines: '1'\n  issue: boom"
    log = "\n".join(f"line {i}" for i in range(60)) + "\nTraceback\n  File \"a.py\", line 1\nValueError: boom"
    env = _call(build_server(), "ordo_filter", {"text": log, "intent": "why?"})
    assert env["route"] == "model" and env["result"]["status"] == "FAIL" and model.calls


def test_errors_come_back_as_data(model):
    env = _call(build_server(), "ordo_slice", {"path": "does/not/exist.py", "intent": "x"})
    assert env["exit"] == 2 and "not a file" in env["error"]


def test_command_argv_uses_shell_only_for_metacharacters():
    assert _command_argv("pytest -q tests") == ["pytest", "-q", "tests"]
    assert _command_argv("npm test | tail -5")[-1] == "npm test | tail -5"


def test_invoke_never_writes_to_real_stdout(capsys, model):
    code, out, err = cli.invoke(["guide"])
    assert code == 0 and out.startswith("## Ordo") and capsys.readouterr().out == ""


@pytest.mark.skipif(not shutil.which("ordo"), reason="ordo is not installed on PATH")
def test_real_mcp_client_over_stdio(tmp_path):
    from mcp.client.session import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    async def scenario():
        params = StdioServerParameters(command=sys.executable, args=["-m", "ordo", "mcp"],
                                       env={**__import__("os").environ, "ORDO_HOME": str(tmp_path)})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = {t.name for t in (await session.list_tools()).tools}
                res = await session.call_tool("ordo_dead_end", {"attempt": "raised timeout", "reason": "still hangs"})
                led = await session.call_tool("ordo_compact", {"goal": "fix the hang"})
                return tools, res, led

    tools, res, led = anyio.run(scenario)
    assert "ordo_run" in tools and not res.is_error
    text = "".join(getattr(c, "text", "") for c in led.content)
    assert "raised timeout" in text and "fix the hang" in text
