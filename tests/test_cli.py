import json
import re
import sys
from pathlib import Path

import pytest

from ordo import cli
from ordo.client import OrdoResult
from ordo.session import Session
from ordo.textproc import clean, est_tokens, intent_terms, window_file, window_log

FIX = Path(__file__).resolve().parent / "fixtures"
SMOKE_LOG = (FIX / "smoke_pytest_log.txt").read_text(encoding="utf-8")
SMOKE_OUT = (FIX / "smoke_pytest_ordo3b.yaml").read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def ordo_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORDO_SESSION", "t")
    return tmp_path


class FakeModel:
    def __init__(self, reply="status: PASS\nsummary: ok"):
        self.reply, self.calls = reply, []

    def __call__(self, mode, intent, raw, **kw):
        self.calls.append({"mode": str(getattr(mode, "value", mode)), "intent": intent, "raw": raw})
        if isinstance(self.reply, Exception):
            raise self.reply
        return OrdoResult(text=self.reply, prompt_tokens=est_tokens(raw), output_tokens=20, seconds=0.5)


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(cli, "call_ollama_ordo", fake)
    return fake


def run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


# ---------------------------------------------------------------- textproc


def test_clean_ansi_and_progress():
    assert clean("\x1b[31mFAIL\x1b[0m x\r\n 10%\r 50%\r100%\n") == "FAIL x\n100%"


def test_window_log_keeps_errors_head_and_tail():
    lines = [f"ok line {i} " + "x" * 60 for i in range(3000)]
    lines[1500] = "E   KeyError: 'BRL' at src/fx.py:12"
    text, cut = window_log("\n".join(lines), budget=1500)
    assert cut and "KeyError: 'BRL'" in text and "ok line 2999" in text and "ok line 0 " in text
    assert "lines omitted" in text and est_tokens(text) < 1800


def test_window_log_noop_when_small():
    assert window_log("a\nb", 100) == ("a\nb", False)


def test_intent_terms():
    strong, weak = intent_terms("Find where repair_yaml_output is defined in `schema.py` callers of callOllama")
    assert "repair_yaml_output" in strong and "schema.py" in strong and "callOllama" in strong


def test_window_file_finds_dotted_name_by_parts():
    body = [f"x{i} = {i}  # filler filler filler filler filler" for i in range(3000)]
    body[2500] = "    def lookup_default(self, name):"
    text, cut = window_file("\n".join(body), "Find Context.lookup_default", budget=800)
    assert cut and "2501 |     def lookup_default(self, name):" in text


def test_window_file_keeps_real_line_numbers():
    body = [f"x{i} = {i}  # filler filler filler filler filler" for i in range(2000)]
    body[1233] = "def repair_yaml_output(raw_text):"
    text, cut = window_file("\n".join(body), "Find where repair_yaml_output is defined", budget=600)
    assert cut and "1234 | def repair_yaml_output(raw_text):" in text


# ---------------------------------------------------------------- commands


def test_bare_ordo_prints_tree_help(capsys):
    code, out = run(capsys)
    assert code == 0 and "commands" in out and "├─ run" in out and "exit codes" in out
    with pytest.raises(SystemExit) as e:
        cli.main(["-h"])
    assert e.value.code == 0 and "└─ mcp" in capsys.readouterr().out


def test_subcommand_help_has_short_usage_and_examples(capsys):
    with pytest.raises(SystemExit):
        cli.main(["slice", "-h"])
    out = capsys.readouterr().out
    assert out.startswith("usage: ordo slice -i INTENT FILE [common options]") and "examples:" in out


def test_raw_output_saved_and_show(model, capsys):
    model.reply = SMOKE_OUT
    script = f"import sys; sys.stdout.write({SMOKE_LOG!r}); sys.exit(1)"
    code, out = run(capsys, "run", "--force", "-i", "why", sys.executable, "-c", script)
    rid = re.search(r"raw=(\S+)", out.splitlines()[0]).group(1)
    code, shown = run(capsys, "show", rid, "--grep", "KeyError", "-C", "0")
    assert code == 0 and f"# ordo show {rid}" in shown and "E       KeyError: 'BRL'" in shown
    assert all(re.match(r"^\s*\d+ \| ", l) or l in ("--",) for l in shown.splitlines()[1:])
    code, shown = run(capsys, "show", "--lines", "1-2")  # last
    assert len(shown.splitlines()) == 3 and " 1 | " in shown
    assert run(capsys, "show", "nope")[0] == 2


def test_json_envelope(model, capsys):
    model.reply = SMOKE_OUT
    script = f"import sys; sys.stdout.write({SMOKE_LOG!r}); sys.exit(1)"
    code, out = run(capsys, "run", "--json", "--force", "-i", "why", sys.executable, "-c", script)
    env = json.loads(out)
    assert code == 1 and env["exit"] == 1 and env["route"] == "model" and env["raw_id"]
    assert env["result"]["status"] == "FAIL" and env["result"]["causal_trace"][0]["file"] == "tests/test_settlement.py"
    assert any("unverified" in w for w in env["warnings"]) and env["text"] is None
    code, out = run(capsys, "run", "--json", "-i", "x", "--", sys.executable, "-c", "print('hi')")
    env = json.loads(out)
    assert env["route"] == "bypass" and env["result"] is None and env["text"].strip() == "hi"


def test_guide(capsys):
    code, out = run(capsys, "guide")
    assert code == 0 and out.startswith("## Ordo") and "ordo show <id>" in out and "ordo run -i" in out


def test_run_bypass_short_output(model, capsys):
    code, out = run(capsys, "run", "-i", "x", "--", sys.executable, "-c", "print('hi')")
    assert code == 0 and "short output" in out and "hi" in out and not model.calls


def test_run_propagates_exit_code_and_filters(model, capsys):
    model.reply = SMOKE_OUT
    script = f"import sys; sys.stdout.write({SMOKE_LOG!r}); sys.exit(1)"
    code, out = run(capsys, "run", "--force", "-i", "why settlement fails", sys.executable, "-c", script)
    assert code == 1
    assert out.startswith("# ordo run | exit=1 |")
    assert model.calls[0]["mode"] == "TOOL_OUTPUT" and "KeyError: 'BRL'" in model.calls[0]["raw"]
    assert "# ordo: unverified: snippet not verbatim: tests/test_settlement.py:88" in out
    ev = list(Session.open().events("run"))[-1]
    assert ev["exit"] == 1 and ev["status"] == "FAIL" and ev["route"] == "model"


def test_run_fallback_when_model_down(model, capsys):
    model.reply = ConnectionError("refused")
    script = f"import sys; sys.stdout.write({SMOKE_LOG!r}); sys.exit(1)"
    code, out = run(capsys, "run", "--force", "-i", "x", sys.executable, "-c", script)
    assert code == 1 and "model call failed" in out and "KeyError: 'BRL'" in out


def test_run_no_fallback(model, capsys):
    model.reply = "Sure! The tests failed because of a missing key."
    with pytest.raises(SystemExit) as e:
        run(capsys, "run", "--no-fallback", "--force", "-i", "x", sys.executable, "-c", "print(1)")
    assert e.value.code == 3


def test_run_repaired_output_is_normalized(model, capsys):
    model.reply = "```yaml\nstatus: FAIL\nsummary: 'it's broken'\n```"
    code, out = run(capsys, "run", "--force", "-i", "x", sys.executable, "-c", "print('it''s broken')")
    assert "summary: it's broken" in out and "# ordo: output YAML was repaired" in out


def test_filter_without_pipe_fails_fast(model, capsys, monkeypatch):
    class Tty:
        def isatty(self):
            return True
    monkeypatch.setattr(sys, "stdin", Tty())
    code = cli.main(["filter", "-i", "x"])
    assert code == 2 and "nothing on stdin" in capsys.readouterr().err and not model.calls


def test_filter_stdin(model, capsys, monkeypatch):
    import io
    monkeypatch.setattr(sys, "stdin", io.TextIOWrapper(io.BytesIO(SMOKE_LOG.encode())))
    code, out = run(capsys, "filter", "--force", "-i", "x")
    assert code == 0 and model.calls and out.startswith("# ordo filter")


def test_slice_sends_numbered_file_and_verifies_against_disk(model, capsys, tmp_path):
    f = tmp_path / "s.py"
    f.write_text("import os\n\n\ndef repair_yaml_output(raw_text):\n    return raw_text\n", encoding="utf-8")
    model.reply = ("status: FOUND\nsummary: x\nrelevant_symbols:\n- file: s.py\n  lines: '4-5'\n"
                   "  symbol: repair_yaml_output\n  issue: y\nexact_snippets:\n- location: s.py:4\n"
                   "  code: '4 | def repair_yaml_output(raw_text):'")
    code, out = run(capsys, "slice", "-i", "where is repair_yaml_output", str(f))
    raw = model.calls[0]["raw"]
    assert model.calls[0]["mode"] == "CODE_SLICE" and "4 | def repair_yaml_output(raw_text):" in raw
    assert "unverified" not in out


def test_search_python_fallback(model, capsys, tmp_path, monkeypatch):
    (tmp_path / "a.py").write_text("def call_ollama_ordo():\n    pass\n", encoding="utf-8")
    monkeypatch.setattr(cli.shutil, "which", lambda name: None if name == "rg" else None)
    code, out = run(capsys, "search", "call_ollama_ordo", str(tmp_path))
    assert code == 0 and "a.py:1:def call_ollama_ordo():" in out  # short: bypass


def _session_steps(capsys):
    run(capsys, "dead-end", "Global asyncio.Lock in webhook", "Caused worker deadlock under load")
    run(capsys, "note", "pool exhausted when batch > 50 (src/ledger/pool.py:42)")
    run(capsys, "run", "-i", "tests", "--", sys.executable, "-m", "pytest", "--version")


def test_compact_deterministic_by_default(model, capsys):
    _session_steps(capsys)
    n_calls = len(model.calls)
    code, out = run(capsys, "compact", "-i", "Stabilize webhook idempotency")
    assert len(model.calls) == n_calls  # no model call
    res = __import__("ordo.schema", fromlist=["parse_output"]).parse_output(out.split("\n", 1)[1], "MEMORY_COMPACT")
    assert res.schema_ok and res.obj.active_goal == "Stabilize webhook idempotency"
    assert res.obj.dead_ends_do_not_repeat[0].attempt == "Global asyncio.Lock in webhook"
    assert "pool exhausted when batch > 50 (src/ledger/pool.py:42)" in res.obj.confirmed_facts
    assert any(f.startswith("Last `") and "pytest --version`: exit 0" in f for f in res.obj.confirmed_facts)
    # the next compact inherits the previous ledger without duplicating it
    run(capsys, "dead-end", "Retry with backoff", "Still double-charged")
    code, out = run(capsys, "compact", "-i", "again")
    res = __import__("ordo.schema", fromlist=["parse_output"]).parse_output(out.split("\n", 1)[1], "MEMORY_COMPACT")
    assert [d.attempt for d in res.obj.dead_ends_do_not_repeat] == ["Global asyncio.Lock in webhook", "Retry with backoff"]


def test_compact_neural_valid_and_invalid(model, capsys):
    _session_steps(capsys)
    model.reply = ("status: COMPACTED\nactive_goal: Stabilize webhook\nconfirmed_facts:\n- pool exhausted\n"
                   "dead_ends_do_not_repeat:\n- attempt: Global asyncio.Lock in webhook\n  reason_failed: deadlock\n")
    code, out = run(capsys, "compact", "--neural", "-i", "Stabilize webhook idempotency")
    traj = model.calls[-1]["raw"]
    assert "[dead-end] attempt: Global asyncio.Lock in webhook" in traj and "[note] pool exhausted" in traj
    assert "[turn 1] $ " in traj and "active_goal: Stabilize webhook" in out
    # output outside the schema (what the published model does) -> deterministic ledger
    model.reply = "status: COMPACTED\nactive_goal: x\nconfirmed_facts:\n- file: a.py\n  lines: '5'\n"
    code, out = run(capsys, "compact", "--neural", "-i", "goal B")
    assert "model output failed the schema; deterministic ledger instead" in out and "active_goal: goal B" in out
    data = json.loads(run(capsys, "stats", "--json")[1])
    assert data["by_command"]["run"]["calls"] == 1 and data["by_command"]["compact"]["calls"] == 2
