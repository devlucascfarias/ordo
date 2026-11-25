import io
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
    monkeypatch.setattr(cli, "call_ordo", fake)
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
    assert code == 0 and out.startswith("## Ordo") and "ordo show <raw_id>" in out and "ordo run -i" in out


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
    code, out = run(capsys, "slice", "--force", "-i", "where is repair_yaml_output", str(f))
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


class _TTY(io.StringIO):
    def isatty(self):
        return True


def test_guide_is_plain_markdown_when_piped_and_rendered_at_a_terminal(monkeypatch, capsys):
    code, out = run(capsys, "guide")  # capsys is not a TTY: exactly the Markdown agents read
    assert code == 0 and out == cli.GUIDE.rstrip("\n") + "\n"
    tty = _TTY()
    monkeypatch.setattr(sys, "stdout", tty)
    assert cli.main(["guide"]) == 0
    shown = re.sub(r"\x1b\[[0-9;]*m", "", tty.getvalue())
    assert "Output contract" in shown and "### " not in shown and "|---|" not in shown
    tty2 = _TTY()
    monkeypatch.setattr(sys, "stdout", tty2)
    assert cli.main(["guide", "--raw"]) == 0 and tty2.getvalue() == cli.GUIDE.rstrip("\n") + "\n"


def test_top_help_is_plain_when_piped_and_colored_at_a_terminal(monkeypatch, capsys):
    code, out = run(capsys)  # bare `ordo`, not a TTY: plain text, no escape codes
    assert code == 0 and out == cli.TOP_HELP.rstrip("\n") + "\n" and "\x1b[" not in out
    for argv in ([], ["--help"]):
        tty = _TTY()
        monkeypatch.setattr(sys, "stdout", tty)
        try:
            code = cli.main(argv)
        except SystemExit as e:  # argparse exits after --help
            code = e.code
        colored = tty.getvalue()
        assert code == 0 and "\x1b[" in colored
        assert re.sub(r"\x1b\[[0-9;]*m", "", colored).rstrip() == cli.TOP_HELP.rstrip()


def _ledger_repo(tmp_path):
    import os
    import subprocess
    import time

    from ordo.session import Session

    repo = tmp_path / "repo"
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    old, new = repo / "old.py", repo / "new.py"
    old.write_text("x = 1\n", encoding="utf-8")
    now = time.time()
    os.utime(old, (now - 600, now - 600))  # changed 10 min ago
    sess = Session(path=tmp_path / "s.jsonl", root=repo)  # session file outside the repository
    return sess, old, new, now


def test_ledger_ignores_trivial_and_stale_commands(tmp_path, monkeypatch):
    sess, *_, now = _ledger_repo(tmp_path)
    monkeypatch.setenv("ORDO_LEDGER_HOURS", "12")
    for cmd, ts in [("python -c print(1)", now - 60), ("pytest -q tests/old", now - 13 * 3600),
                    ("npx tsc --noEmit", now - 30)]:
        sess.append("run", command=cmd, exit=0)
        lines = sess.path.read_text(encoding="utf-8").splitlines()
        lines[-1] = lines[-1].replace('"ts": ', f'"ts": {ts}, "_ts": ', 1)
        sess.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger = cli.deterministic_ledger(sess, "goal")
    runs = [f for f in ledger.confirmed_facts if f.startswith("Last `")]
    assert runs == ["Last `npx tsc --noEmit`: exit 0 (PASS)"]
    # recomputed, never inherited: a later compact does not carry old command results
    sess.append("compact", intent="goal", ledger=cli._normalized(ledger.model_dump()))
    assert not [f for f in cli.deterministic_ledger(sess, "goal").confirmed_facts if f.startswith("Last `")]


def test_ledger_marks_files_changed_after_the_last_test_as_untested(tmp_path):
    import json
    import time

    sess, old, new, now = _ledger_repo(tmp_path)
    sess.path.write_text(json.dumps({"ts": now - 300, "kind": "run", "command": "pytest -q", "exit": 0,
                                     "cmd_seconds": 5}) + "\n", encoding="utf-8")
    new.write_text("y = 2\n", encoding="utf-8")  # created after the test run
    states = {f.file: f.test_status for f in cli.deterministic_ledger(sess, "goal").modified_files_state}
    assert states == {"old.py": "PASS", "new.py": "UNTESTED"}
    states = {f.file: f.test_status for f in cli.deterministic_ledger(
        type(sess)(path=tmp_path / "empty.jsonl", root=sess.root), "goal").modified_files_state}
    assert set(states.values()) == {"UNTESTED"}  # no test run at all


def test_repeated_trace_items_are_collapsed_and_flagged(model, capsys, tmp_path):
    item = "- file: src/app.py\n  lines: '12'\n  issue: Incompatible return value type\n"
    model.reply = "status: FAIL\nsummary: mypy errors\ncausal_trace:\n" + item * 5 + \
        "- file: src/app.py\n  lines: '30'\n  issue: Incompatible return value type\n" + \
        "- file: src/app.py\n  lines: '41'\n  issue: Incompatible return value type\n"
    log = tmp_path / "mypy.txt"
    log.write_text("\n".join(f"src/app.py:{n}: error: Incompatible return value type" for n in (12, 30, 41) * 20),
                   encoding="utf-8")
    code, out = run(capsys, "filter", "-i", "which type errors?", str(log))
    body = out.split("\n", 1)[1]
    assert code == 0 and body.count("lines: '12'") == 1
    assert "# ordo: repeated output: 4 identical causal_trace item(s) collapsed" in out
    assert "the same issue appears in 3 causal_trace items" in out


def test_distinct_trace_items_are_left_alone(model, capsys, tmp_path):
    model.reply = ("status: FAIL\nsummary: two errors\ncausal_trace:\n- file: a.py\n  lines: '1'\n  issue: KeyError\n"
                   "- file: b.py\n  lines: '2'\n  issue: ValueError\n")
    log = tmp_path / "log.txt"
    log.write_text("\n".join(["a.py:1: KeyError", "b.py:2: ValueError"] * 30), encoding="utf-8")
    code, out = run(capsys, "filter", "-i", "why?", str(log))
    assert code == 0 and "repeated output" not in out


def test_ledger_never_turns_the_model_summary_into_a_fact(tmp_path):
    import json
    import time

    sess, *_ = _ledger_repo(tmp_path)
    sess.path.write_text(json.dumps({"ts": time.time(), "kind": "run", "command": "mypy src", "exit": 1,
                                     "status": "PASS", "route": "model",
                                     "summary": "13 mypy errors in 13 files"}) + "\n", encoding="utf-8")
    facts = cli.deterministic_ledger(sess, "goal").confirmed_facts
    assert facts == ["Last `mypy src`: exit 1 (FAIL)"]  # exit code only; the model said PASS and invented counts


def _memory(capsys):
    code, out = run(capsys, "memory", "--json")
    return {m["text"]: m for m in json.loads(out)["items"]}


def test_memory_ids_are_stable_and_items_can_be_edited_and_forgotten(capsys):
    run(capsys, "note", "cache is keyed by item only")
    run(capsys, "note", "quota resets at 13:00")
    run(capsys, "dead-end", "Gemini before 13:00", "quota exhausted")
    mem = _memory(capsys)
    cache, quota = mem["cache is keyed by item only"]["id"], mem["quota resets at 13:00"]["id"]
    dead = mem["Gemini before 13:00"]["id"]
    assert cache.startswith("n") and dead.startswith("d") and len({cache, quota, dead}) == 3
    code, out = run(capsys, "forget", "13:00")  # a note and a dead end match: ambiguous
    assert code == 2 and "use an id" in out
    code, out = run(capsys, "forget", quota)
    assert code == 0 and f"forgot {quota}" in out
    assert _memory(capsys)["cache is keyed by item only"]["id"] == cache  # ids do not shift after a removal
    code, out = run(capsys, "edit", cache, "cache is keyed by item and model")
    code2, out2 = run(capsys, "edit", dead, "--reason", "quota is per account")
    assert code == 0 and code2 == 0
    mem = _memory(capsys)
    assert mem["cache is keyed by item and model"]["id"] == cache
    assert mem["Gemini before 13:00"]["reason"] == "quota is per account"
    code, out = run(capsys, "compact", "-i", "goal")
    assert "cache is keyed by item and model" in out and "quota resets" not in out
    assert "quota is per account" in out
    assert run(capsys, "edit", cache, "--reason", "x")[0] == 2  # --reason is for dead ends only
    assert run(capsys, "edit", "nffffff", "x")[0] == 1
    assert run(capsys, "forget", "nothing like this")[0] == 1


def test_memory_reads_records_written_before_ids(capsys, ordo_home):
    from ordo.session import Session

    sess = Session.open()
    sess.append("note", text="legacy fact")
    sess.append("dead_end", attempt="legacy attempt", reason="legacy reason")
    sess.append("forget", target="note", text="nothing")  # old-style retraction, harmless
    mem = _memory(capsys)
    assert mem["legacy fact"]["id"].startswith("n") and mem["legacy attempt"]["reason"] == "legacy reason"
    assert _memory(capsys)["legacy fact"]["id"] == mem["legacy fact"]["id"]  # derived ids are stable too


def test_tool_name_strips_wrappers_and_keeps_subcommands():
    cases = {"python -m pytest -q": "pytest", "npx tsc --noEmit": "tsc", "npm run build": "npm run build",
             "npm test": "npm test", "go test ./...": "go test", "cargo test --lib": "cargo test",
             "uv run ruff check src": "ruff", "C:/tools/mvn.cmd -q test": "mvn test", "./gradlew build": "gradlew build"}
    for cmd, name in cases.items():
        assert cli.tool_name({"kind": "run", "command": cmd}) == name, cmd
    assert cli.tool_name({"kind": "filter"}) == "ordo filter"


def test_savings_aggregates_all_projects_plain_json_and_terminal(ordo_home, capsys, monkeypatch):
    import io

    from ordo.session import Session

    for name, events in {"api": [("run", "python -m pytest -q", 4000, 60, "model"),
                                 ("run", "python -m pytest -q", 30, 30, "bypass")],
                         "web": [("run", "npx tsc --noEmit", 2000, 100, "model"), ("filter", None, 500, 50, "model")]}.items():
        sess = Session(path=ordo_home / "home" / "sessions" / f"{name}-0123456789.jsonl", root=ordo_home)
        for kind, cmd, raw, out, route in events:
            sess.append(kind, **({"command": cmd} if cmd else {}), raw_tokens=raw, out_tokens=out, route=route, seconds=2)
    code, out = run(capsys, "savings", "--json")
    rep = json.loads(out)
    assert rep["total"]["saved"] == 6290 and rep["total"]["calls"] == 4 and rep["total"]["filtered"] == 3
    assert list(rep["by_tool"]) == ["pytest", "tsc", "ordo filter"] and set(rep["by_project"]) == {"api", "web"}
    code, out = run(capsys, "savings")
    assert code == 0 and out.startswith("Ordo kept 6,290 tokens") and "\x1b[" not in out and "pytest" in out

    class TTY(io.StringIO):
        def isatty(self):
            return True

    tty = TTY()
    monkeypatch.setattr(sys, "stdout", tty)
    assert cli.main(["savings"]) == 0 and "\x1b[" in tty.getvalue() and "6,290" in tty.getvalue()


def test_outputs_and_named_sessions_are_kept_per_project(tmp_path, monkeypatch, capsys):
    import subprocess

    from ordo.session import Session

    monkeypatch.delenv("ORDO_SESSION", raising=False)
    a, b = tmp_path / "proj-a", tmp_path / "proj-b"
    for d in (a, b):
        subprocess.run(["git", "init", "-q", str(d)], check=True)
    sa, sb = Session.open(cwd=a), Session.open(cwd=b)
    ra, rb = cli.save_raw("output of A", sa), cli.save_raw("output of B", sb)
    assert sa.runs_dir() != sb.runs_dir()
    assert cli.find_raw(sa, "last")[0].read_text(encoding="utf-8") == "output of A"  # never B's, even if newer
    assert cli.find_raw(sa, rb)[0].read_text(encoding="utf-8") == "output of B"  # ids resolve from anywhere
    assert Session.open("main", cwd=a).path != Session.open("main", cwd=b).path  # --session main does not mix
    assert Session.open("main", cwd=a).project == "proj-a"


def test_flat_outputs_from_older_versions_move_to_their_project(ordo_home, capsys):
    from ordo.session import Session, ordo_home as home

    sess = Session.open()
    runs = home() / "runs"
    runs.mkdir(parents=True)
    (runs / "20260101-000000-abcd.log").write_text("old output", encoding="utf-8")
    (runs / "20260101-000001-ffff.log").write_text("orphan", encoding="utf-8")
    sess.append("run", command="pytest", exit=1, raw_id="20260101-000000-abcd")
    target, mine = cli.find_raw(sess, "last")
    assert target.read_text(encoding="utf-8") == "old output" and target.parent == sess.runs_dir()
    assert cli.find_raw(sess, "20260101-000001-ffff")[0].read_text(encoding="utf-8") == "orphan"  # still by id


def test_prune_lists_and_removes_only_with_criteria_and_consent(ordo_home, capsys):
    import json as _json
    import shutil as _sh

    from ordo.session import Session, ordo_home as home

    gone_root = ordo_home / "deleted-project"
    gone_root.mkdir()
    gone = Session(path=home() / "sessions" / "deleted-project-0123456789.jsonl", root=gone_root)
    gone.append("note", id="n000001", text="x")
    cli.save_raw("log", gone)
    _sh.rmtree(gone_root)
    keep = Session.open()
    keep.append("note", id="n000002", text="y")
    code, out = run(capsys, "prune")
    assert code == 0 and "deleted-project-0123456789" in out and "[folder missing]" in out
    code, out = run(capsys, "prune", "--missing")  # not a terminal and no --yes: dry run
    assert "nothing removed (dry run)" in out and gone.path.exists()
    code, out = run(capsys, "prune", "--missing", "--yes")
    assert code == 0 and not gone.path.exists() and not gone.runs_dir().exists() and keep.path.exists()
    code, out = run(capsys, "prune", "--json")
    assert [p["session"] for p in _json.loads(out)["projects"]] == [keep.key]


def test_unknown_command_suggests_the_closest_ones(capsys, monkeypatch):
    import io

    with pytest.raises(SystemExit) as e:
        cli.main(["usage"])
    err = capsys.readouterr().err
    assert e.value.code == 2 and "unknown command 'usage'" in err and "ordo savings, ordo stats" in err
    assert "\x1b[" not in err  # captured: plain text for agents
    with pytest.raises(SystemExit):
        cli.main(["stat"])
    assert "did you mean: ordo stats" in capsys.readouterr().err
    with pytest.raises(SystemExit) as e:
        cli.main(["slice"])  # error inside a command: its own usage line
    err = capsys.readouterr().err
    assert e.value.code == 2 and "usage: ordo slice -i INTENT FILE" in err

    class TTY(io.StringIO):
        def isatty(self):
            return True

    tty = TTY()
    monkeypatch.setattr(sys, "stderr", tty)
    with pytest.raises(SystemExit):
        cli.main(["usage"])
    assert "\x1b[" in tty.getvalue() and "ordo savings" in tty.getvalue()


def test_strict_checks_flag_test_names_as_lines_and_wrong_summary_locations(model, capsys, tmp_path):
    log = tmp_path / "log.txt"
    log.write_text("\n".join(["tests/test_api.py::test_users FAILED"] * 30
                             + ["src/api.py:12: KeyError: 'id'", "E   KeyError: 'id'"]), encoding="utf-8")
    model.reply = ("status: FAIL\nsummary: \"KeyError in src/api.py:12, also see src/other.py:99 and v1.2.3:4\"\n"
                   "causal_trace:\n- file: src/api.py\n  lines: '12'\n  issue: KeyError\n"
                   "- file: tests/test_api.py\n  lines: test_users\n  issue: failing test\n")
    code, out = run(capsys, "filter", "--force", "-i", "why?", str(log))
    assert "lines is not a line number ('test_users')" in out
    assert "summary cites a location not in input: src/other.py:99" in out
    assert "src/api.py:12" not in "".join(l for l in out.splitlines() if "summary cites" in l)
    assert "v1.2.3" not in out.split("# ordo:", 1)[1]  # version numbers are not locations


def test_very_long_lines_are_shortened_before_the_model(model, capsys, tmp_path):
    log = tmp_path / "mypy.txt"
    giant = "src/app.py:219: note:     def Popen(" + "x: int = ..., " * 200 + ")"
    log.write_text("\n".join([giant] * 3 + ["src/app.py:12: error: Name 'y' is not defined"] * 40), encoding="utf-8")
    code, out = run(capsys, "filter", "-i", "which errors?", str(log))
    sent = model.calls[-1]["raw"]
    assert max(len(l) for l in sent.splitlines()) < 900 and "chars omitted" in sent
    assert "3 very long line(s) shortened before the model" in out


def test_small_file_slice_is_shown_numbered_without_the_model(model, capsys, tmp_path):
    f = tmp_path / "pricing.py"
    f.write_text("TIERS = [(100, 0.15)]\n\n\ndef unit_price(base, qty):\n    return base\n", encoding="utf-8")
    n = len(model.calls)
    code, out = run(capsys, "slice", "-i", "where are tiers applied?", str(f))
    assert code == 0 and len(model.calls) == n and "short output, passed through" in out
    assert "4 | def unit_price(base, qty):" in out


def test_ledger_keeps_the_latest_result_per_tool(tmp_path):
    import json
    import time

    sess, *_ = _ledger_repo(tmp_path)
    now = time.time()
    sess.path.write_text("".join(json.dumps(e) + "\n" for e in [
        {"ts": now - 60, "kind": "run", "command": "python -m pytest -v", "exit": 1},
        {"ts": now - 30, "kind": "run", "command": "python -m pytest -q", "exit": 0},
        {"ts": now - 20, "kind": "run", "command": "npx tsc --noEmit", "exit": 2}]), encoding="utf-8")
    facts = [f for f in cli.deterministic_ledger(sess, "g").confirmed_facts if f.startswith("Last `")]
    assert facts == ["Last `python -m pytest -q`: exit 0 (PASS)", "Last `npx tsc --noEmit`: exit 2 (FAIL)"]


def test_recording_the_same_note_or_dead_end_again_returns_the_existing_id(capsys):
    _, out = run(capsys, "note", "median uses s[mid+1]")
    nid = re.search(r"note (n[0-9a-f]{6})", out).group(1)
    code, out = run(capsys, "note", "median uses s[mid+1]")
    assert code == 0 and f"note {nid} already recorded" in out
    assert run(capsys, "edit", nid, "median must average s[mid-1] and s[mid]")[0] == 0  # the returned id works
    _, out = run(capsys, "dead-end", "patch the test", "test must not change")
    did = re.search(r"dead-end (d[0-9a-f]{6})", out).group(1)
    _, out = run(capsys, "dead-end", "patch the test", "tests are the spec")
    assert f"dead-end {did} already recorded" in out and "reason updated" in out
    mem = _memory(capsys)
    assert len(mem) == 2 and mem["patch the test"]["reason"] == "tests are the spec"


def test_untracked_build_artifacts_are_not_modified_files(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    for rel in ("calc.py", "__pycache__/calc.cpython-313.pyc", "node_modules/x/index.js", "bin/tool.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("x", encoding="utf-8")
    assert sorted(cli._git_changed(tmp_path)) == ["calc.py"]


@pytest.mark.skipif(sys.platform != "win32", reason="cmd.exe quoting")
def test_cmd_exe_command_line_keeps_inner_quotes(tmp_path):
    cmd = ["cmd", "/d", "/s", "/c", f'"{sys.executable}" -c "print(\'ok\')" | findstr ok']
    target, shell = cli._resolve_command(cmd)
    code, out, _ = cli._execute(target, shell, 60, str(tmp_path))
    assert code == 0 and out.decode().strip() == "ok"


def test_dashboard_writes_a_self_contained_page_and_opens_it(ordo_home, capsys, monkeypatch, tmp_path):
    import webbrowser
    from pathlib import Path as P

    from ordo.session import Session

    fake_home = tmp_path / "someuser"
    fake_home.mkdir()
    monkeypatch.setattr(P, "home", classmethod(lambda cls: fake_home))
    sessions = ordo_home / "home" / "sessions"
    api = Session(path=sessions / "api&co-0123456789.jsonl", root=ordo_home)
    api.append("run", command="python -m pytest -q", raw_tokens=4000, out_tokens=60, route="model", seconds=2)
    api.append("run", command="python -m pytest -q", raw_tokens=30, out_tokens=30, route="bypass")
    home_s = Session(path=sessions / "someuser-0123456789.jsonl", root=fake_home)  # ran outside a repository
    home_s.append("filter", raw_tokens=500, out_tokens=50, route="model", seconds=1)
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda url: opened.append(url) or True)
    code, out = run(capsys, "dashboard")
    page = (ordo_home / "home" / "dashboard.html").read_text(encoding="utf-8")
    assert code == 0 and "dashboard.html" in out and opened and opened[0].startswith("file:")
    assert "<script" not in page and "http" not in page.replace("http-equiv", "")  # no external assets
    assert "api&amp;co" in page and "someuser" not in page and ">~<" in page  # escaped; home folder hidden
    assert "4,390" in page and "4,530" in page and "3,940" in page and "1 short enough to pass through" in page
    assert "Fira Code" in page and page.count("<svg") >= 5  # typeface and inline icons
    opened.clear()
    assert run(capsys, "dashboard", "--no-open")[0] == 0 and not opened


def test_dashboard_with_no_data_explains_how_to_start(ordo_home, capsys):
    code, _ = run(capsys, "dashboard", "--no-open")
    page = (ordo_home / "home" / "dashboard.html").read_text(encoding="utf-8")
    assert code == 0 and "No tool output recorded yet" in page
