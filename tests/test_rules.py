import json
import re
import sys

import pytest

from ordo import cli
from ordo.client import OrdoResult
from ordo.rules import pass_summary
from ordo.schema import GcMode, parse_output
from ordo.textproc import est_tokens


@pytest.fixture(autouse=True)
def ordo_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORDO_SESSION", "rules-test")
    return tmp_path


def _noise(n: int = 60, prefix: str = "test_item") -> str:
    return "\n".join(f"{prefix}_{i} ... ok (0.01s)" for i in range(n))


def _run(capsys, *argv):
    code = cli.main(list(argv))
    return code, capsys.readouterr().out


# ---------------------------------------------------------------- positive runner cases


def test_pass_summary_pytest():
    out = _noise() + "\n===== 121 passed, 120 warnings in 2.5s =====\n"
    assert pass_summary(out, 0) == "pytest: 121 passed, 120 warnings"

    short_q = _noise() + "\n5 passed, 2 skipped in 0.1s\n"
    assert pass_summary(short_q, 0) == "pytest: 5 passed, 2 skipped"


def test_pass_summary_node_test_spec_and_tap():
    spec = (
        _noise(prefix="✔ case")
        + "\nℹ tests 351\nℹ suites 42\nℹ pass 351\nℹ fail 0\nℹ cancelled 0\nℹ skipped 0\nℹ todo 0\nℹ duration_ms 912.4\n"
    )
    assert pass_summary(spec, 0) == "node:test: 351 passed, 0 failed"

    tap = (
        "TAP version 13\n"
        + _noise(prefix="ok")
        + "\n1..351\n# tests 351\n# suites 42\n# pass 351\n# fail 0\n# cancelled 0\n# skipped 0\n# todo 0\n"
    )
    assert pass_summary(tap, 0) == "node:test: 351 passed, 0 failed"


def test_pass_summary_jest():
    out = (
        _noise(prefix="PASS src/pkg")
        + "\nTest Suites: 4 passed, 4 total\n"
        + "Tests:       12 passed, 12 total\n"
        + "Snapshots:   0 total\n"
        + "Time:        1.842 s\n"
    )
    assert pass_summary(out, 0) == "jest: 12 passed, 12 total"


def test_pass_summary_vitest():
    out = (
        _noise(prefix=" ✓ src/mod")
        + "\n Test Files  3 passed (3)\n"
        + "      Tests  12 passed (12)\n"
        + "   Start at  10:00:00\n"
        + "   Duration  845ms\n"
    )
    assert pass_summary(out, 0) == "vitest: 12 passed (12)"


def test_pass_summary_go_test():
    multi = (
        _noise(prefix="=== RUN   TestPkg")
        + "\nok  \tgithub.com/acme/app/pkg/a\t0.12s\n"
        + "ok  \tgithub.com/acme/app/pkg/b\t0.05s\n"
        + "?   \tgithub.com/acme/app/cmd\t[no test files]\n"
    )
    assert pass_summary(multi, 0) == "go test: 2 packages passed"

    single = _noise(prefix="--- PASS: TestOne") + "\nok  \tpkg\t0.12s\n"
    assert pass_summary(single, 0) == "go test: 1 package passed (pkg)"


def test_pass_summary_cargo_test():
    out = (
        _noise(prefix="test core::case")
        + "\ntest result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.04s\n"
    )
    assert pass_summary(out, 0) == "cargo test: 10 passed, 0 failed"


def test_pass_summary_unittest():
    out = (
        _noise(prefix="test_func")
        + "\n----------------------------------------------------------------------\n"
        + "Ran 12 tests in 0.01s\n\nOK\n"
    )
    assert pass_summary(out, 0) == "unittest: 12 tests OK"


def test_pass_summary_uses_last_matching_summary():
    out = (
        _noise()
        + "\n===== 5 passed in 0.1s =====\n"
        + "running integration suite...\n"
        + "===== 42 passed, 1 skipped in 1.2s =====\n"
    )
    assert pass_summary(out, 0) == "pytest: 42 passed, 1 skipped (last of 2 runs, all passing)"


# ---------------------------------------------------------------- negative cases


def test_pass_summary_rejects_nonzero_exit_code():
    out = _noise() + "\n===== 121 passed in 2.5s =====\n"
    assert pass_summary(out, 1) is None


def test_pass_summary_rejects_failures_even_with_exit_zero():
    assert pass_summary(_noise() + "\n===== 1 failed, 120 passed in 2.5s =====\n", 0) is None
    assert pass_summary(_noise() + "\n===== 2 errors, 10 passed in 1.0s =====\n", 0) is None
    assert pass_summary(_noise() + "\nℹ tests 10\nℹ pass 9\nℹ fail 1\n", 0) is None
    assert pass_summary(_noise() + "\nTests:       1 failed, 11 passed, 12 total\n", 0) is None
    assert pass_summary(_noise() + "\n      Tests  1 failed | 11 passed (12)\n", 0) is None
    assert pass_summary(_noise() + "\ntest result: FAILED. 9 passed; 1 failed; 0 ignored\n", 0) is None
    assert pass_summary(_noise() + "\nRan 12 tests in 0.01s\n\nFAILED (failures=1)\n", 0) is None


def test_pass_summary_rejects_unknown_runner():
    out = _noise(prefix="step") + "\nAll checks completed successfully.\n"
    assert pass_summary(out, 0) is None


def test_pass_summary_rejects_go_output_with_fail_line():
    out = (
        _noise(prefix="=== RUN   TestX")
        + "\nok  \tpkg/a\t0.10s\n"
        + "FAIL\tpkg/b\t0.20s\n"
        + "ok  \tpkg/c\t0.05s\n"
    )
    assert pass_summary(out, 0) is None


# ---------------------------------------------------------------- end-to-end ordo run + stats/savings/dashboard


def test_ordo_run_uses_rule_without_calling_model(monkeypatch, capsys):
    def fail_if_called(*args, **kwargs):
        pytest.fail("local model must not be called for a passing test suite")

    monkeypatch.setattr(cli, "call_ordo", fail_if_called)

    fake_pytest = _noise(70, prefix="tests/test_app.py::test_case") + "\n===== 70 passed, 3 warnings in 1.42s =====\n"
    script = f"import sys; sys.stdout.write({fake_pytest!r}); sys.exit(0)"

    # Default YAML output
    code, out = _run(capsys, "run", "-i", "did tests pass?", "--", sys.executable, "-c", script)
    assert code == 0
    header, body = out.split("\n", 1)
    assert header.startswith("# ordo run | exit=0 | ~") and "passing suite (rule, no model) | raw=" in header
    parsed = parse_output(body, GcMode.TOOL_OUTPUT)
    assert parsed.schema_ok
    assert parsed.data == {"status": "PASS", "summary": "pytest: 70 passed, 3 warnings"}

    # Raw output is saved and recoverable via `ordo show`
    rid = re.search(r"raw=(\S+)", header).group(1)
    show_code, shown = _run(capsys, "show", rid, "--grep", "70 passed", "-C", "0")
    assert show_code == 0 and "70 passed, 3 warnings" in shown

    # JSON envelope
    code, out = _run(capsys, "run", "--json", "-i", "did tests pass?", "--", sys.executable, "-c", script)
    assert code == 0
    env = json.loads(out)
    assert env["route"] == "rule"
    assert env["status"] == "PASS"
    assert env["exit"] == 0
    assert env["seconds"] == 0.0
    assert env["raw_id"]
    assert env["result"] == {"status": "PASS", "summary": "pytest: 70 passed, 3 warnings"}
    assert env["text"] is None
    assert env["tokens"]["input"] > env["tokens"]["output"] > 0

    # Stats, savings, and dashboard handle route "rule" properly
    code, stats_out = _run(capsys, "stats", "--json")
    assert code == 0
    stats = json.loads(stats_out)
    assert stats["total"]["rule"] == 2
    assert stats["by_command"]["run"]["rule"] == 2
    assert stats["total"]["calls"] == 2

    code, sav_out = _run(capsys, "savings", "--json")
    assert code == 0
    sav = json.loads(sav_out)
    assert sav["total"]["rule"] == 2
    assert sav["total"]["filtered"] == 2
    assert sav["total"]["saved"] > 0
    assert sav["total"]["model_seconds"] == 0.0

    code, _ = _run(capsys, "dashboard", "--no-open")
    assert code == 0


def test_ordo_run_force_bypasses_rule(monkeypatch, capsys):
    calls = []

    def fake_model(mode, intent, raw, **kw):
        calls.append(raw)
        return OrdoResult(text="status: PASS\nsummary: forced model", prompt_tokens=est_tokens(raw),
                          output_tokens=10, seconds=0.2)

    monkeypatch.setattr(cli, "call_ordo", fake_model)

    fake_pytest = _noise(70) + "\n===== 70 passed in 1.0s =====\n"
    script = f"import sys; sys.stdout.write({fake_pytest!r}); sys.exit(0)"
    code, out = _run(capsys, "run", "--force", "--json", "-i", "check", "--", sys.executable, "-c", script)
    assert code == 0 and len(calls) == 1
    assert json.loads(out)["route"] == "model"


def test_rule_skips_when_any_block_failed_or_intent_is_specific():
    def block(counts):
        return _noise() + "\n=== " + counts + " in 0.1s ===\n"

    monorepo = block("10 passed") + block("1 failed, 9 passed") + block("12 passed")
    assert pass_summary(monorepo, 0) is None  # the script exited 0, but one package failed
    assert pass_summary(block("10 passed") + block("12 passed"), 0) == "pytest: 12 passed (last of 2 runs, all passing)"
    single = block("12 passed, 3 warnings")
    assert pass_summary(single, 0, "confirm all tests pass") == "pytest: 12 passed, 3 warnings"
    assert pass_summary(single, 0, "which deprecation warnings appear?") is None
    assert pass_summary(single, 0, "which tests are slow?") is None


def test_pass_summary_playwright():
    ok = _noise(prefix="  ok 1 tests/scene.spec.ts:38:1 › renders") + "\n\n  8 passed (8.6s)\n"
    assert pass_summary(ok, 0) == "playwright: 8 passed (8.6s)"
    bad = _noise() + "\n  1 failed\n    tests/a.spec.ts:3:1 › x\n  7 passed (9.0s)\n"
    assert pass_summary(bad, 0) is None
    flaky = _noise() + "\n  1 flaky\n  7 passed (9.0s)\n"
    assert pass_summary(flaky, 0) is None
