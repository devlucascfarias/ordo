"""Monorepo logs: the failing package runs in the middle of a long log, paths are relative to the package."""
import json

import pytest

from ordo import cli
from ordo.client import OrdoResult
from ordo.textproc import est_tokens, is_error_line, window_log


@pytest.fixture(autouse=True)
def ordo_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORDO_SESSION", "mono")
    return tmp_path


def _monorepo_log(fail_in: str = "tax") -> str:
    out = []
    for pkg in ["fx", "ledger", "tax", "invoices", "payroll", "reports"]:
        out += [f"> {pkg}:test  python -m pytest", f"rootdir: /repo/packages/{pkg}"]
        out += [f"tests/test_{pkg}.py::test_case[{n}] PASSED   [{n}%]" for n in range(150)]
        if pkg == fail_in:
            out += ["tests/test_tax.py::test_limit FAILED", ">       assert tax_due(4000) == 300.0",
                    "tests/test_tax.py:14: AssertionError", "=== 1 failed, 150 passed in 0.7s ==="]
        else:
            out += ["=== 150 passed in 0.7s ==="]
    out += ["packages: 5 passed, 1 failed", "failed: tax"]
    return "\n".join(out)


def test_zero_failure_counts_are_not_error_lines():
    for line in ("ℹ fail 0", "# fail 0", "=== 0 failed, 3 passed", "errors: 0", "Tests: 0 failed, 4 passed"):
        assert not is_error_line(line), line
    for line in ("ℹ fail 1", "1 failed, 0 errors", "AssertionError: x", "fail 10"):
        assert is_error_line(line), line


def test_window_keeps_the_package_header_of_the_failure():
    text = _monorepo_log()
    body, cut = window_log(text, 1200)
    assert cut and "tests/test_tax.py:14: AssertionError" in body
    tax_root, fail = body.index("rootdir: /repo/packages/tax"), body.index("tests/test_tax.py:14")
    assert tax_root < fail
    assert "rootdir: /repo/packages/invoices" not in body[tax_root:fail]


def test_resolve_paths_names_the_only_match(tmp_path):
    (tmp_path / "packages/pricing/tests").mkdir(parents=True)
    (tmp_path / "packages/pricing/tests/pricing.test.js").write_text("x")
    (tmp_path / "packages/a/tests").mkdir(parents=True)
    (tmp_path / "packages/a/tests/util.js").write_text("x")
    (tmp_path / "packages/b/tests").mkdir(parents=True)
    (tmp_path / "packages/b/tests/util.js").write_text("x")
    data = {"causal_trace": [{"file": "tests\\pricing.test.js"}, {"file": "tests/util.js"}, {"file": "tests/none.js"}]}
    notes = cli.resolve_paths(data, str(tmp_path))
    assert notes == ["path: tests\\pricing.test.js is packages/pricing/tests/pricing.test.js from the working directory"]
    assert cli.resolve_paths({"causal_trace": [{"file": "packages/pricing/tests/pricing.test.js"}]}, str(tmp_path)) == []


class _Fake:
    def __init__(self, reply):
        self.reply = reply

    def __call__(self, mode, intent, raw, **kw):
        return OrdoResult(text=self.reply, prompt_tokens=est_tokens(raw), output_tokens=20, seconds=0.5)


def test_dropped_noise_that_claims_a_failure_is_removed(monkeypatch, capsys, tmp_path):
    reply = ("status: FAIL\nsummary: 'pytest failed at tests/test_tax.py:14'\ncausal_trace:\n"
             "- file: tests/test_tax.py\n  lines: '14'\n  issue: assertion\n"
             "dropped_noise: passing tests and one unrelated failure in invoices\n")
    monkeypatch.setattr(cli, "call_ordo", _Fake(reply))
    log = tmp_path / "log.txt"
    log.write_text(_monorepo_log(), encoding="utf-8")
    assert cli.main(["filter", "-i", "why?", str(log), "--json", "--deep"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["route"] == "model" and "dropped_noise" not in env["result"]
    monkeypatch.setattr(cli, "call_ordo", _Fake(reply.replace("and one unrelated failure in invoices", "only")))
    assert cli.main(["filter", "-i", "why?", str(log), "--json", "--deep"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["dropped_noise"] == "passing tests only"


def test_error_index_and_uncovered_errors():
    from ordo.textproc import error_index
    dump = "\n".join(f"  {i}: shader line" for i in range(150, 170))
    log = "\n".join([
        "[browser:error] THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false", "FRAGMENT",
        "ERROR: 0:166: 'pow' : no matching overloaded function found", dump,
        "[browser:error] THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false",
        "ERROR: 0:166: 'pow' : no matching overloaded function found", dump,
        "  x  2 tests/scene.spec.ts:42:1 › the sun core is bright and warm (127ms)",
        "    Error: Mercurio lit side rgb=5,4,4",
        "src/orbits.ts(23,73): error TS2339: Property 'radius' does not exist on type 'PlanetSpec'.",
    ])
    idx = error_index(log)
    assert idx.count("ERROR: 0:166: 'pow' : no matching overloaded function found") == 1  # repeated dumps once
    answer = "summary: 'failed at src/orbits.ts:23 with Mercurio lit side rgb=5,4,4'"
    left = cli.uncovered_errors(log, answer)
    assert left[0].startswith("ERROR: 0:166: 'pow'")
    assert not any("orbits.ts" in l or "Mercurio" in l for l in left)
    assert any("sun core" in l for l in left)
