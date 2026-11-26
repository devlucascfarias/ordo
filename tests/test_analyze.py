"""Deterministic reading of failures (route index), links to the source, and the comparison between runs."""
import json
import sys
from pathlib import Path

import pytest

from ordo import cli
from ordo.analyze import analyze, cause_files, detect_runners, to_result


@pytest.fixture(autouse=True)
def ordo_home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDO_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("ORDO_SESSION", "an")
    return tmp_path


def _repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "repo"
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")
    return root


def _no_model(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the model must not be called")
    monkeypatch.setattr(cli, "call_ordo", boom)


def test_runners_from_command_and_output():
    assert detect_runners("npx playwright test", "") == ["playwright"]
    assert detect_runners("npm test", "> tsc --noEmit\n> playwright test\nRunning 3 tests using 1 worker") == ["playwright", "tsc"]
    assert "pytest" in detect_runners(None, "===== test session starts =====")


def test_tsc_cascade_becomes_one_cause_at_the_type(tmp_path):
    root = _repo(tmp_path, {"src/types.ts": "export interface PlanetSpec {\n  name: string;\n  bodyRadius: number;\n}\n"})
    log = "\n".join(f"src/f{i}.ts({i},3): error TS2339: Property 'radius' does not exist on type 'PlanetSpec'." for i in range(1, 7))
    res = to_result(analyze(log, "npm run typecheck", root), 2)
    assert len(res["causes"]) == 1
    cause = res["causes"][0]
    assert cause["at"] == "src/types.ts:1" and cause["seen"] == 6
    assert any("renamed to 'bodyRadius'" in l for l in cause["links"])


def test_glsl_undeclared_identifier_links_declaration_guard_and_switch(tmp_path):
    root = _repo(tmp_path, {
        "src/chunks/fresnel.ts": "export default `\n#ifdef USE_FRESNEL\nvarying vec3 vView;\n#endif\n`;\n",
        "src/chunks/corona.ts": "export default `\n#ifdef USE_CORONA\nfloat c = dot(vNormal, vView);\n#endif\n`;\n",
        "src/config.ts": "export const SUN = {\n  fresnel: false,\n  corona: true,\n};\n",
    })
    log = ("[browser:error] THREE.WebGLProgram: Shader Error 0 - VALIDATE_STATUS false\n"
           "ERROR: 0:212: 'vView' : undeclared identifier\n"
           "ERROR: 0:212: 'dot' : no matching overloaded function found\n> 212: float c = dot(vNormal, vView);\n") * 3
    res = to_result(analyze(log, "npx playwright test", root), 1)
    assert len(res["causes"]) == 1  # one failing line, the generic "Shader Error" folded in
    links = "\n".join(res["causes"][0]["links"])
    assert "declared at src/chunks/fresnel.ts:3 (under #ifdef USE_FRESNEL)" in links
    assert "used at src/chunks/corona.ts:3 (under #ifdef USE_CORONA)" in links
    assert "switch for USE_FRESNEL at src/config.ts:2: fresnel: false," in links


def test_nan_and_missing_asset_links(tmp_path):
    root = _repo(tmp_path, {"public/textures/rings/saturn_alpha.png": "x", "public/textures/rings/uranus.png": "x"})
    log = ("[browser:log] [data] moons=Io(e=0.004),Europa(e=NaN)\n"
           "[browser:error] THREE.BufferGeometry.computeBoundingSphere(): Computed radius is NaN.\n"
           "[browser:error] [assets] could not load texture /textures/rings/saturn-alpha.png\n")
    causes = to_result(analyze(log, None, root), 1)["causes"]
    nan = next(c for c in causes if "NaN" in c["error"])
    assert any("Europa(e=NaN)" in l for l in nan["links"])
    tex = next(c for c in causes if "texture" in c["error"])
    assert "public/textures/rings/saturn_alpha.png" in tex["links"][0]
    assert cause_files(analyze(log, None, root)) == {c: [] for c in cause_files(analyze(log, None, root))}


def test_node_test_details_and_package_path(tmp_path):
    root = _repo(tmp_path, {"packages/pricing/tests/pricing.test.js": "x"})
    log = ("> @shop/pricing test\n✖ pricing: ten units (1.8ms)\nℹ tests 111\nℹ fail 1\n✖ failing tests:\n\n"
           "test at tests\\pricing.test.js:13:1\n✖ pricing: ten units (1.8ms)\n"
           "  AssertionError [ERR_ASSERTION]: Expected values to be strictly equal:\n\n  45 !== 39.6\n\n"
           f"      at TestContext.<anonymous> (file:///{root.as_posix()}/packages/pricing/tests/pricing.test.js:14:10)\n")
    t = to_result(analyze(log, "node --test", root), 1)["failing_tests"][0]
    assert t["at"] == "packages/pricing/tests/pricing.test.js:14"
    assert "45 !== 39.6" in t["detail"]


def test_filter_uses_the_index_without_the_model(tmp_path, monkeypatch, capsys):
    _no_model(monkeypatch)
    monkeypatch.chdir(_repo(tmp_path, {"a.py": "x"}))
    log = tmp_path / "log.txt"
    log.write_text("\n".join(f"noise {i}" for i in range(80)) + "\nTraceback (most recent call last):\n"
                   "  File \"a.py\", line 1, in <module>\nValueError: boom\n", encoding="utf-8")
    assert cli.main(["filter", "-i", "why does it fail?", str(log), "--json"]) == 0
    env = json.loads(capsys.readouterr().out)
    assert env["route"] == "index" and env["result"]["causes"][0]["error"] == "ValueError: boom"


def test_specific_questions_and_deep_still_use_the_model(tmp_path, monkeypatch, capsys):
    calls = []

    def fake(mode, intent, raw, **kw):
        from ordo.client import OrdoResult
        calls.append(intent)
        return OrdoResult(text="status: FAIL\nsummary: boom", prompt_tokens=10, output_tokens=5, seconds=0.1)

    monkeypatch.setattr(cli, "call_ordo", fake)
    log = tmp_path / "log.txt"
    log.write_text("\n".join(f"noise {i}" for i in range(80)) + "\nValueError: boom\n", encoding="utf-8")
    cli.main(["filter", "-i", "which deprecation warnings show up?", str(log), "--json"])
    cli.main(["filter", "-i", "why?", str(log), "--json", "--deep"])
    assert len(calls) == 2


def test_run_compares_with_the_last_run_and_diff(tmp_path, monkeypatch, capsys):
    _no_model(monkeypatch)
    root = _repo(tmp_path, {"a.py": "x", "b.py": "x", "step.txt": "1"})
    monkeypatch.chdir(root)
    script = root / "check.py"
    script.write_text(
        "import pathlib\nstep = pathlib.Path('step.txt').read_text().strip()\n"
        "print('\\n'.join(f'noise {i}' for i in range(60)))\n"
        "print('Traceback (most recent call last):')\nprint('  File \"a.py\", line 1, in <module>')\n"
        "print('ValueError: first')\n"
        "if step == '1':\n    print('Traceback (most recent call last):')\n    print('  File \"b.py\", line 1, in <module>')\n"
        "    print('KeyError: second')\n"
        "raise SystemExit(1)\n", encoding="utf-8")
    cmd = ["run", "-i", "why?", "--", sys.executable, "check.py"]
    cli.main(cmd)
    capsys.readouterr()
    (root / "step.txt").write_text("2", encoding="utf-8")  # the KeyError goes away without touching b.py
    cli.main(cmd)
    out = capsys.readouterr().out
    assert "since the last run of this command" in out and "1 failure(s) gone, 0 new, 1 still failing" in out
    assert "'KeyError: second' is gone, but none of the files it pointed to changed (b.py)" in out
    assert cli.main(["diff", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["gone"] == ["KeyError: second"] and d["still"] == ["ValueError: first"] and d["new"] == []


def test_confidence_rule_and_called_from(tmp_path):
    from ordo.analyze import confident
    root = _repo(tmp_path, {"src/app.py": "x", "tests/test_app.py": "x"})
    tb = ("_________ test_total _________\ntests/test_app.py:12: in test_total\n    assert total() == 3\n"
          "src/app.py:4: in total\n    return 1 / 0\nE   ZeroDivisionError: division by zero\n"
          "FAILED tests/test_app.py::test_total - ZeroDivisionError: division by zero\n")
    a = analyze(tb, "pytest", root, "why does it fail?")
    item = to_result(a, 1)["failing_tests"][0]
    assert item["at"] == "src/app.py:4" and item["called_from"] == "tests/test_app.py:12"
    assert confident(a, "why does it fail?")
    assert not confident(a, "what about the test_fmt failure?")  # names something no failure mentions
    unknown = analyze("Something went wrong somewhere\nFatal: it broke\n", None, root)
    assert not confident(unknown)
