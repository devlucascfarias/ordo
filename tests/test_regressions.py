"""Regression tests for fixed CLI bugs (encoding, Windows, windowing, search, sessions)."""
import os
import subprocess
import sys

import pytest

from ordo import cli, client
from ordo.session import Session
from ordo.textproc import window_file, window_log

from test_cli import FakeModel, run  # noqa: F401  (shared fixtures)
from test_cli import model, ordo_home  # noqa: F401

WIN = pytest.mark.skipif(os.name != "nt", reason="Windows only")


# 1. encoding fallback
def test_decode_falls_back_to_system_codepage():
    text = "Café résumé: naïve façade failed in Ångström"
    assert cli._decode(text.encode("cp1252")) == text
    assert cli._decode(text.encode("utf-8")) == text


# 2. git with accented file names
def test_git_changed_with_accented_filename(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "façade.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "Ångström dir").mkdir()
    (tmp_path / "Ångström dir" / "ñ.txt").write_text("y\n", encoding="utf-8")
    files = cli._git_changed(tmp_path)
    assert "façade.py" in files and any("ñ.txt" in f or f.startswith("Ångström dir") for f in files)
    assert Session.open(cwd=tmp_path).root == tmp_path.resolve()


# 3. a huge line does not empty the window
def test_window_keeps_giant_error_line():
    text = "\n".join(["ok"] * 50 + ["E TypeError: boom " + "x" * 30000 + " END"] + ["ok"] * 50)
    out, cut = window_log(text, 6000)
    assert cut and "E TypeError: boom" in out and "END" in out and "chars omitted" in out
    src = "\n".join(["def f():"] + ["var a=" + "y" * 40000] + ["def target_fn():", "  pass"])
    out, cut = window_file(src, "where is target_fn", 2000)
    assert "3 | def target_fn():" in out


# 4. execution on Windows
def test_command_not_found_is_127(model, capsys):
    code, _ = run(capsys, "run", "-i", "x", "--", "definitely-not-a-command-xyz")
    assert code == 127 and not model.calls


def test_cmd_quote():
    assert cli._cmd_quote("jest") == "jest"
    assert cli._cmd_quote("foo & bar") == '"foo & bar"'
    assert cli._cmd_quote('say "hi"') == '"say ""hi"""'
    assert cli._cmd_quote("a|b") == '"a|b"'


@WIN
def test_bat_args_with_shell_metachars(model, capsys, tmp_path, monkeypatch):
    bat = tmp_path / "echoargs.cmd"
    bat.write_text(f'@"{sys.executable}" -c "import sys; print(sys.argv[1:])" %*\r\n', encoding="utf-8")
    monkeypatch.setenv("PATH", str(tmp_path) + os.pathsep + os.environ["PATH"])
    code, out = run(capsys, "run", "-i", "x", "--", "echoargs", "foo & bar", "a|b")
    assert code == 0 and "'foo & bar'" in out and "'a|b'" in out


def test_cmd_timeout_returns_124(model, capsys):
    code, out = run(capsys, "run", "--cmd-timeout", "1", "-i", "x", "--", sys.executable, "-c",
                    "import time; print('started', flush=True); time.sleep(30)")
    assert code == 124 and "killed after 1.0s timeout" in out and "started" in out


# 5. search
def test_search_invalid_regex_and_missing_path(model, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    assert run(capsys, "search", "foo(bar", str(tmp_path))[0] == 2
    assert run(capsys, "search", "foo", str(tmp_path / "nope"))[0] == 2


def test_search_keeps_backslashes_in_code_and_context(model, capsys, tmp_path, monkeypatch):
    monkeypatch.setattr(cli.shutil, "which", lambda name: None)
    (tmp_path / "a.py").write_text("import re\n\nPAT = re.compile(r'\\d+\\n')\nx = 1\n", encoding="utf-8")
    code, out = run(capsys, "search", "-C", "1", "PAT", str(tmp_path))
    assert code == 0 and r"PAT = re.compile(r'\d+\n')" in out
    assert "a.py-2-" in out and "a.py:3:PAT" in out and "a.py-4-x = 1" in out


def test_normalize_paths_only_prefix():
    line = r"C:\repo\src\a.py:3:PAT = re.compile(r'\d+')"
    assert cli._normalize_paths(line) == r"C:/repo/src/a.py:3:PAT = re.compile(r'\d+')"
    assert cli._normalize_paths(r"src\b.py-7-  x = '\\n'") == r"src/b.py-7-  x = '\\n'"


# 6. OLLAMA_HOST in the server format
@pytest.mark.parametrize("raw,expected", [
    ("0.0.0.0", "http://127.0.0.1:11434"),
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    (":11434", "http://127.0.0.1:11434"),
    ("localhost:9999", "http://localhost:9999"),
    ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
    ("https://ollama.example.com", "https://ollama.example.com:11434"),
    ("[::1]:11434", "http://[::1]:11434"),
    ("[::]", "http://[::1]:11434"),
])
def test_normalize_host(raw, expected):
    assert client.normalize_host(raw) == expected


def test_find_model_installed_from_hf_quickstart(monkeypatch):
    monkeypatch.setattr(client, "installed_models", lambda host: ["llama3:8b", "hf.co/devlucascfarias/ordo-3b:latest"])
    assert client.find_ordo_model("x") == "hf.co/devlucascfarias/ordo-3b:latest"
    monkeypatch.setattr(client, "installed_models", lambda host: ["llama3:8b"])
    assert client.find_ordo_model("x") is None


# 7. filter / compact with a missing file; filter --mode CODE_SLICE
def test_missing_input_files_exit_2(model, capsys, tmp_path):
    assert run(capsys, "filter", "-i", "x", str(tmp_path / "nope.log"))[0] == 2
    assert run(capsys, "compact", "-i", "x", "--from", str(tmp_path / "nope.txt"))[0] == 2


def test_filter_code_slice_numbers_input(model, capsys, tmp_path):
    f = tmp_path / "m.py"
    f.write_text("a = 1\n\fb = 2\ndef g():\n    pass\n", encoding="utf-8")
    model.reply = ("status: FOUND\nsummary: x\nrelevant_symbols:\n- file: m.py\n  lines: '3'\n  symbol: g\n"
                   "  issue: y\nexact_snippets:\n- location: m.py:3\n  code: '3 | def g():'")
    code, out = run(capsys, "filter", "--force", "--mode", "CODE_SLICE", "-i", "where is g", str(f))
    raw = model.calls[-1]["raw"]
    assert raw.splitlines()[0].startswith("// file: ") and "3 | def g():" in raw  # \x0c does not create a line
    assert "unverified" not in out
