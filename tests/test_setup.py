"""`ordo setup` against a fake Ollama server: missing model, download with --yes, test call, unreachable server."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ordo import cli, client

REPLY = "status: FAIL\nsummary: 'KeyError: ''BRL'' in src/fx.py:12'\ncausal_trace:\n- file: src/fx.py\n  lines: '12'\n  issue: KeyError"


class FakeOllama:
    def __init__(self, models):
        self.models, self.pulled, self.chats = list(models), [], []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, body):
                data = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/api/tags":
                    return self._json({"models": [{"name": m} for m in fake.models]})
                self.send_response(404)
                self.end_headers()

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path == "/api/pull":
                    fake.pulled.append(body["model"])
                    events = [{"status": "pulling manifest"},
                              {"status": "pulling abc", "total": 100, "completed": 50},
                              {"status": "pulling abc", "total": 100, "completed": 100},
                              {"status": "success"}]
                    data = "".join(json.dumps(e) + "\n" for e in events).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    fake.models.append(body["model"] + ":latest")
                    return
                if self.path == "/api/chat":
                    fake.chats.append(body["model"])
                    return self._json({"message": {"content": REPLY}, "prompt_eval_count": 200, "eval_count": 30,
                                       "total_duration": 1e9, "done_reason": "stop"})
                self.send_response(404)
                self.end_headers()

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()


@pytest.fixture
def ollama(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDO_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("ORDO_BASE_URL", raising=False)
    servers = []

    def make(models):
        s = FakeOllama(models)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.server.shutdown()


def _run(capsys, *argv):
    code = cli.main(["setup", *argv])
    return code, capsys.readouterr().out


def test_ready_server_and_model_pass_the_test_call(ollama, capsys):
    srv = ollama(["ordo:3b"])
    code, out = _run(capsys, "--host", srv.url)
    assert code == 0 and "[ok] model: ordo:3b" in out and "[ok] test call" in out and "cites src/fx.py" in out


def test_missing_model_is_not_downloaded_without_consent(ollama, capsys):
    srv = ollama(["llama3:8b"])
    code, out = _run(capsys, "--host", srv.url, "--no-test")  # not a TTY: no prompt, no download
    assert code == 1 and "[fail] model: Ordo is not installed" in out and "ordo setup --yes" in out
    assert srv.pulled == []


def test_yes_downloads_the_model_then_tests_it(ollama, capsys):
    srv = ollama([])
    code, out = _run(capsys, "--host", srv.url, "--yes")
    assert code == 0 and srv.pulled == [client.PULL_NAME]
    assert f"[ok] model: {client.PULL_NAME}:latest" in out and "[ok] test call" in out


def test_unreachable_server_and_json_report(ollama, capsys):
    code, out = _run(capsys, "--host", "http://127.0.0.1:9", "--json")
    rep = json.loads(out)
    assert code == 1 and rep["ok"] is False and rep["backend"] == "ollama" and rep["hints"]


def test_colors_only_at_an_interactive_terminal(ollama, capsys, monkeypatch):
    import io
    import re
    import sys

    srv = ollama(["ordo:3b"])
    code, plain = _run(capsys, "--host", srv.url)  # captured: no escape codes
    assert code == 0 and "\x1b[" not in plain

    class TTY(io.StringIO):
        def isatty(self):
            return True

    tty = TTY()
    monkeypatch.setattr(sys, "stdout", tty)
    assert cli.main(["setup", "--host", srv.url]) == 0
    colored = tty.getvalue()
    strip = lambda t: re.sub(r"test call: [\d.]+s", "test call: Xs", re.sub(r"\x1b\[[0-9;]*m", "", t))  # noqa: E731
    assert "\x1b[" in colored and strip(colored) == strip(plain)


HF = "hf.co/devlucascfarias/ordo-3b:latest"


def test_hugging_face_pull_is_preferred_over_an_older_local_build(ollama, capsys, monkeypatch):
    monkeypatch.delenv("ORDO_MODEL", raising=False)
    srv = ollama(["ordo:3b", HF])
    code, out = _run(capsys, "--host", srv.url)
    assert code == 0 and f"[ok] model: {HF}" in out and srv.chats == [HF]


def test_a_named_model_is_used_as_is(ollama, capsys, monkeypatch):
    monkeypatch.setenv("ORDO_MODEL", "ordo:3b")
    srv = ollama(["ordo:3b", HF])
    code, out = _run(capsys, "--host", srv.url)
    assert code == 0 and "[ok] model: ordo:3b" in out and srv.chats == ["ordo:3b"]


def test_client_default_resolves_to_the_preferred_install(ollama, monkeypatch):
    from ordo import client
    monkeypatch.delenv("ORDO_MODEL", raising=False)
    monkeypatch.setattr(client, "_resolved", {})
    srv = ollama(["ordo:3b", HF])
    client.call_ollama_ordo("TOOL_OUTPUT", "why?", "log", host=srv.url)
    assert srv.chats == [HF]
