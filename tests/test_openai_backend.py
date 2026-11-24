"""OpenAI-compatible backend (llama.cpp, LM Studio, vLLM): a fake server on a local port."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from ordo import cli, client
from ordo.contract import SYSTEM_PROMPT


class FakeOpenAI:
    def __init__(self, models, reply="status: FAIL\nsummary: boom", finish="stop"):
        self.models, self.reply, self.finish = models, reply, finish
        self.requests: list[dict] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep pytest output clean
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models" and fake.models is not None:
                    return self._send(200, {"object": "list", "data": [{"id": m} for m in fake.models]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                fake.requests.append({"path": self.path, "body": body, "auth": self.headers.get("Authorization")})
                if self.path != "/v1/chat/completions":
                    return self._send(404, {"error": "not found"})
                self._send(200, {"choices": [{"message": {"role": "assistant", "content": fake.reply},
                                              "finish_reason": fake.finish}],
                                 "usage": {"prompt_tokens": 321, "completion_tokens": 12}})

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake_server():
    servers = []

    def make(*a, **kw):
        s = FakeOpenAI(*a, **kw)
        servers.append(s)
        return s

    yield make
    for s in servers:
        s.close()


def test_normalize_base_url():
    assert client.normalize_base_url("localhost:8080") == "http://localhost:8080/v1"
    assert client.normalize_base_url("http://h:1234/v1/") == "http://h:1234/v1"
    assert client.normalize_base_url("https://api.example.com/openai/v1") == "https://api.example.com/openai/v1"


def test_call_picks_the_ordo_model_and_sends_the_contract(fake_server, monkeypatch):
    monkeypatch.setenv("ORDO_API_KEY", "test-key")
    srv = fake_server(["qwen2.5-7b", "ordo-3b-q4_k_m"], finish="length")
    res = client.call_openai_ordo("TOOL_OUTPUT", "why?", "log line", base_url=srv.url)
    req = srv.requests[-1]
    assert req["body"]["model"] == "ordo-3b-q4_k_m" and res.model == "ordo-3b-q4_k_m"
    assert req["body"]["temperature"] == 0.0 and req["body"]["max_tokens"] == 1024
    assert req["body"]["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert "<gc_mode>TOOL_OUTPUT</gc_mode>" in req["body"]["messages"][1]["content"]
    assert req["auth"] == "Bearer test-key"
    assert res.text.startswith("status: FAIL") and res.truncated
    assert (res.prompt_tokens, res.output_tokens) == (321, 12)


def test_single_served_model_is_used_and_unknown_is_an_error(fake_server):
    one = fake_server(["model.gguf"])
    assert client.call_openai_ordo("TOOL_OUTPUT", "x", "y", base_url=one.url).model == "model.gguf"
    many = fake_server(["a", "b"])
    with pytest.raises(RuntimeError, match="not served"):
        client.call_openai_ordo("TOOL_OUTPUT", "x", "y", base_url=many.url)


def test_server_without_model_list_gets_the_configured_name(fake_server):
    srv = fake_server(None)
    res = client.call_openai_ordo("TOOL_OUTPUT", "x", "y", base_url=srv.url, model="whatever")
    assert res.model == "whatever" and srv.requests[-1]["body"]["model"] == "whatever"


def test_cli_filter_uses_the_base_url(fake_server, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("ORDO_HOME", str(tmp_path / "home"))
    srv = fake_server(["ordo:3b"], reply="status: FAIL\nsummary: x failed")
    log = tmp_path / "log.txt"
    log.write_text("\n".join(f"line {i}" for i in range(80)) + "\nERROR x failed\n", encoding="utf-8")
    code = cli.main(["filter", "--base-url", srv.url, "-i", "why?", str(log)])
    out = capsys.readouterr().out
    assert code == 0 and "summary: x failed" in out and srv.requests[-1]["path"] == "/v1/chat/completions"
