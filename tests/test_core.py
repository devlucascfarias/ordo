import json
from pathlib import Path

import pytest
import yaml

from ordo.contract import GcMode, build_messages, number_lines
from ordo.schema import CodeSlice, Snippet, ToolOutput, TraceItem, dump_target, parse_output, repair_yaml_output
from ordo.verify import parse_line_spec, verify

ROOT = Path(__file__).resolve().parents[1]
BENCH = json.loads((ROOT / "tests/fixtures/published_predictions.json").read_text(encoding="utf-8"))
OOD = BENCH["true_unseen_ood_10"]["sample_predictions"]
SMOKE_LOG = (ROOT / "tests/fixtures/smoke_pytest_log.txt").read_text(encoding="utf-8")
SMOKE_OUT = (ROOT / "tests/fixtures/smoke_pytest_ordo3b.yaml").read_text(encoding="utf-8")


def test_user_message_format():
    msgs = build_messages("TOOL_OUTPUT", "why", "log")
    assert msgs[0]["role"] == "system" and msgs[0]["content"].startswith("You are Ordo 3B")
    assert msgs[1]["content"] == "<gc_mode>TOOL_OUTPUT</gc_mode>\n<intent>why</intent>\n<raw_input>log</raw_input>"


def test_number_lines():
    assert number_lines("a\nb", start=9) == " 9 | a\n10 | b"


@pytest.mark.parametrize("suite", ["stratified_30", "true_unseen_ood_10"])
def test_published_golds_are_valid(suite):
    for s in BENCH[suite]["sample_predictions"]:
        res = parse_output(s["gold"], GcMode.TOOL_OUTPUT)
        assert res.schema_ok, res.errors


def test_repair_php_unescaped_quotes():
    res = parse_output(OOD[0]["prediction"], "TOOL_OUTPUT")
    assert not res.raw_yaml_ok and res.repaired and res.schema_ok
    assert res.obj.exact_snippets[0].code == " 84 |         RefundRecord::create(['refund_reason' => $payload['reason'] ?? null]);"


def test_repair_ruby_lone_quote_in_summary():
    res = parse_output(OOD[1]["prediction"], "TOOL_OUTPUT")
    assert res.repaired and res.schema_ok
    assert "undefined method `verified?')" in res.obj.summary


def test_repair_elixir_stray_continuation():
    res = parse_output(OOD[2]["prediction"], "TOOL_OUTPUT")
    assert res.repaired and res.schema_ok
    code = res.obj.exact_snippets[0].code
    assert code.splitlines()[-1].startswith("** (KeyError) key :idempotency_key not found")


def test_repair_strips_fences():
    text = "```yaml\nstatus: PASS\nsummary: ok\n```"
    res = parse_output(text, "TOOL_OUTPUT")
    assert not res.raw_yaml_ok and res.repaired and res.schema_ok


def test_repair_keeps_valid_yaml_untouched():
    gold = BENCH["stratified_30"]["sample_predictions"][0]["gold"]
    assert yaml.safe_load(repair_yaml_output(gold)) == yaml.safe_load(gold)


def test_schema_rejects_unknown_status_and_keys():
    assert not parse_output("status: OK\nsummary: x", "TOOL_OUTPUT").schema_ok
    assert not parse_output("status: PASS\nsummary: x\nfix: do this", "TOOL_OUTPUT").schema_ok


def test_literal_backslash_n_snippet_and_lenient_extra_keys():
    text = ("status: FOUND\nsummary: x\nexact_snippets:\n- location: s.py:4-5\n"
            "  code: '4 | def f():\\n5 |     return 1'\nomitted_sections: lines 1-3\n")
    res = parse_output(text, "CODE_SLICE")
    assert not res.schema_ok and res.lenient is not None
    assert res.lenient.exact_snippets[0].code == "4 | def f():\n5 |     return 1"


def test_dump_target_roundtrip_with_quotes_and_multiline():
    obj = ToolOutput(
        status="FAIL",
        summary="Column 'refund_reason' cannot be null: it's `verified?'",
        causal_trace=[TraceItem(file="app/X.php", lines="84", symbol="create", issue="SQLSTATE[23000]: 'x'")],
        exact_snippets=[Snippet(location="app/X.php:84-85", code="84 | a(['k' => $p['id']]);\n85 | b();")],
    )
    text = dump_target(obj)
    res = parse_output(text, "TOOL_OUTPUT")
    assert res.raw_yaml_ok and not res.repaired and res.obj == obj


def test_dump_target_omits_empty_trace_on_pass():
    assert dump_target(ToolOutput(status="PASS", summary="14 passed")) == "status: PASS\nsummary: 14 passed"


def test_parse_line_spec():
    assert parse_line_spec("55") == [55]
    assert parse_line_spec("122-190") == [122, 190]
    assert parse_line_spec("62:5") == [62]


# ---------------------------------------------------------------- verificador


def test_verify_smoke_catches_invented_snippet_line():
    res = parse_output(SMOKE_OUT, "TOOL_OUTPUT")
    assert res.schema_ok
    rep = verify(res.obj, SMOKE_LOG, "TOOL_OUTPUT")
    assert rep.rate("file") == 1.0 and rep.rate("line") == 1.0
    bad = {c.target for c in rep.failures if c.kind == "snippet"}
    # o snippet multilinha veio entre aspas simples: o YAML dobra as quebras de linha em espaços e o
    # código deixa de existir no log (além do '>' inventado antes de `batch = ...`)
    assert bad == {"tests/test_settlement.py:88"}


def test_verify_log_line_formats():
    log = (
        'File "/app/src/svc.py", line 41, in settle\n'
        "   --> src/main.rs:12:5\n"
        "at com.x.SettlementService.confirm(SettlementService.kt:288)\n"
        "   at Api.Handler.Run() in C:\\src\\Api\\Handler.cs:line 42\n"
    )
    obj = ToolOutput(status="FAIL", summary="x", causal_trace=[
        TraceItem(file="src/svc.py", lines="41", issue="a"),
        TraceItem(file="src/main.rs", lines="12", issue="b"),
        TraceItem(file="src/main/kotlin/com/x/SettlementService.kt", lines="288", issue="c"),
        TraceItem(file="src/Api/Handler.cs", lines="42", issue="d"),
        TraceItem(file="src/svc.py", lines="99", issue="inventada"),
    ])
    rep = verify(obj, log, "TOOL_OUTPUT")
    assert [c.target for c in rep.failures] == ["src/svc.py:99"]


def test_verify_pass_rule():
    obj = ToolOutput(status="PASS", summary="ok", causal_trace=[TraceItem(file="a.py", lines="1", issue="x")])
    rep = verify(obj, "a.py:1 ok", "TOOL_OUTPUT")
    assert any(c.kind == "rule" and not c.ok for c in rep.checks)


def test_verify_code_slice_against_file():
    src = "import os\n\n\ndef repair_yaml_output(raw_text):\n    return raw_text\n"
    for raw in ("// file: s.py (5 lines)\n" + src, "// file: s.py (5 lines)\n" + number_lines(src)):
        hallucinated = CodeSlice(status="FOUND", summary="x", relevant_symbols=[
            TraceItem(file="other.py", lines="4", symbol="repair_yaml_output", issue="y")])
        assert [c.kind for c in verify(hallucinated, raw, "CODE_SLICE").failures] == ["file"]
        good = CodeSlice(status="FOUND", summary="x",
                         relevant_symbols=[TraceItem(file="s.py", lines="4-5", symbol="repair_yaml_output", issue="y")],
                         exact_snippets=[Snippet(location="s.py:4", code="4 | def repair_yaml_output(raw_text):")])
        assert verify(good, raw, "CODE_SLICE").ok
        wrong = good.model_copy(update={
            "relevant_symbols": [TraceItem(file="s.py", lines="1-2", symbol="repair_yaml_output", issue="y")],
            "exact_snippets": [Snippet(location="s.py:3", code="3 | def repair_yaml_output(raw_text):")],
        })
        kinds = {c.kind for c in verify(wrong, raw, "CODE_SLICE").failures}
        assert kinds == {"symbol", "snippet"}
