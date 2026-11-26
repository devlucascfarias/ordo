"""Deterministic pass-summary rules for known test runners.

When a test command exits 0 and its output ends with a recognized passing summary
from a known runner (zero failures), `pass_summary` returns a one-line summary so
`ordo run` can skip the local model completely.
"""
from __future__ import annotations

import re

# pytest: "===== 121 passed, 120 warnings in 2.5s =====", "5 passed, 2 skipped in 0.1s"
_PYTEST_LINE = re.compile(
    r"^\s*=*\s*(?P<counts>\d+\s+[a-zA-Z]+(?:\s*,\s*\d+\s+[a-zA-Z]+)*)"
    r"\s+in\s+\d+(?:\.\d+)?s(?:\s+\([^)]+\))?\s*=*\s*$"
)
_PYTEST_PASSED = re.compile(r"\b\d+\s+passed\b", re.IGNORECASE)
_PYTEST_FAIL = re.compile(r"\b[1-9]\d*\s+(?:failed|errors?)\b", re.IGNORECASE)
_PYTEST_KEYWORDS = re.compile(
    r"\b(?:passed|failed|errors?|skipped|xfailed|xpassed|warnings?|deselected|rerun)\b",
    re.IGNORECASE,
)

# node:test spec/TAP reporter:
#   ℹ tests 351 / ℹ pass 351 / ℹ fail 0
#   # tests 351 / # pass 351 / # fail 0
_NODE_TEST_LINE = re.compile(
    r"^\s*(?:ℹ|#)\s+(?P<key>tests|suites|pass|fail|cancelled|skipped|todo|duration_ms)\s+(?P<val>\d+(?:\.\d+)?)\s*$"
)

# jest: "Tests:       12 passed, 12 total"
_JEST_TESTS = re.compile(r"^\s*Tests:\s+(?P<counts>.+?\b\d+\s+total)\s*$")
_JEST_SUITES = re.compile(r"^\s*Test Suites:\s+(?P<counts>.+?\b\d+\s+total)\s*$")
_NONZERO_FAIL = re.compile(r"\b[1-9]\d*\s+failed\b", re.IGNORECASE)
_HAS_PASSED = re.compile(r"\b\d+\s+passed\b", re.IGNORECASE)

# vitest: "Tests  12 passed (12)" or "Tests  10 passed | 2 skipped (12)"
_VITEST_TESTS = re.compile(
    r"^\s*Tests\s{2,}(?P<counts>\d+\s+(?:passed|failed|skipped|todo)(?:\s*\|\s*\d+\s+[a-zA-Z]+)*\s*\(\d+\))\s*$"
)
_VITEST_FILES = re.compile(
    r"^\s*Test Files\s{2,}(?P<counts>\d+\s+(?:passed|failed|skipped|todo)(?:\s*\|\s*\d+\s+[a-zA-Z]+)*\s*\(\d+\))\s*$"
)

# go test: "ok  \tpkg\t0.12s" (or "(cached)") with no FAIL line anywhere
_GO_OK = re.compile(r"^ok\s+(?P<pkg>\S+)\s+(?:\d+(?:\.\d+)?s|\(cached\))(?:\s+.*)?$")
_GO_FAIL = re.compile(r"^\s*(?:FAIL\b|---\s+FAIL:)")

# cargo test: "test result: ok. 10 passed; 0 failed; ..."
_CARGO_RESULT = re.compile(
    r"^\s*test result:\s+(?P<status>ok|FAILED)\.\s+(?P<passed>\d+)\s+passed;\s+(?P<failed>\d+)\s+failed(?P<rest>.*)$"
)
_CARGO_IGNORED = re.compile(r"\b(\d+)\s+ignored\b")

# unittest: "Ran 12 tests in 0.01s" followed by "OK" (or "OK (skipped=2)")
_UNITTEST_RAN = re.compile(r"^\s*Ran\s+(?P<count>\d+)\s+tests?\s+in\s+\d+(?:\.\d+)?s\s*$")
_UNITTEST_OK = re.compile(r"^\s*OK(?:\s+\((?P<extra>[^)]+)\))?\s*$")
_UNITTEST_FAIL = re.compile(r"^\s*FAILED\b")


def _norm_spaces(text: str) -> str:
    return " ".join(text.split())


def _match_pytest(lines: list[str]) -> tuple[int, str | None] | None:
    last: tuple[int, str | None] | None = None
    for idx, line in enumerate(lines):
        m = _PYTEST_LINE.match(line)
        if not m:
            continue
        counts = _norm_spaces(m.group("counts"))
        if not _PYTEST_KEYWORDS.search(counts):
            continue
        if _PYTEST_FAIL.search(counts) or not _PYTEST_PASSED.search(counts):
            last = (idx, None)
        else:
            last = (idx, f"pytest: {counts}")
    return last


def _match_node_test(lines: list[str]) -> tuple[int, str | None] | None:
    last: tuple[int, str | None] | None = None
    i = 0
    n = len(lines)
    while i < n:
        m = _NODE_TEST_LINE.match(lines[i])
        if not m:
            i += 1
            continue
        block: dict[str, str] = {}
        end_idx = i
        while i < n:
            m2 = _NODE_TEST_LINE.match(lines[i])
            if not m2:
                break
            block[m2.group("key")] = m2.group("val")
            end_idx = i
            i += 1
        if "pass" in block and "fail" in block:
            passed = int(float(block["pass"]))
            failed = int(float(block["fail"]))
            cancelled = int(float(block.get("cancelled", "0")))
            skipped = int(float(block.get("skipped", "0")))
            if failed != 0 or cancelled != 0:
                last = (end_idx, None)
            else:
                extra = f", {skipped} skipped" if skipped else ""
                last = (end_idx, f"node:test: {passed} passed, 0 failed{extra}")
    return last


def _match_jest(lines: list[str]) -> tuple[int, str | None] | None:
    last: tuple[int, str | None] | None = None
    suites_failed = False
    for idx, line in enumerate(lines):
        ms = _JEST_SUITES.match(line)
        if ms:
            suites_failed = bool(_NONZERO_FAIL.search(ms.group("counts")))
            continue
        mt = _JEST_TESTS.match(line)
        if not mt:
            continue
        counts = _norm_spaces(mt.group("counts"))
        if suites_failed or _NONZERO_FAIL.search(counts) or not _HAS_PASSED.search(counts):
            last = (idx, None)
        else:
            last = (idx, f"jest: {counts}")
        suites_failed = False
    return last


def _match_vitest(lines: list[str]) -> tuple[int, str | None] | None:
    last: tuple[int, str | None] | None = None
    files_failed = False
    for idx, line in enumerate(lines):
        mf = _VITEST_FILES.match(line)
        if mf:
            files_failed = bool(_NONZERO_FAIL.search(mf.group("counts")))
            continue
        mt = _VITEST_TESTS.match(line)
        if not mt:
            continue
        counts = _norm_spaces(mt.group("counts"))
        if files_failed or _NONZERO_FAIL.search(counts) or not _HAS_PASSED.search(counts):
            last = (idx, None)
        else:
            last = (idx, f"vitest: {counts}")
        files_failed = False
    return last


def _match_go_test(lines: list[str]) -> tuple[int, str | None] | None:
    ok_pkgs: list[str] = []
    last_ok_idx = -1
    last_fail_idx = -1
    for idx, line in enumerate(lines):
        if _GO_FAIL.match(line):
            last_fail_idx = idx
            continue
        m = _GO_OK.match(line)
        if m:
            ok_pkgs.append(m.group("pkg"))
            last_ok_idx = idx
    if not ok_pkgs:
        return None
    if last_fail_idx != -1:
        return (max(last_ok_idx, last_fail_idx), None)
    if len(ok_pkgs) == 1:
        return (last_ok_idx, f"go test: 1 package passed ({ok_pkgs[0]})")
    return (last_ok_idx, f"go test: {len(ok_pkgs)} packages passed")


def _match_cargo_test(lines: list[str]) -> tuple[int, str | None] | None:
    last_idx = -1
    last_summary: str | None = None
    any_failed = False
    for idx, line in enumerate(lines):
        m = _CARGO_RESULT.match(line)
        if not m:
            continue
        last_idx = idx
        status = m.group("status")
        passed = int(m.group("passed"))
        failed = int(m.group("failed"))
        if status != "ok" or failed != 0:
            any_failed = True
            last_summary = None
        else:
            ign = _CARGO_IGNORED.search(m.group("rest"))
            ignored = int(ign.group(1)) if ign else 0
            extra = f", {ignored} ignored" if ignored else ""
            last_summary = f"cargo test: {passed} passed, 0 failed{extra}"
    if last_idx == -1:
        return None
    return (last_idx, None if any_failed else last_summary)


def _match_unittest(lines: list[str]) -> tuple[int, str | None] | None:
    last: tuple[int, str | None] | None = None
    n = len(lines)
    for idx, line in enumerate(lines):
        m = _UNITTEST_RAN.match(line)
        if not m:
            continue
        count = int(m.group("count"))
        j = idx + 1
        while j < n and not lines[j].strip():
            j += 1
        if j >= n:
            continue
        mok = _UNITTEST_OK.match(lines[j])
        if mok:
            unit = "test" if count == 1 else "tests"
            extra = f" ({mok.group('extra')})" if mok.group("extra") else ""
            last = (j, f"unittest: {count} {unit} OK{extra}")
        elif _UNITTEST_FAIL.match(lines[j]):
            last = (j, None)
    return last


_MATCHERS = (
    _match_pytest,
    _match_node_test,
    _match_jest,
    _match_vitest,
    _match_go_test,
    _match_cargo_test,
    _match_unittest,
)


# questions a bare "N passed" does not answer: the model reads the log for these
_SPECIFIC_INTENT = re.compile(
    r"warn|deprecat|slow|durat|how long|timing|skip|xfail|flak|coverage|print|stdout|stderr|log line|output of",
    re.IGNORECASE,
)


def _failing_block(lines: list[str]) -> bool:
    """A failing runner summary anywhere (a monorepo script can exit 0 after one package failed)."""
    for line in lines:
        m = _PYTEST_LINE.match(line)
        if m and _PYTEST_FAIL.search(m.group("counts")):
            return True
        m = _NODE_TEST_LINE.match(line)
        if m and m.group("key") in ("fail", "cancelled") and float(m.group("val")) > 0:
            return True
        if (_JEST_TESTS.match(line) or _VITEST_TESTS.match(line)) and _NONZERO_FAIL.search(line):
            return True
        if _GO_FAIL.match(line) or _UNITTEST_FAIL.match(line):
            return True
        m = _CARGO_RESULT.match(line)
        if m and (m.group("status") != "ok" or m.group("failed") != "0"):
            return True
    return False


def pass_summary(text: str, code: int, intent: str = "") -> str | None:
    """Return a one-line summary when `code == 0` and `text` ends with a known passing test summary.

    None (the model decides) when the exit code is not 0, no known runner summary is found, any runner block in
    the output reports failures, or the intent asks something a pass count does not answer (warnings, timing...).
    """
    if code != 0 or not text or not text.strip() or _SPECIFIC_INTENT.search(intent or ""):
        return None
    lines = text.splitlines()
    candidates = [res for matcher in _MATCHERS if (res := matcher(lines)) is not None]
    if not candidates or _failing_block(lines):
        return None
    _, summary = max(candidates, key=lambda item: item[0])
    if summary is None:
        return None
    blocks = sum(bool(_PYTEST_LINE.match(l)) or bool(re.match(r"^\s*(?:ℹ|#)\s+pass\s+\d", l)) for l in lines)
    return summary + (f" (last of {blocks} runs, all passing)" if blocks > 1 else "")
