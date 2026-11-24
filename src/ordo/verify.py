"""Deterministic grounding check: everything Ordo cites must be present in the <raw_input>.

The CLI uses it to warn the agent about citations that are not in the input.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .contract import GcMode
from .schema import CodeSlice, MemoryCompact, ToolOutput, _Strict

_NUMBERED = re.compile(r"^\s*(\d+)\s*\|\s?(.*)$")
_RANGE = re.compile(r"\d+")


@dataclass
class Check:
    kind: str  # file | line | snippet | symbol | rule | summary
    target: str
    ok: bool
    detail: str = ""


@dataclass
class VerifyReport:
    checks: list[Check] = field(default_factory=list)

    def add(self, kind: str, target: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(kind, target, ok, detail))

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def rate(self, kind: str) -> float | None:
        cs = [c for c in self.checks if c.kind == kind]
        return sum(c.ok for c in cs) / len(cs) if cs else None


def norm_path(p: str) -> str:
    return p.strip().strip("`'\"").replace("\\", "/")


def parse_line_spec(spec: str) -> list[int]:
    """'55' -> [55]; '122-190' -> [122, 190]; '62,64' -> [62, 64]; '62:5' -> [62]."""
    spec = spec.split(":")[0] if re.fullmatch(r"\d+:\d+", spec.strip()) else spec
    return [int(n) for n in _RANGE.findall(spec)]


def split_numbered(code: str) -> list[tuple[int | None, str]]:
    """Snippet lines: (number, content) when in the `NN | code` format, otherwise (None, line)."""
    out = []
    for line in code.splitlines():
        m = _NUMBERED.match(line)
        out.append((int(m.group(1)), m.group(2)) if m else (None, line))
    return out


def file_lines(raw_input: str) -> list[str]:
    """File lines in CODE_SLICE (accepts input already numbered as `NN | ...`)."""
    lines = raw_input.split("\n")
    if lines and lines[0].startswith("// file: "):  # header the CLI adds in CODE_SLICE
        lines = lines[1:]
    if lines and lines[-1] == "":
        lines.pop()
    numbered = [_NUMBERED.match(l) for l in lines]
    if lines and sum(bool(m) for m in numbered) >= 0.9 * len(lines):
        return [m.group(2) if m else l for m, l in zip(numbered, lines)]
    return lines


# ---------------------------------------------------------------- anchors in logs


def _file_in(raw: str, path: str) -> bool:
    p = norm_path(path)
    return p in raw or p.rsplit("/", 1)[-1] in raw


def _line_in_log(raw: str, path: str, n: int) -> bool:
    """The line number appears in the log next to the file (formats of several tools)."""
    base = re.escape(norm_path(path).rsplit("/", 1)[-1])
    patterns = [
        rf"{base}:{n}\b",  # py/go/rust/ts/eslint/gcc: file:N[:col]
        rf"{base}\"?, line {n}\b",  # python traceback
        rf"{base}\({n}[,)]",  # c#/msbuild; kotlin/java (File.kt:N) is covered above
        rf"{base}:line {n}\b",  # .NET stack
        rf"{base}\s+line {n}\b",
        rf"{base}:\[{n},\d+\]",  # maven/javac: File.java:[17,30]
    ]
    if any(re.search(p, raw) for p in patterns):
        return True
    for m in re.finditer(base, raw):  # "line N" right after a mention of the file
        if re.search(rf"\bline:? {n}\b", raw[m.end(): m.end() + 200]):
            return True
        # eslint/stylelint: file in a header, then "  N:col  error ..." until the next header
        block = re.split(r"\n\s*\n", raw[m.end(): m.end() + 20000], maxsplit=1)[0]
        if re.search(rf"(?m)^\s+{n}:\d+\s", block):
            return True
        # jest/vitest/rust: code frame "> N |" or " N |" right below the mention
        if re.search(rf"(?m)^\s*>?\s*{n}\s*\|", raw[m.end(): m.end() + 2000]):
            return True
    return False


def _trivial(code_line: str) -> bool:
    """Lines that exist in any file (`}`, `return`, `end`) prove nothing."""
    s = code_line.strip()
    return len(re.sub(r"[\W_]", "", s)) < 4 or s in ("return", "pass", "break", "continue", "else:", "try:")


def _snippet_in_log(raw: str, code: str) -> list[str]:
    """Snippet lines that do NOT appear verbatim in the log (ignoring the `NN | ` prefix)."""
    raw_lines = {l.strip() for l in raw.splitlines()}
    missing = []
    # the log reached us with `\` turned into `/` (Windows paths); compare the snippet the same way
    lines = [content.strip() for _, content in split_numbered(code.replace("\\", "/"))]
    meaningful = [c for c in lines if c and not _trivial(c)]
    if not meaningful and any(lines):
        return ["(snippet has only trivial lines)"]
    for c in meaningful:
        if c not in raw and c not in raw_lines:
            missing.append(c)
    return missing


# ---------------------------------------------------------------- per mode


def _check_trace(rep: VerifyReport, raw: str, items: Iterable, *, log: bool, flines: list[str] | None) -> None:
    for it in items:
        f = norm_path(it.file)
        rep.add("file", f, _file_in(raw, f))  # in CODE_SLICE the `// file:` header carries the name
        nums = parse_line_spec(it.lines)
        for n in nums:
            if flines is not None:
                rep.add("line", f"{f}:{n}", 1 <= n <= len(flines), "outside the file")
            elif log:
                rep.add("line", f"{f}:{n}", _line_in_log(raw, f, n))
        if flines is not None and it.symbol and len(nums) >= 1:
            lo, hi = nums[0], nums[-1]
            parts = [p for p in re.split(r"[.#:/\s]+", it.symbol.split("(")[0]) if p]
            name = parts[-1] if parts else ""
            window = "\n".join(flines[max(lo - 1, 0): hi])
            rep.add("symbol", it.symbol, bool(name) and name in window, f"lines {lo}-{hi}")


def _check_snippets_log(rep: VerifyReport, raw: str, snippets) -> None:
    for s in snippets:
        missing = _snippet_in_log(raw, s.code)
        rep.add("snippet", s.location, not missing, "; ".join(missing[:2]))


def _check_snippets_file(rep: VerifyReport, flines: list[str], snippets) -> None:
    for s in snippets:
        bad = []
        for n, content in split_numbered(s.code):
            if not content.strip():
                continue
            if n is None:
                if content.strip() not in (l.strip() for l in flines):
                    bad.append(content.strip())
            elif not (1 <= n <= len(flines)) or flines[n - 1].strip() != content.strip():
                bad.append(f"{n}: {content.strip()}")
        rep.add("snippet", s.location, not bad, "; ".join(bad[:2]))


_CITE = re.compile(r"(?<![\w/:.])([\w.-]*[/\\]?[\w./\\-]*\.[A-Za-z]{1,6}):(\d+)\b")


def _check_strict(rep: VerifyReport, raw: str, summary: str, items: Iterable, *, log: bool,
                  flines: list[str] | None) -> None:
    """Checks the CLI adds for the agent (kept out of the benchmark's grounding metric for comparability)."""
    bad = [it for it in items if not _RANGE.search(str(it.lines))]
    if len(bad) <= 2:
        for it in bad:
            rep.add("rule", f"{norm_path(it.file)}: lines is not a line number ({it.lines!r})", False)
    else:  # one warning, not one per item
        rep.add("rule", f"{len(bad)} items have no line number in `lines` (e.g. {bad[0].lines!r})", False)
    for path, n in _CITE.findall(summary or ""):
        if path.startswith(("http", "www")) or not re.search(r"[A-Za-z]", path.rsplit(".", 1)[0]):
            continue  # URLs, version numbers, durations
        p, n = norm_path(path), int(n)
        ok = _file_in(raw, p) and (1 <= n <= len(flines) if flines is not None else _line_in_log(raw, p, n))
        rep.add("summary", f"{p}:{n}", ok)


def verify(obj: _Strict, raw_input: str, mode: GcMode | str, *, source_lines: list[str] | None = None,
           strict: bool = False) -> VerifyReport:
    """`source_lines`: in CODE_SLICE, the full file on disk (when raw_input is only an excerpt).

    `strict` (the CLI): also flag `lines` without a line number and `path:line` cited in the summary that
    is not in the input.
    """
    mode = GcMode(mode)
    rep = VerifyReport()
    raw = raw_input.replace("\\", "/")
    if isinstance(obj, ToolOutput):
        if obj.status == "PASS":
            rep.add("rule", "PASS without trace", not obj.causal_trace and not obj.exact_snippets)
        _check_trace(rep, raw, obj.causal_trace, log=True, flines=None)
        _check_snippets_log(rep, raw, obj.exact_snippets)
        if strict:
            _check_strict(rep, raw, obj.summary, obj.causal_trace, log=True, flines=None)
    elif isinstance(obj, CodeSlice):
        flines = source_lines if source_lines is not None else file_lines(raw_input)
        if obj.status == "NOT_FOUND":
            rep.add("rule", "NOT_FOUND without symbols", not obj.relevant_symbols and not obj.exact_snippets)
        _check_trace(rep, raw, obj.relevant_symbols, log=False, flines=flines)
        _check_snippets_file(rep, flines, obj.exact_snippets)
        if strict:
            _check_strict(rep, raw, obj.summary, obj.relevant_symbols, log=False, flines=flines)
    elif isinstance(obj, MemoryCompact):
        for fs in obj.modified_files_state:
            rep.add("file", fs.file, _file_in(raw, fs.file))
        for fact in obj.confirmed_facts:
            for path in re.findall(r"[\w./-]+\.\w{1,5}(?=[:)\s]|$)", fact):
                if "/" in path:
                    rep.add("file", path, _file_in(raw, path), "cited in confirmed_facts")
    return rep
