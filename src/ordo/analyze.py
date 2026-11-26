"""Deterministic reading of a failing command's output: no model, no network, a few milliseconds.

`analyze(text, command, cwd)` finds the distinct failures (compiler errors, browser console errors, failing
tests), separates causes from symptoms, and links each one to the source when the repository can tell:

- tsc cascades: many "'x' does not exist on type 'T'" errors become one cause at the definition of T;
- GLSL compile errors: where the undeclared identifier is declared and used, and under which #ifdef;
- NaN: the earlier log lines where NaN first shows up;
- missing assets (404, ENOENT): the closest existing file name in the tree.

The model is only needed when nothing here is recognized (an unknown log format) or when the question asks
something a list of failures does not answer.
"""
from __future__ import annotations

import difflib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .textproc import shorten_line

# ---------------------------------------------------------------- runners

_BY_COMMAND = [
    (r"\bplaywright\b", "playwright"), (r"\bvitest\b", "vitest"), (r"\bjest\b", "jest"),
    (r"\bnode\s+--test\b", "node:test"), (r"\bpy(?:thon\d?\s+-m\s+)?\.?test\b|\bpytest\b", "pytest"),
    (r"\bgo\s+test\b", "go test"), (r"\bcargo\s+test\b", "cargo test"), (r"\bcargo\s+(?:build|check)\b", "cargo"),
    (r"\btsc\b", "tsc"), (r"\beslint\b", "eslint"), (r"\bmypy\b", "mypy"), (r"\bruff\b", "ruff"),
    (r"\bunittest\b", "unittest"), (r"\bvite\s+build\b", "vite build"),
]
_BY_OUTPUT = [
    (r"^> playwright test|^Running \d+ tests? using \d+ workers?", "playwright"),
    (r"^> tsc\b|error TS\d{4}:", "tsc"),
    (r"^> vite build|^vite v\d", "vite build"),
    (r"^=+ test session starts =+", "pytest"),
    (r"^\s*(?:ℹ|#) (?:tests|pass|fail) \d+", "node:test"),
    (r"^\s*Test Suites: ", "jest"),
    (r"^\s*Test Files\s{2,}", "vitest"),
    (r"^(?:ok|FAIL)\s+\S+\s+[\d.]+s$|^--- FAIL: ", "go test"),
    (r"^test result: ", "cargo test"),
    (r"THREE\.WebGLProgram|WebGL", "webgl"),
]


def detect_runners(command: str | None, text: str) -> list[str]:
    """Tools that produced the output: from the command first, then from signature lines (npm scripts)."""
    found: list[str] = []
    for rx, name in _BY_COMMAND:
        if command and re.search(rx, command) and name not in found:
            found.append(name)
    head = text[:200_000]
    for rx, name in _BY_OUTPUT:
        if name not in found and re.search(rx, head, re.MULTILINE):
            found.append(name)
    return found


# ---------------------------------------------------------------- failures

_CAUSE = [
    re.compile(r"^(?:\[browser:error\]\s*)?ERROR: \d+:\d+: .+"),                 # GLSL compiler
    re.compile(r"^\S+\(\d+,\d+\): error TS\d{4}: .+"),                             # tsc
    re.compile(r"^\S+:\d+:\d+ - error TS\d{4}: .+"),                               # tsc --pretty
    re.compile(r"^\s*error(?:\[E\d+\])?: .+"),                                     # rustc and friends
    re.compile(r"^\s*\S+\.(?:go|java|kt|cs|cpp|c|h):\d+(?::\d+)?: (?:error|fatal).+"),
    re.compile(r"^\S+\.py:\d+: \w*(?:Error|Exception)\b.*"),                       # python crash location
    re.compile(r"^\[browser:error\] .+"),                                          # browser console
    re.compile(r"^\s*(?:Uncaught |Unhandled )?\w*(?:Error|Exception): (?!expect\().+"),  # JS/Python exception line
    re.compile(r"^thread '[^']+' panicked at \S+:\d+"),                            # rust panic
    re.compile(r"^panic: .+"),                                                     # go panic
    re.compile(r"^_+ ERROR collecting \S+ _+$"),                                   # pytest collection error
    re.compile(r"^=+\d*=+\s*ERROR: \w+Sanitizer: .+"),                             # ASan/UBSan
    # compiler/linter line that starts with a path and a line: go build, dotnet, mvn, mypy, ruff, gcc, eslint unix
    re.compile(r"^(?:\[(?:ERROR|error)\]\s+)?(?:[A-Za-z]:)?[\w./\\@-]+\.[A-Za-z]{1,5}"
               r"(?::\[?\d+(?:[,:]\d+)?\]?|\(\d+(?:,\d+)?\)):?\s+(?!note:)(?!\s*PASSED)\S.+"),
]
_SYMPTOM = [
    re.compile(r"^\s*(?:x|✘|×)\s+(?:\d+\s+)?\S+.*›.+"),                          # playwright list reporter
    re.compile(r"^\s*✖ (?!failing tests:).+"),                                     # node:test spec
    re.compile(r"^\s*● .+›.+"),                                                    # jest
    re.compile(r"^FAILED \S+.*"),                                                  # pytest short summary
    re.compile(r"^\s*--- FAIL: .+"),                                               # go test
    re.compile(r"^\s+\d+\) [A-Z][\w:#. ]+"),                                       # rspec / exunit / phpunit
    re.compile(r"^\s*(?:FAIL|×)\s+\S+\.\w{1,4}\s+>\s+.+"),                            # vitest
]
_DURATION = re.compile(r"\s*\((?:\d+(?:\.\d+)?\s*(?:ms|s|m))\)\s*$|\s+\+\d+ms\b")
_PREFIX = re.compile(r"^(?:\[browser:error\]\s*|Error:\s*|Uncaught\s+)+")
_TAIL_LOCATION = re.compile(r"\s*\((?:https?|file)://[^)]*\)\s*$")
_LOC = re.compile(r"((?:[A-Za-z]:)?[\w./\\@-]*[\w-]\.(?:py|tsx?|jsx?|mjs|cjs|vue|svelte|go|rs|java|kt|cs|cpp|cc|c|h|rb|php|glsl|vert|frag))"
                  r"(?::(\d+)(?::\d+)?|\((\d+),\d+\)|:\[(\d+),\d+\])")
_FRAMES = [
    re.compile(r"^\s*-->\s+([^\s:]+):(\d+)"),                                       # rustc
    re.compile(r"❯\s+((?:[A-Za-z]:)?[^\s:]+\.\w{1,5}):(\d+)"),                          # vitest
    re.compile(r"\bpanicked at ([^\s:]+):(\d+)"),                                   # rust
    re.compile(r"^\s*#\s*\./?([^:\s]+):(\d+):in\b"),                               # ruby
    re.compile(r"#\d+ 0x[0-9a-f]+ in \S+ ([^\s:]+):(\d+)"),                         # asan
    re.compile(r'^\s*File "([^"]+)", line (\d+)'),                                   # python
    re.compile(r"^\s+([\w./-]+\.(?:go|exs?|php|rb|py|rs|cs|java|kt|ts|js)):(\d+)"),  # go test, exunit, phpunit
    re.compile(r"\bat (?:[\w.$<>\[\]/ ]+? )?\(?(?:file:///)?((?:[A-Za-z]:)?[^\s():]+\.\w{1,5}):(\d+)(?::\d+)?\)?\s*$"),  # js
    re.compile(r"\bat [\w.$]+\(([\w.]+\.(?:java|kt|scala)):(\d+)\)"),               # jvm
    re.compile(r"\bin ((?:[A-Za-z]:)?[\w./\\-]+\.cs):line (\d+)"),                  # dotnet
    re.compile(r"^file:///([^\s]+?):(\d+)\s*$"),                                   # node ESM, the line above the error
]
_TEST_FILE = re.compile(r"(?:^|/)(?:tests?|spec|__tests__)/|(?:^|/)test_[^/]+\.py$|_test\.(?:go|py|exs)$|"
                        r"\.(?:test|spec)\.[jt]sx?$|_spec\.rb$|Tests?\.(?:cs|java|kt)$|(?:^|/)tests/[^/]+\.rs$")
_NOT_OWN = re.compile(r"node_modules|node:internal|^internal/|/usr/(?:local/)?(?:go|lib)/|site-packages|"
                      r"dist-packages|<anonymous>|/rustc/|\.cargo/registry|/gems/|vendor/|testing\.go|"
                      r"jest-circus|ExUnit\.|PHPUnit\\|\bjunit\b|xunit", re.IGNORECASE)
_FILE_URL = re.compile(r"file:///([^\s)]+?):(\d+)(?::\d+)?")


_NOISE_CTX = re.compile(r"^\s*\[browser:(?:log|info|debug|warning)\]|^\s*[+-]\s|^\s*Error Context:|^\s*\d+ \|")
_CTX_STOP = re.compile(r"^\s*(?:={3,}|_{3,}|-{3,}|ℹ |# (?:tests|pass|fail|suites)\b|> \S)")
_PY_FRAME = re.compile(r'^\s*File "([^"]+)", line (\d+)')
_PY_CRASH = re.compile(r"^\S+\.py:\d+: \w*(?:Error|Exception)\b")
_ASSERTION = re.compile(r"expect\(|Expected|Received|AssertionError|assert ")


@dataclass
class Finding:
    kind: str  # cause | symptom
    text: str
    key: str
    file: str | None = None
    line: int | None = None
    context: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    count: int = 1
    also_at: list[str] = field(default_factory=list)  # the test (or caller) frame when the failure lives elsewhere


def _key(line: str) -> str:
    s = _TAIL_LOCATION.sub("", _PREFIX.sub("", line.strip()))
    s = _DURATION.sub("", s)
    s = re.sub(r"^(?:x|✘|×)\s+\d+\s+", "", s)  # playwright test index
    return re.sub(r"\s+", " ", s).strip()


_PASSING = re.compile(r"\b(?:ok|pass(?:ed|es)?|success(?:ful)?|done)\b", re.IGNORECASE)
_ESLINT_ROW = re.compile(r"^\s+(\d+):(\d+)\s+error\s+(.+?)\s*$")
_PATH_ALONE = re.compile(r"^((?:[A-Za-z]:)?[/\\]?[\w.@-]+(?:[/\\][\w.@-]+)*\.[A-Za-z]{1,5})\s*$")


_NOT_HEADLINE = re.compile(r"^\S+\.py:\d+: in \S+|^error: could not compile\b|^error: aborting due to|"
                           r"^For more information about|^error: test failed, to rerun")


def _classify(line: str) -> str | None:
    if re.match(r"^\s*✖ \d+ problems?\b", line) or _NOT_HEADLINE.match(line):  # footers and frames
        return None
    if any(rx.match(line) for rx in _SYMPTOM):
        return "symptom"
    for rx in _CAUSE:
        if rx.match(line):
            if re.match(r"^\s*\S+\.\w{1,5}[:(\[]", line) and _PASSING.search(line.split(": ", 1)[-1][:80]):
                return None  # "path:line: message" that reports success (go t.Log, verbose loggers)
            return "cause"
    if _ESLINT_ROW.match(line):
        return "cause"
    return None


def _rel(path: str, cwd: Path) -> str:
    p = path.replace("\\", "/")
    p = re.sub(r"^\./", "", p)
    try:
        full = Path(p)
        if full.is_absolute():
            return full.resolve().relative_to(cwd.resolve()).as_posix()
    except (ValueError, OSError):
        pass
    return p


def _locate(lines: list[str], cwd: Path) -> tuple[str | None, int | None]:
    for line in lines:
        m = _FILE_URL.search(line)
        if m:
            path = m.group(1)
            path = path if re.match(r"[A-Za-z]:", path) else "/" + path
            if "node_modules" not in path:
                return _rel(path, cwd), int(m.group(2))
        m = _LOC.search(line)
        if m and not _NOT_OWN.search(m.group(1)) and not m.group(1).startswith(("http", "//")):
            return _rel(m.group(1), cwd), int(m.group(2) or m.group(3) or m.group(4))
    return None, None


_GO_FAIL = re.compile(r"^\s*--- FAIL: (\S+)")
_GO_LOG = re.compile(r"^\s+([\w./-]+\.go):(\d+): (.+)")


def _go_details(lines: list[str], i: int) -> tuple[str, int, str] | None:
    """go test -v prints t.Errorf output under '=== RUN Name', before the '--- FAIL: Name' summary."""
    m = _GO_FAIL.match(lines[i])
    if not m:
        return None
    name = m.group(1)
    start = next((k for k in range(i - 1, -1, -1) if re.match(rf"^=== (?:RUN|CONT)\s+{re.escape(name)}\s*$", lines[k])), None)
    scan = lines[start + 1:i] if start is not None else lines[i + 1:i + 20]
    for line in scan:
        if start is not None and re.match(r"^=== (?:RUN|CONT|PAUSE)\s+", line) and not line.strip().endswith(name):
            if not re.match(rf"^=== \w+\s+{re.escape(name)}/", line):
                break
        g = _GO_LOG.match(line)
        if g:
            return g.group(1), int(g.group(2)), g.group(3)
    return None


def own_frames(lines: list[str], i: int, cwd: Path, depth: int = 60) -> list[tuple[str, int]]:
    """Own-code frames of the stack below a failure, innermost first (as printed)."""
    out: list[tuple[str, int]] = []
    for j, nxt in enumerate(lines[i + 1:i + depth]):
        if j > 0 and _classify(nxt) is not None:
            break
        for rx in _FRAMES:
            m = rx.search(nxt)
            if m and not _NOT_OWN.search(m.group(1)):
                path = m.group(1)
                if "file:" in nxt and not (path.startswith("/") or re.match(r"[A-Za-z]:", path)):
                    path = "/" + path
                loc = (_rel(path, cwd), int(m.group(2)))
                if loc not in out:
                    out.append(loc)
                break
    return out


def _frame(lines: list[str], i: int, cwd: Path) -> tuple[str | None, int | None]:
    """The first frame of the project's own code below the failure (stack trace), or just above it (node ESM)."""
    go = _go_details(lines, i)
    if go:
        return _rel(go[0], cwd), go[1]
    frames = own_frames(lines, i, cwd)
    if frames:
        return frames[0]
    for prev in reversed(lines[max(0, i - 3):i]):
        m = _FRAMES[-1].search(prev)
        if m and not _NOT_OWN.search(m.group(1)):
            path = m.group(1)
            return _rel(path if re.match(r"[A-Za-z]:", path) else "/" + path, cwd), int(m.group(2))
    return None, None


_PY_SECTION = re.compile(r"^_{3,} (.+?) _{3,}$")
_PY_TB_FRAME = re.compile(r"^((?:[A-Za-z]:)?[^\s:]+\.py):(\d+): (?:in \S+|\w*(?:Error|Exception|Warning)\b)")


def _pytest_section(lines: list[str], name: str) -> list[str]:
    for k, line in enumerate(lines):
        m = _PY_SECTION.match(line)
        if m and (m.group(1) == name or m.group(1).endswith("." + name) or m.group(1).endswith(name)):
            out = []
            for nxt in lines[k + 1:k + 400]:
                if _PY_SECTION.match(nxt) or nxt.startswith("=" * 5):
                    break
                out.append(nxt)
            return out
    return []


def _pytest_locate(f: "Finding", lines: list[str], cwd: Path) -> None:
    """FAILED tests/x.py::Cls::test or ERROR collecting x.py: the innermost own frame in its traceback section."""
    m = re.match(r"^FAILED (\S+?\.py)::(\S+?)(?:\[.*?\])?(?: - |$)", f.text)
    c = re.match(r"^_+ ERROR collecting (\S+) _+$", f.text)
    if m:
        test = m.group(2).split("::")[-1]
        section = _pytest_section(lines, m.group(2).replace("::", ".")) or _pytest_section(lines, test)
    elif c:
        section = _pytest_section(lines, f"ERROR collecting {c.group(1)}")
    else:
        return
    frames = [(fm.group(1), int(fm.group(2))) for fm in map(_PY_TB_FRAME.match, section)
              if fm and not _NOT_OWN.search(fm.group(1))]
    errs = [l.strip() for l in section if l.startswith("E ")][:3]
    if frames:
        inner = frames[-1]
        f.file, f.line = _rel(inner[0], cwd), inner[1]
        if frames[0] != inner:
            f.also_at.append(f"{_rel(frames[0][0], cwd)}:{frames[0][1]}")
    if errs:
        f.context = errs + [x for x in f.context if x not in errs]


def findings(text: str, cwd: Path, limit: int = 60) -> list[Finding]:
    lines = text.split("\n")
    seen: dict[str, Finding] = {}
    order: list[Finding] = []
    last_path = None
    for i, line in enumerate(lines):
        pa = _PATH_ALONE.match(line)
        if pa:
            last_path = pa.group(1)
        kind = _classify(line)
        if kind is None:
            continue
        es = _ESLINT_ROW.match(line)
        if es and last_path:
            rel = _rel(last_path, cwd)
            key = f"{rel}:{es.group(1)}: {es.group(3)}"
            if key not in seen:
                f = Finding("cause", key, key, file=rel, line=int(es.group(1)))
                seen[key] = f
                order.append(f)
            continue
        key = _key(line)
        if not key or key.lower().startswith(("expect(", "at ")) or "to rerun pass" in key:
            continue
        ctx: list[str] = []
        if _PY_CRASH.match(line):  # pytest prints the assertion (E lines) above the location line
            for prev in reversed(lines[max(0, i - 8):i]):
                if re.match(r"^(?:E\s|>\s)", prev):
                    ctx.insert(0, shorten_line(prev.rstrip(), 200))
                elif ctx:
                    break
            ctx = ctx[-4:]
        for nxt in lines[i + 1:i + 14]:
            if _classify(nxt) is not None or len(ctx) >= 5 or _CTX_STOP.match(nxt):
                break
            if nxt.strip() and not _NOISE_CTX.match(nxt):
                ctx.append(shorten_line(nxt.rstrip(), 200))
        if key in seen:  # a later report of the same failure often carries the details (node:test, jest)
            old = seen[key]
            old.count += 1
            if any(_ASSERTION.search(c) for c in ctx) and not any(_ASSERTION.search(c) for c in old.context):
                old.context = ctx
                fl = _locate([line] + ctx, cwd)
                if fl[0]:
                    old.file, old.line = fl
            continue
        if kind == "cause" and (re.match(r"\s*(?:Error|AssertionError)\b", line) or _PY_CRASH.match(line)) \
                and (any(_ASSERTION.search(c) for c in ctx) or "AssertionError" in line):
            kind = "symptom"
        f = Finding(kind, shorten_line(line.strip(), 240), key, context=ctx)
        f.file, f.line = _locate([line] + ctx, cwd)
        _pytest_locate(f, lines, cwd)
        go = _go_details(lines, i)
        if go and not any(go[2][:40] in c for c in f.context):
            f.context.insert(0, shorten_line(f"{go[0]}:{go[1]}: {go[2]}", 200))
        if f.file is None:
            f.file, f.line = _frame(lines, i, cwd)
        if f.file and not _TEST_FILE.search(f.file):
            tests = [x for x in own_frames(lines, i, cwd) if _TEST_FILE.search(x[0])]
            if tests:
                f.also_at.append(f"{tests[0][0]}:{tests[0][1]}")
        if f.file is None:  # a Python traceback names the frame above the exception line: take the innermost own frame
            for prev in reversed(lines[max(0, i - 30):i]):
                m = _PY_FRAME.match(prev)
                if m and not re.search(r"site-packages|dist-packages|[\\/]lib[\\/]python|<frozen", m.group(1)):
                    f.file, f.line = _rel(m.group(1), cwd), int(m.group(2))
                    break
        seen[key] = f
        order.append(f)
        if len(order) >= limit:
            break
    # a browser error repeated by the test runner as "Error: ..." is the same failure
    by_core: dict[str, Finding] = {}
    out: list[Finding] = []
    for f in order:
        core = f.key[:120]
        if core in by_core:
            by_core[core].count += f.count
            continue
        by_core[core] = f
        out.append(f)
    return out


# ---------------------------------------------------------------- links to the source

_SOURCE_EXT = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".vue", ".svelte", ".py", ".go", ".rs", ".java",
               ".glsl", ".vert", ".frag", ".wgsl", ".json", ".c", ".cpp", ".h", ".cs", ".rb", ".php"}
_SKIP = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target", ".next", ".ordo",
         "test-results", "playwright-report", "coverage"}


class Tree:
    """Files of the working directory, listed once (bounded) and read lazily."""

    def __init__(self, cwd: Path, max_files: int = 20000):
        self.cwd = cwd
        self.files: list[Path] = []
        self._text: dict[Path, list[str]] = {}
        for dirpath, dirnames, filenames in os.walk(cwd):
            dirnames[:] = [d for d in dirnames if d not in _SKIP]
            for name in filenames:
                self.files.append(Path(dirpath, name))
            if len(self.files) > max_files:
                self.files = []
                break

    def rel(self, p: Path) -> str:
        return p.relative_to(self.cwd).as_posix()

    def lines(self, p: Path) -> list[str]:
        if p not in self._text:
            try:
                self._text[p] = p.read_text(encoding="utf-8", errors="replace").split("\n") if p.stat().st_size < 400_000 else []
            except OSError:
                self._text[p] = []
        return self._text[p]

    def grep(self, rx: re.Pattern, exts: set[str] | None = None, limit: int = 12) -> list[tuple[Path, int, str]]:
        hits = []
        for p in self.files:
            if p.suffix not in (exts or _SOURCE_EXT):
                continue
            for n, line in enumerate(self.lines(p), 1):
                if rx.search(line):
                    hits.append((p, n, line))
                    if len(hits) >= limit:
                        return hits
        return hits


_TS_MEMBER = re.compile(r"'(\w+)' does not exist (?:on|in) type '(\w+)'")


def _link_ts(found: list[Finding], tree: Tree) -> list[Finding]:
    """Group a tsc cascade by (member, type) into one cause at the type's definition."""
    groups: dict[tuple[str, str], list[Finding]] = {}
    for f in found:
        m = _TS_MEMBER.search(f.text)
        if m and "error TS" in f.text:
            groups.setdefault((m.group(1), m.group(2)), []).append(f)
    roots = []
    for (member, typ), members in groups.items():
        if len(members) < 2:
            continue
        files = sorted({f.file for f in members if f.file})
        label = (f"{len(members)} tsc errors: '{member}' does not exist on type '{typ}' "
                 f"(in {', '.join(files[:6])}{'...' if len(files) > 6 else ''})")
        root = Finding("cause", label, label)
        root.count = len(members)
        for p, n, line in tree.grep(re.compile(rf"\b(?:interface|type|class|enum)\s+{re.escape(typ)}\b"),
                                    {".ts", ".tsx", ".mts", ".cts", ".d.ts"}, limit=2):
            root.file, root.line = tree.rel(p), n
            body = []
            for l in tree.lines(p)[n:n + 60]:
                if l.startswith("}"):
                    break
                body.append(l)
            fields = [l.strip() for l in body if re.match(r"\s*\w+\??\s*:", l)][:14]
            root.links.append(f"{typ} is defined at {tree.rel(p)}:{n}; its fields: "
                              + ", ".join(re.match(r"(\w+\??)", x).group(1) for x in fields))
            names = [re.match(r"(\w+)", x).group(1) for x in fields]
            close = [] if member in names else difflib.get_close_matches(member, names, n=1, cutoff=0.4)
            if close:
                root.links.append(f"'{member}' may have been renamed to '{close[0]}': update the users or the type")
            break
        for f in members:
            f.kind = "grouped"
        roots.append(root)
    return roots + [f for f in found if f.kind != "grouped"]


_GLSL = re.compile(r"ERROR: \d+:(\d+): '(\w+)' : (.+)")
_GUARD = re.compile(r"^\s*#\s*(ifdef|ifndef|if)\s+(.+)$")


def _guard(lines: list[str], n: int) -> str | None:
    depth = 0
    for line in reversed(lines[:n - 1]):
        s = line.strip()
        if s.startswith("#endif"):
            depth += 1
        elif _GUARD.match(s):
            if depth == 0:
                m = _GUARD.match(s)
                return f"#{m.group(1)} {m.group(2).strip()}"
            depth -= 1
    return None


def _link_glsl(f: Finding, text: str, tree: Tree) -> None:
    m = _GLSL.search(f.text)
    if not m:
        return
    lineno, ident = m.group(1), m.group(2)
    if ident in ("dot", "pow", "mix", "normalize", "texture", "texture2D", "vec2", "vec3", "vec4", "float", "int"):
        dumped = re.search(rf"^>\s*{lineno}:\s*(.+)$", text, re.MULTILINE)
        if dumped:
            code = dumped.group(1).strip()
            f.links.append(f"failing line: {code}")
            for p, n, _ in tree.grep(re.compile(re.escape(code)), limit=2):
                f.links.append(f"source: {tree.rel(p)}:{n}")
        return
    decl = re.compile(rf"\b(?:varying|uniform|attribute|in|out|const|vec[234]|float|int|mat[234]|sampler2D)\s+[\w\s]*\b{re.escape(ident)}\b\s*[;=\[]")
    for p, n, line in tree.grep(re.compile(rf"\b{re.escape(ident)}\b"), limit=12):
        guard = _guard(tree.lines(p), n)
        what = "declared" if decl.search(line) else "used"
        f.links.append(f"'{ident}' {what} at {tree.rel(p)}:{n}" + (f" (under {guard})" if guard else ""))
        if f.file is None:
            f.file, f.line = tree.rel(p), n
    if len(f.links) > 6:
        f.links = f.links[:6] + [f"... more uses of '{ident}'"]
    _link_switches(f, tree)


_MACRO = re.compile(r"\b(?:USE|HAS|ENABLE|WITH)_([A-Z0-9_]+)\b")


def _link_switches(f: Finding, tree: Tree) -> None:
    """For #ifdef USE_X guards, where the feature X is switched on or off in the code (config objects, flags)."""
    stems = []
    for l in f.links:
        for m in _MACRO.finditer(l):
            if m.group(1) not in stems:
                stems.append(m.group(1))
    for stem in stems[:4]:
        name = stem.lower()
        camel = re.sub(r"_(\w)", lambda m: m.group(1).upper(), name)
        rx = re.compile(rf"\b[\"']?(?:{re.escape(name)}|{re.escape(camel)})[\"']?\s*[:=]\s*(?:true|false|0|1)\b", re.IGNORECASE)
        for p, n, line in tree.grep(rx, {".ts", ".tsx", ".js", ".mjs", ".json", ".py"}, limit=3):
            f.links.append(f"switch for USE_{stem} at {tree.rel(p)}:{n}: {line.strip()[:100]}")


def _link_nan(f: Finding, text: str) -> None:
    if "NaN" not in f.text:
        return
    seen, out = set(), []
    for line in text.split("\n"):
        if "NaN" in line and _classify(line) is None and not re.match(r"\s*[+-]\s", line) and f.key[:40] not in line:
            k = re.sub(r"\+\d+ms|\d{2}:\d{2}:\d{2}", "", line.strip())
            if k not in seen:
                seen.add(k)
                out.append(shorten_line(line.strip(), 200))
        if len(out) >= 3:
            break
    f.links += [f"NaN shows up in: {l}" for l in out]


_MISSING = re.compile(r"404|not found|could not load|failed to load|ENOENT|no such file", re.IGNORECASE)
_ASSET = re.compile(r"(/?[\w.@-]+(?:/[\w.@-]+)*\.(?:png|jpe?g|webp|gif|svg|ktx2|glb|gltf|hdr|exr|json|wasm|css|js|mp3|ogg|wav|bin))")


def _link_asset(f: Finding, tree: Tree) -> None:
    if not _MISSING.search(f.text):
        return
    m = _ASSET.search(_TAIL_LOCATION.sub("", f.text))
    if not m:
        return
    want = m.group(1)
    name = Path(want).name
    if any(tree.rel(p).endswith(want.lstrip("/")) for p in tree.files):
        return
    candidates = {tree.rel(p): p.name for p in tree.files if p.suffix == Path(name).suffix}
    close = difflib.get_close_matches(name, list(set(candidates.values())), n=2, cutoff=0.6)
    hits = [rel for rel, base in candidates.items() if base in close][:3]
    if hits:
        f.links.append(f"{want} does not exist; closest files: {', '.join(hits)}")


def _resolve(f: Finding, tree: Tree) -> None:
    """Runners in a monorepo print paths relative to the package: use the only file in the tree that matches."""
    if not f.file or (tree.cwd / f.file).exists():
        return
    tail = "/" + f.file.lstrip("./")
    hits = [tree.rel(p) for p in tree.files if ("/" + tree.rel(p)).endswith(tail)]
    if len(hits) == 1:
        f.file = hits[0]


def _merge_pytest(found: list[Finding]) -> list[Finding]:
    """'FAILED tests/x.py::test - msg' and 'tests/x.py:17: AssertionError' describe one failure: keep one."""
    crashes = {}
    for f in found:
        m = _PY_CRASH.match(f.text)
        if m and f.file:
            crashes.setdefault(Path(f.file.replace("\\", "/")).name, f)
    used = set()
    for f in found:
        m = re.match(r"^FAILED (\S+?\.py)::", f.text)
        if m and f.file is None:
            crash = crashes.get(Path(m.group(1).replace("\\", "/")).name)
            if crash is not None and id(crash) not in used:
                f.file, f.line = crash.file, crash.line
                f.context = crash.context + f.context
                used.add(id(crash))
    return [f for f in found if id(f) not in used]


def _merge_glsl(found: list[Finding]) -> list[Finding]:
    by_line: dict[str, Finding] = {}
    out = []
    has_compiler = any(_GLSL.search(f.text) for f in found)
    for f in found:
        if has_compiler and "WebGLProgram: Shader Error" in f.text:
            continue
        m = _GLSL.search(f.text)
        if m:
            first = by_line.get(m.group(1))
            if first is not None:
                first.context = [f"also: {f.key}"] + first.context
                for l in f.links:
                    if l not in first.links:
                        first.links.append(l)
                continue
            by_line[m.group(1)] = f
        out.append(f)
    return out


def _cause_rank(f: Finding) -> int:
    if _GLSL.search(f.text) or "error TS" in f.text or f.key.startswith(("ts:", )) or "tsc errors:" in f.key:
        return 0
    return 1 if f.text.startswith("[browser:error]") or f.links else 2


_STOP = set("""about after also and any are because before being both but can cause check com como could does
doing dont each error errors even fail failed failing fails failure failures falha falhando falham find fix
for from have here how into isso just make matter matters more must need never not now only other para pela
pelo por porque qual quais que quero should show some sobre stop sure tell test tests testes than that the
their them then there these they this those todos tracked tudo very want what when where which while why will
with without would wrong your entender only elsewhere ignore ignora ignorar ignore ele ela está esta até onde
quando então mais menos já ainda mas uma uns umas dos das nos nas the one ones all including investigate known
separate issue""".split())


def _terms(intent: str) -> tuple[set[str], set[str]]:
    """Words of the question that can tell failures apart, and the ones it asks to ignore."""
    words = [w.lower().strip(".") for w in re.findall(r"[A-Za-z_][\w.]{2,}", intent)]
    neg: set[str] = set()
    m = re.search(r"\b(?:ignore|ignora|ignorar|not|sem|except|exceto|tracked elsewhere)\b(.*)", intent, re.IGNORECASE)
    if m:
        neg = {w.lower() for w in re.findall(r"[A-Za-z_][\w.]{3,}", m.group(1))} - _STOP
    pos = {w for w in words if len(w) >= 3 and w not in _STOP} - neg
    return pos, neg


def _focus(found: list[Finding], intent: str) -> None:
    """When the question names a part (a module, a test), failures elsewhere move to "other"."""
    pos, neg = _terms(intent or "")
    if not pos:
        return
    def hits(f: Finding, terms: set[str]) -> int:
        hay = " ".join([f.key, f.file or "", *f.context, *f.links]).lower()
        return sum(1 for t in terms if re.search(rf"(?<![a-z0-9]){re.escape(t)}", hay))
    scores = [(hits(f, pos), hits(f, neg)) for f in found]
    if not any(p for p, _ in scores) or all(p for p, _ in scores):
        return
    for f, (p, n) in zip(found, scores):
        if p == 0 or n > p:
            f.kind = "other"


def analyze(text: str, command: str | None, cwd: str | Path, intent: str = "") -> dict:
    """status, runners, causes (with links) and symptoms; `findings` is empty when nothing is recognized."""
    root = Path(cwd)
    found = findings(text, root)
    tree = Tree(root) if found else None
    if tree is not None:
        found = _link_ts(found, tree)
        for f in found:
            _resolve(f, tree)
            _link_glsl(f, text, tree)
            _link_nan(f, text)
            _link_asset(f, tree)
    found = _merge_glsl(_merge_pytest(found))
    _focus(found, intent)
    found.sort(key=lambda f: ({"cause": 0, "symptom": 1}.get(f.kind, 2), _cause_rank(f) if f.kind == "cause" else 0))
    return {"runners": detect_runners(command, text), "findings": found}


def confident(analysis: dict, intent: str = "") -> bool:
    """Answer without the model only when every failure is pinned down and the question is about them.

    Measured on 36 held-out validation logs and the 45 failing benchmark cases: with this rule the index answers
    83% and 76% of them in milliseconds; file recall 77.8% vs 83.3% (model) on the first, 95.6% vs 86.7% on the second.
    """
    fs = [f for f in analysis["findings"] if f.kind in ("cause", "symptom")]
    causes = [f for f in fs if f.kind == "cause"]
    pinned = lambda f: bool(f.file or f.links)  # noqa: E731
    # with causes, they must all be pinned (assertion messages are their symptoms); without, every failing test must be
    if not fs or not (all(map(pinned, causes)) if causes else all(map(pinned, fs))):
        return False
    # names, not words: identifiers (test_fmt, TestRefill, cpf.test.js, case_78) and quoted text
    quoted = {q.lower() for q in re.findall(r"[`'\"]([^`'\"]{3,60})[`'\"]", intent or "")}
    idents = {w.lower().strip(".") for w in re.findall(r"[A-Za-z_][\w./-]{2,}", intent or "")
              if re.search(r"[_/]|\.\w|\d|[a-z][A-Z]", w) and w.lower() not in ("file:line",)}
    specific = quoted | idents
    if specific:  # "the test_fmt failure", "the escape table": one of the failures must be about it
        hay = " ".join(f"{f.key} {f.file or ''} {' '.join(f.context)} {' '.join(f.links)}" for f in fs).lower()
        if not any(t in hay for t in specific):
            return False
    return True


def to_result(analysis: dict, exit_code: int | None, *, max_causes: int = 8, max_symptoms: int = 10) -> dict:
    """The YAML/JSON body of route "index"."""
    causes = [f for f in analysis["findings"] if f.kind == "cause"]
    symptoms = [f for f in analysis["findings"] if f.kind == "symptom"]
    others = [f for f in analysis["findings"] if f.kind == "other"]
    runners = analysis["runners"]
    if exit_code == 0:  # the exit code is a fact: a passing command with findings has warnings, not failures
        status = "WARN" if (causes or symptoms) else "PASS"
    else:
        status = "FAIL" if (exit_code is not None or causes or symptoms) else "PASS"
    parts = []
    if causes:
        parts.append(f"{len(causes)} distinct error{'s' if len(causes) != 1 else ''}")
    if symptoms:
        parts.append(f"{len(symptoms)} failing test{'s' if len(symptoms) != 1 else ''}")
    summary = f"{', '.join(runners) or 'command'}: {' and '.join(parts) or 'no failure found'}"
    if causes:
        summary += f"; first: {causes[0].key[:160]}"
    elif symptoms:
        summary += f"; first: {symptoms[0].key[:160]}"
    out: dict = {"status": status, "summary": summary}
    if causes:
        out["causes"] = [_item(f) for f in causes[:max_causes]]
        if len(causes) > max_causes:
            out["causes"].append({"error": f"... {len(causes) - max_causes} more (ordo show <raw_id> --grep ...)"})
    if symptoms:
        out["failing_tests"] = [_item(f, short=True) for f in symptoms[:max_symptoms]]
        if len(symptoms) > max_symptoms:
            out["failing_tests"].append({"test": f"... {len(symptoms) - max_symptoms} more"})
    if others:
        out["other_failures"] = [f.key[:160] for f in others[:5]] + ([f"... {len(others) - 5} more"] if len(others) > 5 else [])
    return out


def _item(f: Finding, short: bool = False) -> dict:
    item: dict = {"test" if short else "error": f.key}
    if f.file:
        item["at"] = f"{f.file}:{f.line}" if f.line else f.file
    detail = [c.strip() for c in f.context if c.strip()][:2 if short else 3]
    if detail and not short:
        item["detail"] = detail
    elif detail and short:
        item["detail"] = detail[0] if len(detail) == 1 else detail
    if f.also_at:
        item["called_from"] = f.also_at[0]
    if f.links:
        item["links"] = f.links
    if f.count > 1:
        item["seen"] = f.count
    return item


def error_keys(analysis: dict) -> list[str]:
    return [f.key[:200] for f in analysis["findings"]]


def cause_files(analysis: dict) -> dict[str, list[str]]:
    """For each failure key, the files it points to (its location and linked definitions)."""
    out = {}
    for f in analysis["findings"]:
        if f.kind != "cause":  # a failing test points at the test file, which a correct fix never touches
            continue
        files = [f.file] if f.file else []
        for l in f.links:  # only where the failure lives, not suggestions (closest files) or log lines
            if re.search(r"\b(?:defined|declared|used) at |^source: ", l):
                files += [m.group(1) for m in re.finditer(r"([\w./@-]+\.\w{1,5}):\d+", l)]
        out[f.key[:200]] = sorted({x for x in files if x})
    return out
