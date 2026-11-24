"""Ordo CLI: filters tool output, slices source files and keeps session memory for coding agents.

Principle: the CLI never leaves the agent without a useful answer. If the model fails (server down,
unrecoverable YAML), it prints the windowed raw output with a header explaining why (turn this off with
--no-fallback). Verifier warnings are YAML comments (`# ordo: ...`), so the output is still valid YAML.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path

from . import __version__
from .client import DEFAULT_BASE_URL, DEFAULT_HOST, DEFAULT_MODEL, call_ordo
from .contract import GcMode
from .schema import parse_output
from .session import Session
from .textproc import clean, est_tokens, window_file, window_log
from .verify import verify

DEFAULT_BUDGET = int(os.environ.get("ORDO_MAX_INPUT_TOKENS", "6000"))
BYPASS_LINES, BYPASS_TOKENS = 40, 400  # below this, the model spends more tokens than it saves
RAW_KEEP = 300  # raw logs kept in ~/.ordo/runs (the oldest are removed)


@dataclass
class Outcome:
    text: str  # what goes to stdout
    status: str | None = None
    summary: str | None = None
    raw_tokens: int = 0
    out_tokens: int = 0
    seconds: float = 0.0
    route: str = "model"  # model | bypass | fallback
    windowed: bool = False
    warnings: list[str] = field(default_factory=list)
    data: dict | None = None  # parsed result (for --json)

    def event(self) -> dict:
        return {"status": self.status, "summary": self.summary, "raw_tokens": self.raw_tokens,
                "out_tokens": self.out_tokens, "seconds": round(self.seconds, 2), "route": self.route,
                "warnings": len(self.warnings)}


_SINK: ContextVar = ContextVar("ordo_sink", default=None)  # output destination when called by the MCP server


def _print(text: str) -> None:
    sink = _SINK.get()
    if sink is not None:
        sink.write(text.rstrip("\n") + "\n")
        return
    sys.stdout.write(text.rstrip("\n") + "\n")
    sys.stdout.flush()


def _fallback(args, reason: str, raw: str, raw_tokens: int) -> Outcome:
    if args.no_fallback:
        print(f"ordo: {reason}", file=sys.stderr)
        raise SystemExit(3)
    body, cut = window_log(raw, args.budget)
    head = f"# ordo: {reason}; showing raw output" + (" (windowed)" if cut else "")
    return Outcome(text=f"{head}\n{body}", raw_tokens=raw_tokens, out_tokens=est_tokens(body), route="fallback")


def run_model(args, mode: GcMode, intent: str, raw: str, *, source_lines: list[str] | None = None,
              raw_tokens: int | None = None) -> Outcome:
    raw_tokens = raw_tokens if raw_tokens is not None else est_tokens(raw)
    try:
        res = call_ordo(mode, intent, raw, model=args.model, host=args.host, base_url=args.base_url,
                        timeout=args.timeout)
    except Exception as e:  # noqa: BLE001 - any server failure becomes a fallback
        return _fallback(args, f"model call failed ({type(e).__name__}: {e})", raw, raw_tokens)
    parsed = parse_output(res.text, mode)
    if not parsed.yaml_ok:
        return _fallback(args, "model returned invalid YAML", raw, raw_tokens)
    text = res.text.strip() if parsed.raw_yaml_ok else _normalized(parsed.data)
    out = Outcome(text=text, raw_tokens=raw_tokens, out_tokens=res.output_tokens, seconds=res.seconds,
                  data=parsed.data)
    out.status = str(parsed.data.get("status")) if parsed.data.get("status") else None
    out.summary = parsed.data.get("summary") or parsed.data.get("active_goal")
    if getattr(res, "truncated", False):
        out.warnings.append("model output hit the token limit and was cut")
    if parsed.repaired:
        out.warnings.append("output YAML was repaired")
    out.warnings += [e for e in parsed.errors]
    checked = parsed.obj or parsed.lenient
    if checked is not None:
        rep = verify(checked, raw, mode, source_lines=source_lines)
        for c in rep.failures:
            what = {"file": "file not in input", "line": "line not in input", "snippet": "snippet not verbatim",
                    "symbol": "symbol not in cited lines", "rule": "rule broken"}[c.kind]
            out.warnings.append(f"unverified: {what}: {c.target}" + (f" ({c.detail})" if c.detail else ""))
    if out.warnings:
        out.text += "\n" + "\n".join(f"# ordo: {w}" for w in out.warnings)
    return out


def _normalized(data: dict) -> str:
    from .schema import dump_target
    return dump_target(data)


def _header(label: str, out: Outcome, extra: str = "") -> str:
    parts = [f"# ordo {label}"]
    if extra:
        parts.append(extra)
    if out.route == "model":
        parts.append(f"~{out.raw_tokens} -> {out.out_tokens} tokens, {out.seconds:.1f}s")
    elif out.route == "bypass":
        parts.append("short output, passed through")
    return " | ".join(parts)


def save_raw(text: str) -> str:
    """Save the full output (ANSI already stripped) so the agent can recover parts with `ordo show`."""
    from .session import ordo_home

    folder = ordo_home() / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
    (folder / f"{rid}.log").write_text(text, encoding="utf-8")
    for f in sorted(folder.glob("*.log"))[:-RAW_KEEP]:
        f.unlink(missing_ok=True)
    return rid


def emit(args, label: str, out: Outcome, extra: str = "", *, code: int | None = None, raw_id: str | None = None,
         **fields) -> None:
    """Single output path for every command: YAML with a header (default) or a JSON envelope (--json)."""
    if getattr(args, "json", False):
        env = {"command": label, "route": out.route, "status": out.status, "exit": code, "raw_id": raw_id,
               "tokens": {"input": out.raw_tokens, "output": out.out_tokens}, "seconds": round(out.seconds, 2),
               "warnings": out.warnings, **fields,
               "result": out.data if out.route == "model" else None,
               "text": None if out.route == "model" else out.text}
        _print(json.dumps(env, ensure_ascii=False, indent=1))
        return
    head = _header(label, out, extra)
    if raw_id:
        head += f" | raw={raw_id}"
    _print(head + "\n" + out.text)


def _filter_or_bypass(args, mode: GcMode, intent: str, text: str) -> Outcome:
    tokens = est_tokens(text)
    if not args.force and len(text.splitlines()) <= BYPASS_LINES and tokens <= BYPASS_TOKENS:
        return Outcome(text=text, raw_tokens=tokens, out_tokens=tokens, route="bypass")
    body, cut = window_log(text, args.budget)
    out = run_model(args, mode, intent, body, raw_tokens=tokens)
    if cut and out.route == "model":
        out.warnings.append(f"input exceeded ~{args.budget} tokens and was windowed before the model")
        out.text += f"\n# ordo: {out.warnings[-1]}"
    return out


# ---------------------------------------------------------------- comandos


_CMD_META = set('&|<>^()%!"') | {" ", "\t"}


def _cmd_quote(arg: str) -> str:
    """Quoting for cmd.exe: inside quotes, & | < > ^ ( ) are literal (only % still expands)."""
    if arg and not (set(arg) & _CMD_META):
        return arg
    return '"' + arg.replace('"', '""') + '"'


def _resolve_command(cmd: list[str]) -> tuple[list[str] | str, bool]:
    """Resolve the executable. On Windows, only .cmd/.bat (npm, npx, yarn, mvn...) go through cmd.exe.

    A missing command does not go to the shell: it raises FileNotFoundError and exits 127, as on POSIX.
    """
    exe = shutil.which(cmd[0])
    if exe is None:
        return cmd, False
    if os.name == "nt" and exe.lower().endswith((".cmd", ".bat")):
        return " ".join([_cmd_quote(exe)] + [_cmd_quote(a) for a in cmd[1:]]), True
    return [exe] + cmd[1:], False


def _decode(data: bytes) -> str:
    """UTF-8; otherwise the system code page (cp1252/cp850 on Windows), never failing."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        pass
    import locale

    for enc in (locale.getpreferredencoding(False), "cp850" if os.name == "nt" else "latin-1"):
        try:
            return data.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("utf-8", errors="replace")


def _kill_tree(proc: subprocess.Popen) -> None:
    if os.name == "nt":  # killing only cmd.exe would leave node/java orphaned
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(proc.pid, 9)
        except (ProcessLookupError, PermissionError):
            proc.kill()


def _execute(target, shell: bool, timeout: float | None, cwd: str | None = None) -> tuple[int, bytes, bool]:
    kw = {"start_new_session": True} if os.name != "nt" else {}
    proc = subprocess.Popen(target, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, cwd=cwd or None, **kw)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return proc.returncode, out, False
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        out, _ = proc.communicate()
        return 124, out or b"", True


def _read_input(path: str | None, label: str) -> bytes | None:
    if not path or path == "-":
        if sys.stdin is None or sys.stdin.isatty():  # no pipe: it would wait for the keyboard forever
            print(f"ordo {label}: nothing on stdin; pipe the output in (cmd 2>&1 | ordo {label} -i ...) "
                  f"or pass a file", file=sys.stderr)
            return None
        return sys.stdin.buffer.read()
    try:
        return Path(path).read_bytes()
    except OSError as e:
        print(f"ordo {label}: cannot read {path}: {e.strerror or e}", file=sys.stderr)
        return None


def cmd_run(args) -> int:
    cmd = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not cmd:
        print("ordo run: missing command (usage: ordo run -i INTENT -- CMD ...)", file=sys.stderr)
        return 2
    target, shell = _resolve_command(cmd)
    t0 = time.time()
    try:
        code, data, timed_out = _execute(target, shell, args.cmd_timeout, getattr(args, "cwd", None))
    except FileNotFoundError:
        print(f"ordo run: command not found: {cmd[0]}", file=sys.stderr)
        return 127
    output = _decode(data)
    if timed_out:
        output += f"\n[ordo: command killed after {args.cmd_timeout}s timeout]"
    elapsed = time.time() - t0
    text = clean(output)
    intent = args.intent or f"Report the outcome of `{' '.join(cmd)}`"
    out = _filter_or_bypass(args, GcMode.TOOL_OUTPUT, intent, text)
    if out.status is None:  # short output or fallback: status comes from the exit code
        out.status = "PASS" if code == 0 else "FAIL"
    rid = save_raw(text)
    emit(args, "run", out, f"exit={code}", code=code, raw_id=rid)
    args.session_obj.append("run", command=" ".join(cmd), exit=code, intent=intent, cmd_seconds=round(elapsed, 2),
                            raw_id=rid, **out.event())
    return code


def cmd_filter(args) -> int:
    data = _read_input(args.file, "filter")
    if data is None:
        return 2
    text = _decode(data)
    mode = GcMode(args.mode)
    intent = args.intent or "Summarize the outcome and any failure cause"
    if mode is GcMode.CODE_SLICE:  # input treated as a source file: number it and verify against it
        name = args.file if args.file not in (None, "-") else "stdin"
        args.intent = intent
        out = _slice_text(args, text.replace("\r\n", "\n"), name.replace("\\", "/"))
    else:
        out = _filter_or_bypass(args, mode, intent, clean(text))
    rid = save_raw(text) if mode is not GcMode.CODE_SLICE else None
    emit(args, "filter", out, raw_id=rid)
    args.session_obj.append("filter", intent=intent, mode=mode.value, raw_id=rid, **out.event())
    return 0


def source_lines(src: str) -> list[str]:
    """Count lines like an editor: only \\n separates; a trailing \\n does not add a line."""
    lines = src.split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines


def _slice_text(args, src: str, rel: str) -> Outcome:
    lines = source_lines(src)
    body, cut = window_file("\n".join(lines), args.intent, args.budget)
    raw = f"// file: {rel} ({len(lines)} lines)\n{body}"
    out = run_model(args, GcMode.CODE_SLICE, args.intent, raw, source_lines=lines, raw_tokens=est_tokens(src))
    out.windowed = cut
    return out


def cmd_slice(args) -> int:
    path = Path(args.file)
    if not path.is_file():
        print(f"ordo slice: not a file: {path} (looked in {Path.cwd()})", file=sys.stderr)
        return 2
    src = _decode(path.read_bytes()).replace("\r\n", "\n")
    rel = _display_path(path, args.session_obj.root, args.file)
    out = _slice_text(args, src, rel)
    extra = f"{rel}, {len(source_lines(src))} lines" + (", pre-filtered by intent" if out.windowed else "")
    emit(args, "slice", out, extra, file=rel)
    args.session_obj.append("slice", file=rel, intent=args.intent, windowed=out.windowed, **out.event())
    return 0


def _display_path(path: Path, root: Path, given: str) -> str:
    """Short path as it appears in logs: relative to the session repo, to the file's own repo, or as typed."""
    from .session import repo_root

    for base in (root, repo_root(path.resolve().parent)):
        try:
            return path.resolve().relative_to(base).as_posix()
        except ValueError:
            continue
    return given.replace("\\", "/")


_SKIP_DIRS = {".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target", ".next", ".ordo"}


def _py_search(rx: re.Pattern, paths: list[str], context: int, limit: int = 400) -> str:
    """Search without rg, in the same format (`path:N:match`, `path-N-context`, `--` between blocks)."""
    out: list[str] = []
    hits = 0
    for base in paths:
        bp = Path(base)
        files = [bp] if bp.is_file() else sorted(p for p in bp.rglob("*") if p.is_file())
        for f in files:
            if any(part in _SKIP_DIRS for part in f.parts):
                continue
            try:
                data = f.read_bytes()
            except OSError:
                continue
            if b"\0" in data[:4096]:
                continue
            lines = _decode(data).splitlines()
            idx = [i for i, l in enumerate(lines) if rx.search(l)]
            if not idx:
                continue
            spans: list[list[int]] = []
            for i in idx:
                a, b = max(i - context, 0), min(i + context, len(lines) - 1)
                if spans and a <= spans[-1][1] + 1:
                    spans[-1][1] = b
                else:
                    spans.append([a, b])
            name, hitset = f.as_posix(), set(idx)
            for a, b in spans:
                if out:
                    out.append("--")
                for j in range(a, b + 1):
                    sep = ":" if j in hitset else "-"
                    out.append(f"{name}{sep}{j + 1}{sep}{lines[j].rstrip()}")
            hits += len(idx)
            if hits >= limit:
                return "\n".join(out) + f"\n[ordo: stopped after {limit} matches]"
    return "\n".join(out)


_RG_LINE = re.compile(r"^((?:[A-Za-z]:)?[^:\n]*?)([:-]\d+[:-])")


def _normalize_paths(text: str) -> str:
    """Replace \\ with / only in the path at the start of each line (the matched code is left intact)."""
    out = []
    for line in text.split("\n"):
        m = _RG_LINE.match(line)
        out.append(m.group(1).replace("\\", "/") + line[m.end(1):] if m else line)
    return "\n".join(out)


def cmd_search(args) -> int:
    paths = args.paths or ["."]
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "-n", "--no-heading", "--color", "never", "-S", "-C", str(args.context), "--", args.pattern, *paths]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 2 and not proc.stdout:
            print(f"ordo search: {_decode(proc.stderr).strip()}", file=sys.stderr)
            return 2
        text = _decode(proc.stdout)
    else:
        smart_case = 0 if any(c.isupper() for c in args.pattern) else re.IGNORECASE
        try:
            rx = re.compile(args.pattern, smart_case)
        except re.error as e:
            print(f"ordo search: invalid regex {args.pattern!r}: {e}", file=sys.stderr)
            return 2
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            print(f"ordo search: path not found: {', '.join(missing)} (looked in {Path.cwd()})", file=sys.stderr)
            return 2
        text = _py_search(rx, paths, args.context)
    text = _normalize_paths(clean(text))
    if not text.strip():
        _print(f"# ordo search: no matches for {args.pattern!r}")
        return 1
    intent = args.intent or f"Find where {args.pattern} is defined and used"
    out = _filter_or_bypass(args, GcMode.TOOL_OUTPUT, intent, text)
    rid = save_raw(text)
    emit(args, "search", out, f"pattern={args.pattern!r}", raw_id=rid)
    args.session_obj.append("search", pattern=args.pattern, intent=intent, raw_id=rid, **out.event())
    return 0


def cmd_dead_end(args) -> int:
    args.session_obj.append("dead_end", attempt=args.attempt, reason=args.reason)
    _print(f"# ordo: dead-end recorded in {args.session_obj.path.name}\n"
           f"attempt: {json.dumps(args.attempt, ensure_ascii=False)}\nreason_failed: {json.dumps(args.reason, ensure_ascii=False)}")
    return 0


def build_trajectory(session: Session, extra: str = "") -> str:
    parts: list[str] = []
    prev = session.last_compact()
    if prev and prev.get("ledger"):
        parts.append("[previous ledger]\n" + prev["ledger"])
    turn = 0
    for ev in session.since_last_compact():
        k = ev["kind"]
        if k == "run":
            turn += 1
            parts.append(f"[turn {turn}] $ {ev['command']}  (exit {ev['exit']})\n"
                         f"intent: {ev.get('intent', '')}\nresult: {ev.get('status') or '-'}: {ev.get('summary') or '-'}")
        elif k in ("slice", "search", "filter"):
            turn += 1
            what = ev.get("file") or ev.get("pattern") or ""
            parts.append(f"[turn {turn}] ordo {k} {what}\nintent: {ev.get('intent', '')}\n"
                         f"result: {ev.get('status') or '-'}: {ev.get('summary') or '-'}")
        elif k == "dead_end":
            parts.append(f"[dead-end] attempt: {ev['attempt']}\nreason_failed: {ev['reason']}")
        elif k == "note":
            parts.append(f"[note] {ev['text']}")
    if extra.strip():
        parts.append("[agent transcript]\n" + extra.strip())
    return "\n\n".join(parts)


_TEST_CMD = re.compile(r"\b(pytest|jest|vitest|mocha|rspec|phpunit|go test|cargo test|dotnet test|gradle\w*\s+test|"
                       r"mvn\w*\s+test|npm (run )?test|yarn test|pnpm test|mix test|tox|nox|unittest)\b")


def _git_changed(root: Path) -> list[str]:
    try:
        out = subprocess.run(["git", "-c", "core.quotepath=false", "status", "--porcelain", "--untracked-files=all"],
                             cwd=root, capture_output=True, encoding="utf-8", errors="replace", timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return []
    files = []
    for line in out.stdout.splitlines():
        path = line[3:].split(" -> ")[-1].strip().strip('"')
        if path and not path.endswith("/"):
            files.append(path)
    return files


def deterministic_ledger(session: Session, goal: str) -> "MemoryCompact":
    """Ledger built only from what the session recorded: nothing is inferred by a model.

    The MEMORY_COMPACT output of the published ordo-3b does not follow the schema and invents facts,
    so this is the default path of `ordo compact`.
    """
    from .schema import DeadEnd, FileState, MemoryCompact

    facts: list[str] = []
    dead: list[DeadEnd] = []
    prev = session.last_compact()
    if prev and prev.get("ledger"):
        p = parse_output(prev["ledger"], GcMode.MEMORY_COMPACT)
        if p.obj is not None:
            facts += p.obj.confirmed_facts
            dead += p.obj.dead_ends_do_not_repeat
    last_by_cmd: dict[str, dict] = {}
    last_test: dict | None = None
    for ev in session.since_last_compact():
        if ev["kind"] == "note":
            facts.append(ev["text"])
        elif ev["kind"] == "dead_end":
            dead.append(DeadEnd(attempt=ev["attempt"], reason_failed=ev["reason"]))
        elif ev["kind"] == "run":
            last_by_cmd[ev["command"]] = ev
            if _TEST_CMD.search(ev["command"]):
                last_test = ev
    for cmd, ev in last_by_cmd.items():
        result = ev.get("status") or ("PASS" if ev["exit"] == 0 else "FAIL")
        line = f"Last `{cmd}`: exit {ev['exit']} ({result})"
        if ev.get("summary") and ev.get("route") == "model":
            line += f" — {ev['summary']}"
        facts.append(line)
    seen: set[str] = set()
    facts = [f for f in facts if not (f in seen or seen.add(f))]
    seen_d: set[str] = set()
    dead = [d for d in dead if not (d.attempt in seen_d or seen_d.add(d.attempt))]
    test_status = "UNKNOWN" if last_test is None else ("PASS" if last_test["exit"] == 0 else "FAIL")
    files = [FileState(file=f, test_status=test_status) for f in _git_changed(session.root)]
    return MemoryCompact(status="COMPACTED", active_goal=goal, confirmed_facts=facts,
                         dead_ends_do_not_repeat=dead, modified_files_state=files)


def cmd_compact(args) -> int:
    from .schema import dump_target

    extra = ""
    if args.from_file:
        data = _read_input(args.from_file, "compact")
        if data is None:
            return 2
        extra = clean(_decode(data))
    if not (args.neural or extra):
        ledger = dump_target(deterministic_ledger(args.session_obj, args.intent))
        _print(f"# ordo compact | {args.session_obj.path.name} | deterministic (session records only)\n{ledger}")
        args.session_obj.append("compact", intent=args.intent, ledger=ledger, route="deterministic")
        return 0
    traj = build_trajectory(args.session_obj, extra)
    if not traj.strip():
        _print("# ordo compact: session is empty (use ordo run / dead-end first, or --from FILE)")
        return 1
    body, _ = window_log(traj, args.budget)
    out = run_model(args, GcMode.MEMORY_COMPACT, args.intent, body, raw_tokens=est_tokens(traj))
    if out.route == "model" and not parse_output(_strip_comments(out.text), GcMode.MEMORY_COMPACT).schema_ok:
        if args.no_fallback:
            _print(_header("compact", out, args.session_obj.path.name) + "\n" + out.text)
            return 3
        ledger = dump_target(deterministic_ledger(args.session_obj, args.intent))
        _print(f"# ordo compact | {args.session_obj.path.name} | model output failed the schema; "
               f"deterministic ledger instead\n{ledger}")
        args.session_obj.append("compact", intent=args.intent, ledger=ledger, **{**out.event(), "route": "fallback"})
        return 0
    _print(_header("compact", out, args.session_obj.path.name) + "\n" + out.text)
    if out.route == "model":
        args.session_obj.append("compact", intent=args.intent, ledger=_strip_comments(out.text), **out.event())
    return 0


def _strip_comments(text: str) -> str:
    return "\n".join(l for l in text.splitlines() if not l.startswith("# ordo:"))


def cmd_note(args) -> int:
    args.session_obj.append("note", text=args.text)
    _print(f"# ordo: note recorded in {args.session_obj.path.name}")
    return 0


def cmd_stats(args) -> int:
    rows: dict[str, dict] = {}
    for ev in args.session_obj.events():
        if "raw_tokens" not in ev:
            continue
        r = rows.setdefault(ev["kind"], {"calls": 0, "raw": 0, "out": 0, "secs": 0.0, "model": 0, "bypass": 0,
                                         "fallback": 0, "warned": 0})
        r["calls"] += 1
        r["raw"] += ev.get("raw_tokens", 0)
        r["out"] += ev.get("out_tokens", 0)
        r["secs"] += ev.get("seconds", 0.0)
        r[ev.get("route", "model")] += 1
        r["warned"] += 1 if ev.get("warnings") else 0
    total = {k: sum(r[k] for r in rows.values()) for k in ("calls", "raw", "out", "secs", "model", "bypass", "fallback", "warned")}
    if args.json:
        _print(json.dumps({"session": str(args.session_obj.path), "by_command": rows, "total": total}, indent=2))
        return 0
    from rich.console import Console
    from rich.table import Table

    t = Table(title=f"ordo stats: {args.session_obj.path.name}", caption="in = estimated tokens of raw output; "
              "routes = model/bypass/fallback")
    for col in ("command", "calls", "in", "out", "saved", "routes", "warned", "time"):
        t.add_column(col, justify="right" if col != "command" else "left", no_wrap=True)
    for name, r in sorted(rows.items()) + [("TOTAL", total)]:
        saved = r["raw"] - r["out"]
        pct = f" ({saved / r['raw']:.0%})" if r["raw"] else ""
        t.add_row(name, str(r["calls"]), f"{r['raw']:,}", f"{r['out']:,}", f"{saved:,}{pct}",
                  f"{r['model']}/{r['bypass']}/{r['fallback']}", str(r["warned"]), f"{r['secs']:.0f}s")
    Console(width=max(Console().width, 100)).print(t)
    if not rows:
        _print("no calls recorded yet in this session")
    return 0


# ---------------------------------------------------------------- argparse


def cmd_show(args) -> int:
    """Recover the full output of a run/filter/search (the filter compresses, but the original is kept)."""
    from .session import ordo_home

    folder = ordo_home() / "runs"
    logs = sorted(folder.glob("*.log")) if folder.exists() else []
    if not logs:
        print("ordo show: no saved output yet (run, filter and search save theirs)", file=sys.stderr)
        return 1
    target = logs[-1] if args.id in (None, "last") else folder / f"{args.id}.log"
    if not target.exists():
        print(f"ordo show: unknown id {args.id!r}; recent: {', '.join(p.stem for p in logs[-5:])}", file=sys.stderr)
        return 2
    lines = target.read_text(encoding="utf-8").split("\n")
    keep: list[int] = list(range(len(lines)))
    if args.lines:
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", args.lines.strip())
        if not m:
            print("ordo show: --lines takes N or A-B (1-based)", file=sys.stderr)
            return 2
        a = int(m.group(1))
        b = int(m.group(2) or a)
        keep = list(range(max(a - 1, 0), min(b, len(lines))))
    if args.grep:
        try:
            rx = re.compile(args.grep, 0 if any(c.isupper() for c in args.grep) else re.IGNORECASE)
        except re.error as e:
            print(f"ordo show: invalid regex: {e}", file=sys.stderr)
            return 2
        hits = [i for i in keep if rx.search(lines[i])]
        around: set[int] = set()
        for i in hits:
            around.update(range(max(i - args.context, 0), min(i + args.context + 1, len(lines))))
        keep = sorted(around)
    if args.tail:
        keep = keep[-args.tail:]
    width = len(str(len(lines)))
    out, prev = [f"# ordo show {target.stem} | {len(lines)} lines"], None
    for i in keep:
        if prev is not None and i != prev + 1:
            out.append("--")
        out.append(f"{i + 1:>{width}} | {lines[i]}")
        prev = i
    _print("\n".join(out))
    return 0


GUIDE = """## Ordo: tool-output filter for this repository

`ordo` runs commands and condenses their output into causal YAML (status, file, line, symbol, error) using a
local model, so long test, build and lint logs cost a few hundred tokens instead of thousands. The full
output is always kept and can be read back verbatim.

### Commands

| Task | Command |
|---|---|
| Run tests, builds, linters | `ordo run -i "<question>" -- <command> [args]` |
| Condense output you already have | `ordo filter -i "<question>" <file>` (or pipe it on stdin) |
| Locate code in a large file | `ordo slice -i "<symbol or behaviour>" <file>` |
| Search the repository | `ordo search -i "<question>" <regex> [path]` |
| Read original lines | `ordo show <raw_id> --grep "<regex>"` or `--lines A-B` or `--tail N` |
| Record a failed approach | `ordo dead-end "<attempt>" "<reason it failed>"` |
| Record a confirmed fact | `ordo note "<fact, with path:line>"` |
| Print the session state | `ordo compact -i "<current goal>"` |

`-i` is the question the output must answer. Be specific: name the failing test, module or symptom
("why does test_refund_partial fail?"), not "check the output".

### Output contract

- First line is a header, e.g. `# ordo run | exit=1 | ~933 -> 249 tokens, 11.1s | raw=<raw_id>`
  (`exit` only on `run`; `raw` on run, filter and search).
- The body is YAML: `status` (PASS, FAIL, ERROR, WARN; FOUND or NOT_FOUND for slice and search), `summary`,
  `causal_trace` (file, lines, symbol, issue), `exact_snippets`, `dropped_noise`. `slice` returns
  `relevant_symbols` with 1-based line ranges.
- `short output, passed through` in the header: output under ~40 lines is printed unchanged.
- `# ordo: unverified: ...` comments mark citations the grounding check could not find in the input.
  Do not act on them before confirming with `ordo show <raw_id>`.
- `# ordo: ... showing raw output`: the model was unavailable or returned invalid YAML; the body is the
  original output (windowed around errors). Read it as a normal log.
  If it keeps happening, ask the user to run `ordo setup`, which diagnoses the model server.
- Exit codes: `run` returns the command's own exit code (127 not found, 124 timeout); 2 is a usage error;
  `search` returns 1 when nothing matches.
- `--json` on run, filter, slice, search and stats prints an envelope: `command`, `route` (model, bypass,
  fallback), `status`, `exit`, `raw_id`, `tokens`, `warnings`, and `result` (model route) or `text`.

### Working rules

1. Route long-running or verbose commands through `ordo run`; run short commands directly.
2. When the summary is not enough, use `ordo show <raw_id>` instead of re-running the command.
3. After an approach fails, record it with `ordo dead-end`; after verifying a fact, record it with `ordo note`.
4. Before a long task, after many steps, or when resuming work, read `ordo compact` and do not repeat
   any listed dead end.

### MCP

Agents with MCP support can register `ordo mcp` (command `ordo`, args `["mcp"]`; add `--no-run` for
read-only tools). Tools: ordo_run, ordo_filter, ordo_slice, ordo_search, ordo_show, ordo_dead_end, ordo_note,
ordo_compact, ordo_stats. They return the same structure as `--json`.

### Configuration

`ordo setup` checks the installation. The model server is configured by environment, not by the agent:
Ollama by default (`OLLAMA_HOST`, `ORDO_MODEL`), or an OpenAI-compatible server (llama.cpp, LM Studio, vLLM)
via `ORDO_BASE_URL` and `ORDO_API_KEY`. Session memory is stored outside the repository, in `~/.ordo`.
"""


def cmd_mcp(args) -> int:
    try:
        from .mcp_server import serve
    except ImportError:
        print("ordo mcp: the MCP SDK is not installed; run: pip install \"ordo[mcp]\"", file=sys.stderr)
        return 2
    serve(allow_run=not args.no_run)
    return 0


SETUP_LOG = """============================= test session starts ==============================
collected 3 items

tests/test_fx.py ..F                                                     [100%]

=================================== FAILURES ===================================
_______________________________ test_convert_brl _______________________________
    def test_convert_brl():
>       assert convert(10, "BRL") == 1.8
E       KeyError: 'BRL'

src/fx.py:12: KeyError
=========================== short test summary info ============================
FAILED tests/test_fx.py::test_convert_brl - KeyError: 'BRL'
========================= 1 failed, 2 passed in 0.04s ==========================
"""
HF_URL = "https://huggingface.co/devlucascfarias/ordo-3b"


def _confirm(question: str) -> bool:
    if _SINK.get() is not None or not (sys.stdin and sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        return input(f"{question} [Y/n] ").strip().lower() in ("", "y", "yes", "s", "sim")
    except EOFError:
        return False


def _pull_progress():
    """Progress for `ollama pull` events: one updating line at a terminal, a few plain lines otherwise."""
    tty = sys.stderr.isatty()
    last = {"pct": -10, "status": ""}

    def show(ev: dict) -> None:
        status, total, done = ev.get("status", ""), ev.get("total") or 0, ev.get("completed") or 0
        if total:
            pct = int(done * 100 / total)
            if tty:
                print(f"\r  {status[:40]:<40} {done / 1e9:5.2f}/{total / 1e9:.2f} GB {pct:3d}%", end="",
                      file=sys.stderr, flush=True)
            elif pct >= last["pct"] + 10:
                print(f"  {status[:40]} {pct}%", file=sys.stderr, flush=True)
                last["pct"] = pct
        elif status != last["status"]:
            print(("\n" if tty and last["status"].startswith("pulling") else "") + f"  {status}", file=sys.stderr,
                  flush=True)
        last["status"] = status

    return show


def cmd_setup(args) -> int:
    """Check the model server, install the model if missing and run one test call."""
    from .client import (PULL_NAME, find_ordo_model, installed_models, match_ordo, normalize_base_url,
                         normalize_host, pull_model, served_models)

    report: dict = {"backend": "openai" if args.base_url else "ollama", "server": None, "model": None, "test": None}
    lines: list[str] = []
    human = _SINK.get() is None and sys.stdout.isatty()  # colors only for a person at a terminal

    def say(text: str, style: str = "", tag_style: str = "") -> None:
        if not human:
            _print(text)
            return
        from rich.console import Console
        from rich.text import Text

        t = Text(text, style=style)
        if tag_style and text.startswith("["):
            t.stylize(tag_style, 0, text.index("]") + 1)
        Console(highlight=False, soft_wrap=True).print(t)

    def step(ok: bool | None, text: str) -> None:
        lines.append(f"[{'ok' if ok else 'skip' if ok is None else 'fail'}] {text}")
        if not args.json:
            say(lines[-1], tag_style={True: "bold green", False: "bold red", None: "yellow"}[ok])

    def done(code: int, *hints: str) -> int:
        report["ok"] = code == 0
        report["hints"] = list(hints)
        if args.json:
            _print(json.dumps(report, ensure_ascii=False, indent=1))
        else:
            for h in hints:
                say(f"  {h}", style="dim")
        return code

    name = None
    if args.base_url:
        base = normalize_base_url(args.base_url)
        report["server"] = base
        try:
            names = served_models(base)
        except Exception as e:  # noqa: BLE001
            step(False, f"server: cannot reach {base} ({type(e).__name__})")
            return done(1, "start your server, e.g.: llama-server -m ordo-3b.gguf -c 16384 --port 8080",
                        f"the GGUF is in {HF_URL} (ordo-3b.gguf)")
        step(True, f"server: {base} (OpenAI-compatible, {len(names)} model(s) listed)")
        name = args.model if args.model in names else match_ordo(names) or (names[0] if len(names) == 1 else None)
        if not names:
            name = args.model
        if name is None:
            step(False, f"model: no Ordo model among {', '.join(names[:6])}")
            return done(1, f"load ordo-3b.gguf from {HF_URL}, or set ORDO_MODEL to the served name")
    else:
        host = normalize_host(args.host)
        report["server"] = host
        try:
            names = installed_models(host)
        except Exception:  # noqa: BLE001
            step(False, f"server: Ollama is not reachable at {host}")
            if shutil.which("ollama"):
                return done(1, "Ollama is installed but not running: start it with `ollama serve` or open the app")
            return done(1, "install Ollama from https://ollama.com/download",
                        "or use another server: set ORDO_BASE_URL (llama.cpp, LM Studio, vLLM)")
        step(True, f"server: Ollama at {host} ({len(names)} model(s) installed)")
        name = args.model if args.model in names else find_ordo_model(host)
        if name is None:
            question = f"Download {PULL_NAME} (about 6.2 GB) with Ollama?"
            if not (args.yes or _confirm(question)):
                step(False, "model: Ordo is not installed")
                return done(1, "run `ordo setup --yes` to download it, or `ollama pull " + PULL_NAME + "`")
            if not args.json:
                say(f"downloading {PULL_NAME} ...", style="cyan")
            try:
                pull_model(host, PULL_NAME, progress=None if args.json else _pull_progress())
            except Exception as e:  # noqa: BLE001
                step(False, f"model: download failed ({e})")
                return done(1, f"retry with `ollama pull {PULL_NAME}`")
            if not args.json and sys.stderr.isatty():
                print(file=sys.stderr)
            name = find_ordo_model(host) or PULL_NAME
    report["model"] = name
    step(True, f"model: {name}")

    if args.no_test:
        step(None, "test call skipped (--no-test)")
        return done(0, "next: `ordo guide >> AGENTS.md`, or register `ordo mcp` in your agent")
    if not args.json:
        say("running a test call (the first one loads the model and can take a minute) ...", style="cyan")
    t0 = time.time()
    try:
        res = call_ordo(GcMode.TOOL_OUTPUT, "Which test fails and why?", SETUP_LOG, model=name, host=args.host,
                        base_url=args.base_url, timeout=args.timeout)
    except Exception as e:  # noqa: BLE001
        report["test"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        step(False, f"test call failed ({type(e).__name__}: {e})")
        return done(1, "check the server logs; with llama.cpp use a context of at least 16384 (-c 16384)")
    parsed = parse_output(res.text, GcMode.TOOL_OUTPUT)
    cites = "src/fx.py" in res.text
    ok = parsed.yaml_ok and str((parsed.data or {}).get("status", "")).upper() == "FAIL"
    report["test"] = {"ok": ok, "seconds": round(time.time() - t0, 1), "yaml": parsed.yaml_ok,
                      "schema": parsed.schema_ok, "cites_failing_file": cites}
    step(ok, f"test call: {time.time() - t0:.1f}s, " + ("valid YAML" if parsed.yaml_ok else "invalid YAML")
         + (", status FAIL" if ok else "") + (", cites src/fx.py" if cites else ""))
    if not ok:
        return done(1, "the server answered but the output is not what Ordo produces; check the model name and "
                       "the chat template")
    return done(0, "next: `ordo guide >> AGENTS.md`, or register `ordo mcp` in your agent")


def cmd_guide(args) -> int:
    """Instructions to paste into the project's AGENTS.md.

    Plain Markdown when piped, redirected, called through MCP or with --raw (what agents read and what goes
    into AGENTS.md); rendered with rich only for a person at an interactive terminal.
    """
    interactive = _SINK.get() is None and sys.stdout.isatty() and not args.raw
    if not interactive:
        _print(GUIDE)
        return 0
    from rich.console import Console
    from rich.markdown import Heading, Markdown
    from rich.text import Text

    class _LeftHeading(Heading):  # rich centers headings; a reference text reads better left-aligned
        def __rich_console__(self, console, options):
            self.text.justify = "left"
            if self.tag != "h2":
                yield Text("")
            yield self.text

    class _Guide(Markdown):
        elements = {**Markdown.elements, "heading_open": _LeftHeading}

    console = Console()
    console.width = min(console.width, 110)
    console.print(_Guide(GUIDE, code_theme="ansi_dark", inline_code_theme="ansi_dark"))
    console.print(Text("\nplain Markdown for AGENTS.md: ordo guide --raw >> AGENTS.md", style="dim"))
    return 0


TOP_HELP = f"""ordo {__version__} - context filter for coding agents (Ordo 3B, local model)

usage: ordo <command> [options]

commands
├─ filter output
│  ├─ run       -i INTENT -- CMD...        run CMD, print causal YAML; exits with CMD's exit code
│  └─ filter    -i INTENT [FILE]           filter text from FILE or stdin (--mode to pick the mode)
├─ inspect code
│  ├─ slice     -i INTENT FILE             relevant symbols with exact 1-based line ranges
│  └─ search    -i INTENT PATTERN [PATH]   ripgrep (or built-in) search filtered by intent
├─ recover the original
│  └─ show      [ID] [--grep RX|--lines A-B]  exact lines of a saved output (default: last)
├─ session memory
│  ├─ dead-end  ATTEMPT REASON             record an approach that failed (do not repeat it)
│  ├─ note      TEXT                       record a confirmed fact
│  ├─ compact   -i GOAL                    state ledger: goal, facts, dead ends, file states
│  └─ stats                                token savings for this session
├─ setup     [--yes]                       check the server, download the model, run a test call
├─ guide                                   instructions for any coding agent (AGENTS.md)
└─ mcp       [--no-run]                    serve these tools over MCP (stdio) for coding agents

output
  stdout is YAML (status, summary, causal_trace / relevant_symbols, exact_snippets).
  '# ordo: ...' lines are warnings; '# ordo: unverified: ...' marks citations not found in the input.
  the header ends with raw=<id>: 'ordo show <id>' prints the full original output (nothing is lost).
  short output (<=40 lines) is passed through; if the model is down, the raw output is printed.
  --json (run, filter, slice, search, stats) prints a machine-readable envelope with result and raw_id.

exit codes
  run: the command's own code (127 not found, 124 --cmd-timeout)   2: usage error
  3: model failed with --no-fallback   search: 1 when nothing matches

environment
  ORDO_MODEL (ordo:3b)   OLLAMA_HOST (127.0.0.1:11434)   ORDO_MAX_INPUT_TOKENS (6000)
  ORDO_SESSION (one per repository)   ORDO_HOME (~/.ordo)   ORDO_KEEP_ALIVE (30m)
  ORDO_BASE_URL + ORDO_API_KEY: use an OpenAI-compatible server (llama.cpp, LM Studio, vLLM) instead of Ollama

examples (run inside your project; paths are placeholders)
  ordo run -i "why does the build fail?" -- npm run build
  ordo slice -i "where is the retry delay computed?" src/client.py
  ordo search -i "who calls parse_header?" parse_header src/
  pytest -q 2>&1 | ordo filter -i "which test fails and why?"

'ordo <command> -h' shows the options of a command.
"""

COMMON_HELP = ("common options: --json --model --host --base-url --budget --timeout --session --force "
               "--no-fallback")


_SECTIONS = ("usage:", "commands", "output", "exit codes", "environment", "examples")


def _styled_top_help():
    """TOP_HELP with colors for a person at a terminal; same text and layout as the plain version."""
    from rich.text import Text

    out = Text()
    for i, line in enumerate(TOP_HELP.rstrip("\n").split("\n")):
        t = Text(line)
        tree = re.match(r"^([│├└─ ]+)([a-z-]+)(.*)$", line)
        if i == 0:
            t.stylize("bold cyan", 0, 4)
            t.stylize("dim", line.index(" - ") if " - " in line else len(line))
        elif line.startswith(_SECTIONS):
            t.stylize("bold yellow", 0, len(line) if not line.startswith("usage:") else 6)
        elif tree:
            t.stylize("dim", 0, tree.end(1))
            parts = re.split(r"(\s{2,})", tree.group(3))
            if len(parts) < 3:  # group label ("filter output")
                t.stylize("bold", tree.start(2))
            else:
                t.stylize("bold cyan", tree.start(2), tree.end(2))
                args_end = tree.start(3) + len("".join(parts[:-2]))
                t.stylize("green", tree.start(3), args_end)
        elif line.startswith("  ") and re.match(r"\s+(ordo|pytest)\b", line):
            t.stylize("green", 2)
        else:
            t.highlight_regex(r"\b[A-Z][A-Z_]{2,}\b", "cyan")  # environment variables
            t.highlight_regex(r"'[^']+'|--[a-z][a-z-]*|raw=<id>", "cyan")
        if line.startswith("'ordo <command> -h'"):
            t.stylize("dim")
        out.append_text(t)
        out.append("\n")
    return out


def print_top_help() -> None:
    """Plain text when piped, redirected or captured (agents read it); colored at an interactive terminal."""
    if _SINK.get() is None and sys.stdout.isatty():
        from rich.console import Console

        Console(highlight=False, soft_wrap=True).print(_styled_top_help(), end="")
        return
    _print(TOP_HELP)


class _Parser(argparse.ArgumentParser):
    def format_help(self) -> str:
        return TOP_HELP if self.prog == "ordo" else super().format_help()

    def print_help(self, file=None) -> None:
        if self.prog == "ordo" and file is None:
            print_top_help()
            return
        super().print_help(file)


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("common options")
    g.add_argument("--model", default=DEFAULT_MODEL, help="model name (env ORDO_MODEL, default ordo:3b)")
    g.add_argument("--host", default=DEFAULT_HOST, help="Ollama URL (env OLLAMA_HOST)")
    g.add_argument("--base-url", default=DEFAULT_BASE_URL,
                   help="OpenAI-compatible server instead of Ollama: llama.cpp, LM Studio, vLLM (env ORDO_BASE_URL)")
    g.add_argument("--session", help="session name (env ORDO_SESSION; default: one per repository)")
    g.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                   help="max input tokens sent to the model (env ORDO_MAX_INPUT_TOKENS)")
    g.add_argument("--timeout", type=float, default=300.0, help="model timeout in seconds")
    g.add_argument("--no-fallback", action="store_true", help="exit 3 instead of printing raw output")
    g.add_argument("--force", action="store_true", help="always call the model, even for short outputs")
    g.add_argument("--json", action="store_true", help="machine-readable JSON envelope instead of YAML")

    p = _Parser(prog="ordo", add_help=True)
    p.add_argument("--version", action="version", version=f"ordo {__version__}")
    sub = p.add_subparsers(dest="cmd", parser_class=argparse.ArgumentParser)
    fmt = argparse.RawDescriptionHelpFormatter

    usages = {"run": "-i INTENT [--cmd-timeout S] -- CMD...", "filter": "-i INTENT [--mode MODE] [FILE]",
              "slice": "-i INTENT FILE", "search": "-i INTENT [-C N] PATTERN [PATH...]",
              "dead-end": "ATTEMPT REASON", "note": "TEXT", "compact": "-i GOAL [--from FILE] [--neural]",
              "stats": "", "mcp": "[--no-run]", "show": "[ID|last] [--lines A-B] [--grep RX] [-C N] [--tail N]", "guide": "[--raw]", "setup": "[--yes] [--no-test]"}

    def add(name: str, summary: str, examples: list[str], func) -> argparse.ArgumentParser:
        s = sub.add_parser(name, parents=[common], help=summary, description=summary, formatter_class=fmt,
                           usage=f"ordo {name} {usages[name]} [common options]",
                           epilog="examples:\n" + "\n".join(f"  {e}" for e in examples))
        s.set_defaults(func=func)
        return s

    s = add("run", "run a command and print its output as causal YAML (TOOL_OUTPUT); exits with the command's code",
            ['ordo run -i "why do the tests fail?" -- pytest -q',
             'ordo run -i "what broke the build?" --cmd-timeout 600 -- npm run build'], cmd_run)
    s.add_argument("-i", "--intent", help="what the agent wants to know (default: the command's outcome)")
    s.add_argument("--cmd-timeout", type=float, default=None, help="kill the command after N seconds (exit 124)")
    s.add_argument("--cwd", default=None, help="run the command in this directory")
    s.add_argument("command", nargs=argparse.REMAINDER, help="command to run; put -- before it")

    s = add("filter", "filter text from a file or stdin",
            ['pytest -q 2>&1 | ordo filter -i "which test fails?"',
             'ordo filter -i "summarize the failure" build.log',
             'ordo filter --mode CODE_SLICE -i "where is the cache invalidated?" app.py'], cmd_filter)
    s.add_argument("-i", "--intent", help="what the agent wants to know")
    s.add_argument("--mode", default="TOOL_OUTPUT", choices=[m.value for m in GcMode], help="default: TOOL_OUTPUT")
    s.add_argument("file", nargs="?", help="file to read (default: stdin)")

    s = add("slice", "relevant symbols of a source file with exact 1-based line ranges (CODE_SLICE)",
            ['ordo slice -i "where is the retry delay computed?" src/client.py',
             'ordo slice -i "find Parser.parse_header" lib/http.py'], cmd_slice)
    s.add_argument("-i", "--intent", required=True, help="what to locate (name it, or describe the behavior)")
    s.add_argument("file", help="source file")

    s = add("search", "repository search (ripgrep, or built-in) filtered by intent",
            ['ordo search -i "who calls parse_header?" parse_header src/',
             'ordo search -C 4 -i "where is the timeout set?" "timeout\\s*=" .'], cmd_search)
    s.add_argument("-i", "--intent", help="what the agent wants to find")
    s.add_argument("-C", "--context", type=int, default=2, help="context lines around matches (default 2)")
    s.add_argument("pattern", help="regex (smart case)")
    s.add_argument("paths", nargs="*", help="files or folders (default: .)")

    s = add("dead-end", "record an approach that failed, so it is not tried again",
            ['ordo dead-end "raised the timeout to 30s" "still hangs: the channel is never closed"'], cmd_dead_end)
    s.add_argument("attempt", help="what was tried")
    s.add_argument("reason", help="why it failed")

    s = add("note", "record a confirmed fact for the next compact",
            ['ordo note "pool exhausted when batch > 50 (src/pool.py:42)"'], cmd_note)
    s.add_argument("text", help="the fact, with paths and lines when possible")

    s = add("compact", "state ledger of the session: goal, facts, dead ends, file states (MEMORY_COMPACT)",
            ['ordo compact -i "stabilize the payment client"',
             'ordo compact -i "fix the import" --from transcript.txt'], cmd_compact)
    s.add_argument("-i", "--intent", required=True, help="the active goal")
    s.add_argument("--from", dest="from_file", help="also compact an agent transcript with the model ('-' = stdin)")
    s.add_argument("--neural", action="store_true",
                   help="ask the model instead of building the ledger from session records (experimental)")

    s = add("stats", "token savings telemetry for this session", ["ordo stats", "ordo stats --json"], cmd_stats)

    s = add("show", "print exact lines of a saved run/filter/search output (the original is never lost)",
            ["ordo show last --grep KeyError", "ordo show 20261123-221502-ab12 --lines 120-160", "ordo show --tail 30"],
            cmd_show)
    s.add_argument("id", nargs="?", default="last", help="raw id from a header (default: last)")
    s.add_argument("--lines", help="N or A-B, 1-based")
    s.add_argument("--grep", help="regex (smart case)")
    s.add_argument("-C", "--context", type=int, default=2, help="lines around --grep matches (default 2)")
    s.add_argument("--tail", type=int, help="only the last N selected lines")

    s = add("mcp", "serve the Ordo tools over MCP (stdio) for any MCP-capable coding agent",
            ["claude mcp add ordo -- ordo mcp", "ordo mcp --no-run   # read-only tools, no command execution"],
            cmd_mcp)
    s.add_argument("--no-run", action="store_true", help="do not expose ordo_run (no command execution)")

    s = add("setup", "check the model server, download the Ordo model if missing, run a test call",
            ["ordo setup", "ordo setup --yes   # download without asking", "ordo setup --json --no-test"],
            cmd_setup)
    s.add_argument("-y", "--yes", action="store_true", help="download the model without asking")
    s.add_argument("--no-test", action="store_true", help="skip the test call")

    s = add("guide", "instructions for coding agents, to paste into AGENTS.md (rendered at a terminal, Markdown when piped)",
            ["ordo guide >> AGENTS.md", "ordo guide --raw   # plain Markdown at a terminal too"], cmd_guide)
    s.add_argument("--raw", action="store_true", help="print plain Markdown even at an interactive terminal")
    return p


def invoke(argv: list[str]) -> tuple[int, str, str]:
    """Run a CLI command and capture its output (for the MCP server): (code, stdout, stderr).

    Output goes to a per-context destination, never to the real stdout (the MCP protocol channel).
    """
    import contextlib
    import io

    out, err = io.StringIO(), io.StringIO()
    token = _SINK.set(out)
    try:
        with contextlib.redirect_stderr(err):
            try:
                code = _run(argv)
            except SystemExit as e:  # argparse and --no-fallback
                code = e.code if isinstance(e.code, int) else (0 if e.code is None else 2)
    finally:
        _SINK.reset(token)
    return code, out.getvalue(), err.getvalue()


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows: do not crash on characters outside cp1252
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    return _run(argv)


def _run(argv: list[str] | None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):  # bare `ordo`: show the help, without an error
        print_top_help()
        return 0
    args.session_obj = Session.open(args.session)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
