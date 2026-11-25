"""Ordo CLI: filters tool output, slices source files and keeps session memory for coding agents.

Principle: the CLI never leaves the agent without a useful answer. If the model fails (server down,
unrecoverable YAML), it prints the windowed raw output with a header explaining why (turn this off with
--no-fallback). Verifier warnings are YAML comments (`# ordo: ...`), so the output is still valid YAML.
"""
from __future__ import annotations

import argparse
import hashlib
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
from .contract import GcMode, number_lines
from .schema import parse_output
from .session import Session
from .textproc import clean, est_tokens, shorten_line, window_file, window_log
from .verify import verify

# prefill dominates local latency and grows faster than linearly (6.4k tokens: 62 s, 2.9k: 13 s on a mid GPU);
# the error-first window keeps the cause at 2.5k on every long benchmark log
DEFAULT_BUDGET = int(os.environ.get("ORDO_MAX_INPUT_TOKENS", "2500"))
BYPASS_LINES, BYPASS_TOKENS = 40, 400  # below this, the model spends more tokens than it saves
LONG_LINE = 800  # characters; longer lines are shortened (start + marker + end) before the model
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
    data, repeats = _collapse_repeats(parsed.data)
    if repeats:  # re-validate the collapsed structure; keep the repair flag of the original text
        repaired = parsed.repaired
        parsed = parse_output(_normalized(data), mode)
        parsed.repaired = repaired
    text = res.text.strip() if parsed.raw_yaml_ok and not repeats else _normalized(parsed.data)
    out = Outcome(text=text, raw_tokens=raw_tokens, out_tokens=res.output_tokens, seconds=res.seconds,
                  data=parsed.data)
    out.warnings += repeats
    out.status = str(parsed.data.get("status")) if parsed.data.get("status") else None
    out.summary = parsed.data.get("summary") or parsed.data.get("active_goal")
    if getattr(res, "truncated", False):
        out.warnings.append("model output hit the token limit and was cut")
    if parsed.repaired:
        out.warnings.append("output YAML was repaired")
    out.warnings += [e for e in parsed.errors]
    checked = parsed.obj or parsed.lenient
    if checked is not None:
        rep = verify(checked, raw, mode, source_lines=source_lines, strict=True)
        for c in rep.failures:
            what = {"file": "file not in input", "line": "line not in input", "snippet": "snippet not verbatim",
                    "symbol": "symbol not in cited lines", "rule": "rule broken",
                    "summary": "summary cites a location not in input"}[c.kind]
            out.warnings.append(f"unverified: {what}: {c.target}" + (f" ({c.detail})" if c.detail else ""))
    seen: set[str] = set()
    out.warnings = [w for w in out.warnings if not (w in seen or seen.add(w))]
    if out.warnings:
        out.text += "\n" + "\n".join(f"# ordo: {w}" for w in out.warnings)
    return out


def _collapse_repeats(data: dict) -> tuple[dict, list[str]]:
    """Drop identical items the model repeated (a known v1 loop on logs with many similar errors).

    Returns the cleaned data and warnings. Items that only share the same issue text are kept, but flagged
    when it repeats three or more times, since that usually means the model looped over one error.
    """
    data, warnings = dict(data), []
    for key in ("causal_trace", "relevant_symbols", "exact_snippets"):
        items = data.get(key)
        if not isinstance(items, list) or len(items) < 2:
            continue
        seen: set[str] = set()
        kept = [it for it in items
                if not ((k := json.dumps(it, sort_keys=True, default=str)) in seen or seen.add(k))]
        if len(kept) < len(items):
            warnings.append(f"repeated output: {len(items) - len(kept)} identical {key} item(s) collapsed")
            data[key] = kept
        if key != "exact_snippets":
            issues = [str(it.get("issue", "")).strip() for it in kept if isinstance(it, dict)]
            top = max(set(issues), key=issues.count) if issues else ""
            if top and issues.count(top) >= 3:
                warnings.append(f"repeated output: the same issue appears in {issues.count(top)} {key} items; "
                                "the model may have looped, check with ordo show")
    return data, warnings


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


def save_raw(text: str, session: Session) -> str:
    """Save the full output (ANSI already stripped) so the agent can recover parts with `ordo show`.

    Kept per project (`~/.ordo/runs/<project>/`, the newest RAW_KEEP), so one busy project never evicts
    another's outputs and `ordo show last` never returns another project's log.
    """
    folder = session.runs_dir()
    folder.mkdir(parents=True, exist_ok=True)
    rid = time.strftime("%Y%m%d-%H%M%S") + "-" + os.urandom(2).hex()
    (folder / f"{rid}.log").write_text(text, encoding="utf-8")
    for f in sorted(folder.glob("*.log"))[:-RAW_KEEP]:
        f.unlink(missing_ok=True)
    return rid


def migrate_flat_runs() -> int:
    """Move outputs saved before they were kept per project into the folder of the project that recorded them.

    Runs once (only while `~/.ordo/runs/*.log` exist). An output no session refers to stays where it is and
    is still found by id.
    """
    from .session import ordo_home

    runs = ordo_home() / "runs"
    flat = {p.stem: p for p in runs.glob("*.log")} if runs.exists() else {}
    moved = 0
    if not flat:
        return 0
    for spath in (ordo_home() / "sessions").glob("*.jsonl"):
        for ev in Session(path=spath, root=Path(".")).events():
            src = flat.pop(ev.get("raw_id") or "", None)
            if src is not None and src.exists():
                dest = runs / spath.stem / src.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                src.replace(dest)
                moved += 1
    return moved


def find_raw(session: Session, rid: str | None) -> tuple[Path | None, list[Path]]:
    """The saved output for `rid` (or the latest of this project) and this project's recent outputs.

    An id is looked up in this project first, then in any project, then among outputs saved before
    outputs were kept per project, so an id quoted anywhere keeps working.
    """
    from .session import ordo_home

    migrate_flat_runs()
    mine = sorted(session.runs_dir().glob("*.log")) if session.runs_dir().exists() else []
    if rid in (None, "last"):
        return (mine[-1] if mine else None), mine
    runs = ordo_home() / "runs"
    for candidate in [session.runs_dir() / f"{rid}.log", *runs.glob(f"*/{rid}.log"), runs / f"{rid}.log"]:
        if candidate.exists():
            return candidate, mine
    return None, mine


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
    lines = text.split("\n")
    long_lines = sum(len(l) > LONG_LINE for l in lines)
    if long_lines and mode is not GcMode.CODE_SLICE:  # dumps, minified code, overload lists: noise for the model
        text = "\n".join(shorten_line(l, LONG_LINE) for l in lines)
    body, cut = window_log(text, args.budget)
    out = run_model(args, mode, intent, body, raw_tokens=tokens)
    notes = []
    if cut and out.route == "model":
        notes.append(f"input exceeded ~{args.budget} tokens and was windowed before the model")
    if long_lines and mode is not GcMode.CODE_SLICE and out.route == "model":
        notes.append(f"{long_lines} very long line(s) shortened before the model; `ordo show` has them in full")
    for n in notes:
        out.warnings.append(n)
        out.text += f"\n# ordo: {n}"
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
    if os.name == "nt" and os.path.basename(exe).lower() == "cmd.exe":
        # cmd.exe does not parse the \" escapes subprocess would add: pass the command line as written
        flags = [a for a in cmd[1:] if a.startswith("/")]
        k = next((i for i, a in enumerate(cmd) if a.lower() in ("/c", "/k")), None)
        if k is not None and k + 1 < len(cmd):
            rest = " ".join(cmd[k + 1:])
            body = f'"{rest}"' if any(f.lower() == "/s" for f in flags) else rest
            return " ".join([_cmd_quote(exe)] + cmd[1:k + 1] + [body]), False
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
    elif out.route == "model" and (code == 0) != (out.status in ("PASS", "WARN")):
        # the exit code is a fact; a status that contradicts it is the model's reading and must not be trusted blindly
        note = (f"the command exited {code} but the model reported {out.status}; "
                "check the original with `ordo show`")
        out.warnings.append(note)
        out.text += f"\n# ordo: {note}"
    rid = save_raw(text, args.session_obj)
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
    rid = save_raw(text, args.session_obj) if mode is not GcMode.CODE_SLICE else None
    emit(args, "filter", out, raw_id=rid)
    args.session_obj.append("filter", intent=intent, mode=mode.value, raw_id=rid, **out.event())
    return 0


def source_lines(src: str) -> list[str]:
    """Count lines like an editor: only \\n separates; a trailing \\n does not add a line."""
    lines = src.split("\n")
    return lines[:-1] if lines and lines[-1] == "" else lines


def _slice_text(args, src: str, rel: str) -> Outcome:
    lines = source_lines(src)
    tokens = est_tokens(src)
    if not args.force and len(lines) <= BYPASS_LINES and tokens <= BYPASS_TOKENS:
        # a small file costs fewer tokens than the model's answer: show it numbered, like `run` passes short output
        out = Outcome(text=number_lines("\n".join(lines)), raw_tokens=tokens, out_tokens=tokens, route="bypass")
        out.windowed = False
        return out
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
    rid = save_raw(text, args.session_obj)
    emit(args, "search", out, f"pattern={args.pattern!r}", raw_id=rid)
    args.session_obj.append("search", pattern=args.pattern, intent=intent, raw_id=rid, **out.event())
    return 0


def cmd_dead_end(args) -> int:
    if (m := _existing(args.session_obj, "dead_end", args.attempt)):
        if m.reason != args.reason:  # same attempt, new reason: update it under the same id
            args.session_obj.append("edit", id=m.id, reason=args.reason)
        _print(f"# ordo: dead-end {m.id} already recorded in {args.session_obj.path.name}"
               + ("; reason updated" if m.reason != args.reason else ""))
        return 0
    mid = _new_id("d")
    args.session_obj.append("dead_end", id=mid, attempt=args.attempt, reason=args.reason)
    _print(f"# ordo: dead-end {mid} recorded in {args.session_obj.path.name}\n"
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
        if path and not path.endswith("/") and not (line.startswith("??") and _ARTIFACT.search(path)):
            files.append(path)
    return files


# build and cache output that shows up as untracked in a repo without a .gitignore: never source state
_ARTIFACT = re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache|\.tox|\.nox|\.venv|venv|"
                       r"node_modules|\.next|\.gradle|\.idea|\.vscode|target|dist|build|out|bin|obj|coverage|"
                       r"\.coverage[^/]*|\.eggs|[^/]+\.egg-info)(/|$)|\.(pyc|pyo|class|o|obj|so|dll|exe|pdb|log)$")


_CHECK_CMD = re.compile(_TEST_CMD.pattern + r"|\b(tsc|eslint|ruff|mypy|pyright|flake8|clippy|golangci-lint|go (build|vet)|"
                        r"cargo (build|check)|npm (run )?(build|lint)|yarn (build|lint)|pnpm (build|lint)|make|cmake|"
                        r"dotnet build|mvn\w*\s+(compile|package|verify)|gradle\w*\s+build)\b")


def _ledger_window_seconds() -> float:
    """Command results older than this are not turned into facts (session files persist across days)."""
    try:
        return float(os.environ.get("ORDO_LEDGER_HOURS", "12")) * 3600
    except ValueError:
        return 12 * 3600


def deterministic_ledger(session: Session, goal: str) -> "MemoryCompact":
    """Ledger built only from what the session recorded: nothing is inferred by a model.

    The MEMORY_COMPACT output of the published ordo-3b does not follow the schema and invents facts,
    so this is the default path of `ordo compact`.

    - Notes and dead ends come from the session memory (`ordo memory`), with edits and retractions applied.
    - Command results become facts only for tests, builds, linters and type checkers run since the last
      compact and within ORDO_LEDGER_HOURS (default 12); they are recomputed, never inherited.
    - A modified file is PASS or FAIL only if the last test run started after its last change; otherwise
      it is UNTESTED.
    """
    from .schema import DeadEnd, FileState, MemoryCompact

    mem = memory(session)
    facts: list[str] = [m.text for m in mem if m.kind == "note"]
    dead: list[DeadEnd] = [DeadEnd(attempt=m.text, reason_failed=m.reason) for m in mem if m.kind == "dead_end"]
    oldest = time.time() - _ledger_window_seconds()
    last_by_cmd: dict[str, dict] = {}
    last_test: dict | None = None
    for ev in session.since_last_compact():
        if ev["kind"] == "run" and ev.get("ts", 0) >= oldest and _CHECK_CMD.search(ev["command"]):
            # one result per tool: `pytest -v` that failed is superseded by a later `pytest -q` that passed
            last_by_cmd.pop(tool_name(ev), None)
            last_by_cmd[tool_name(ev)] = ev
            if _TEST_CMD.search(ev["command"]):
                last_test = ev
    for ev in last_by_cmd.values():
        # only what the process reported: the model's summary can be wrong and must not become a "fact"
        facts.append(f"Last `{ev['command']}`: exit {ev['exit']} ({'PASS' if ev['exit'] == 0 else 'FAIL'})")
    seen: set[str] = set()
    facts = [f for f in facts if not (f in seen or seen.add(f))]
    test_started = None if last_test is None else last_test["ts"] - (last_test.get("cmd_seconds") or 0)
    test_result = None if last_test is None else ("PASS" if last_test["exit"] == 0 else "FAIL")
    files = []
    for f in _git_changed(session.root):
        try:
            changed = (session.root / f).stat().st_mtime
        except OSError:  # deleted: no test has seen the deletion unless one ran after it, which we cannot tell
            changed = float("inf")
        status = test_result if test_started is not None and changed <= test_started else "UNTESTED"
        files.append(FileState(file=f, test_status=status))
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


def _existing(session: Session, kind: str, text: str) -> "MemItem | None":
    """An active item with the same text: recording it again would give an id `memory` never shows."""
    return next((m for m in memory(session) if m.kind == kind and m.text == text), None)


def cmd_note(args) -> int:
    if (m := _existing(args.session_obj, "note", args.text)):
        _print(f"# ordo: note {m.id} already recorded in {args.session_obj.path.name}")
        return 0
    mid = _new_id("n")
    args.session_obj.append("note", id=mid, text=args.text)
    _print(f"# ordo: note {mid} recorded in {args.session_obj.path.name}")
    return 0


@dataclass
class MemItem:
    id: str
    kind: str  # note | dead_end
    text: str  # the fact, or the attempt of a dead end
    reason: str = ""


def _new_id(prefix: str) -> str:
    return prefix + os.urandom(3).hex()


def _legacy_id(prefix: str, ev: dict, text: str) -> str:  # records written before ids existed
    return prefix + hashlib.sha1(f"{ev.get('ts')}|{text}".encode()).hexdigest()[:6]


def memory(session: Session) -> list[MemItem]:
    """Notes and dead ends with edits and retractions applied, in creation order.

    Ids are stable: assigned when the item is recorded, never renumbered. Facts that a model-based compact
    (`--neural`, `--from`) added are included too, with an id derived from their text, so they can be
    edited or retracted like any other item.
    """
    items: dict[str, MemItem] = {}

    def add(item: MemItem) -> None:
        if not any(m.kind == item.kind and m.text == item.text for m in items.values()):
            items[item.id] = item

    for ev in session.events():
        kind = ev["kind"]
        if kind == "note":
            add(MemItem(ev.get("id") or _legacy_id("n", ev, ev["text"]), "note", ev["text"]))
        elif kind == "dead_end":
            add(MemItem(ev.get("id") or _legacy_id("d", ev, ev["attempt"]), "dead_end", ev["attempt"], ev["reason"]))
        elif kind == "compact" and ev.get("route") == "model" and ev.get("ledger"):
            p = parse_output(ev["ledger"], GcMode.MEMORY_COMPACT)
            obj = p.obj or p.lenient
            for f in getattr(obj, "confirmed_facts", []) or []:
                if not f.startswith("Last `"):
                    add(MemItem("n" + hashlib.sha1(f.encode()).hexdigest()[:6], "note", f))
            for d in getattr(obj, "dead_ends_do_not_repeat", []) or []:
                add(MemItem("d" + hashlib.sha1(d.attempt.encode()).hexdigest()[:6], "dead_end", d.attempt,
                            d.reason_failed))
        elif kind == "edit" and ev.get("id") in items:
            m = items[ev["id"]]
            m.text = ev.get("text") or m.text
            m.reason = ev.get("reason") or m.reason
        elif kind == "forget":
            if ev.get("id"):
                items.pop(ev["id"], None)
            else:  # older retractions recorded the kind and text
                for k in [k for k, m in items.items() if m.kind == ev.get("target") and m.text == ev.get("text")]:
                    items.pop(k)
    return list(items.values())


def _find_memory(items: list[MemItem], key: str) -> list[MemItem]:
    key = key.strip()
    return [m for m in items if m.id == key] or [m for m in items if key.casefold() in m.text.casefold()]


def _memory_lines(items: list[MemItem]) -> str:
    return "\n".join(f"{m.id}  {m.text}" + (f"  (failed: {m.reason})" if m.kind == "dead_end" else "")
                     for m in items)


def cmd_memory(args) -> int:
    """List the session memory (notes and dead ends) with their stable ids."""
    items = memory(args.session_obj)
    if args.grep:
        rx = re.compile(args.grep, 0 if any(c.isupper() for c in args.grep) else re.IGNORECASE)
        items = [m for m in items if rx.search(m.text) or rx.search(m.reason)]
    if args.json:
        _print(json.dumps({"session": args.session_obj.path.name,
                           "items": [{"id": m.id, "kind": m.kind, "text": m.text,
                                      **({"reason": m.reason} if m.kind == "dead_end" else {})} for m in items]},
                          ensure_ascii=False, indent=1))
        return 0
    _print(f"# ordo memory | {args.session_obj.path.name} | {len(items)} item(s); n = note, d = dead end")
    if items:
        _print(_memory_lines(items))
    return 0


def cmd_edit(args) -> int:
    """Rewrite a note, or the attempt or reason of a dead end, keeping its id."""
    hits = [m for m in memory(args.session_obj) if m.id == args.id]
    if not hits:
        _print(f"# ordo edit: no item with id {args.id!r}; `ordo memory` lists the ids")
        return 1
    m = hits[0]
    if not args.text and not args.reason:
        _print("# ordo edit: give the new text, --reason, or both")
        return 2
    if args.reason and m.kind != "dead_end":
        _print(f"# ordo edit: {m.id} is a note; --reason applies to dead ends only")
        return 2
    args.session_obj.append("edit", id=m.id, **({"text": args.text} if args.text else {}),
                            **({"reason": args.reason} if args.reason else {}))
    _print(f"# ordo: edited {m.id}: {args.text or m.text}" + (f"  (failed: {args.reason})" if args.reason else ""))
    return 0


def cmd_forget(args) -> int:
    """Retract a note or a dead end that is wrong or obsolete (the record stays in the session file)."""
    items = memory(args.session_obj)
    if not args.item:
        _print("# ordo forget: give an id or text; active items:" if items else
               "# ordo forget: no notes or dead ends recorded")
        if items:
            _print(_memory_lines(items))
        return 0
    hits = _find_memory(items, args.item)
    if not hits:
        _print(f"# ordo forget: nothing matches {args.item!r}; `ordo memory` lists the ids")
        return 1
    if len(hits) > 1 and not args.all:
        _print(f"# ordo forget: {len(hits)} items match {args.item!r}; use an id, or --all to retract all of them")
        _print(_memory_lines(hits))
        return 2
    for m in hits:
        args.session_obj.append("forget", id=m.id)
        _print(f"# ordo: forgot {m.id}: {m.text}")
    return 0


_WRAPPERS = re.compile(r"^(python3?|py)(\.exe)?\s+-m\s+|^(npx|bunx|uvx)\s+|^(uv|poetry|pipenv|pdm|hatch)\s+run\s+|"
                       r"^(pnpm|yarn)\s+exec\s+|^dotnet\s+tool\s+run\s+")
_TWO_WORD = {"npm", "yarn", "pnpm", "bun", "go", "cargo", "dotnet", "mvn", "gradle", "gradlew", "mix", "bundle",
             "composer", "make", "deno"}


def tool_name(event: dict) -> str:
    """Which tool produced the output: `python -m pytest -q` -> pytest, `npm run build` -> npm run build."""
    if event["kind"] != "run":
        return {"filter": "ordo filter", "slice": "ordo slice", "search": "ordo search"}.get(event["kind"], event["kind"])
    words = _WRAPPERS.sub("", event.get("command", "").strip()).split()
    if not words:
        return "(unknown)"
    first = Path(words[0]).name.removesuffix(".exe").removesuffix(".cmd")
    if first in _TWO_WORD and len(words) > 1:
        rest = [w for w in words[1:3] if not w.startswith("-")]
        if first in ("npm", "yarn", "pnpm", "bun") and rest[:1] == ["run"]:
            return " ".join([first] + rest[:2])
        return " ".join([first] + rest[:1])
    return first


def _display_project(sess: Session) -> str:
    """Folder name of the project; `~` when the session ran in the home folder (its name is the username)."""
    root = sess.root_recorded()
    try:
        at_home = Path(root).resolve() == Path.home().resolve() if root else sess.project == Path.home().name
    except OSError:
        at_home = False
    return "~" if at_home else sess.project


def savings_report(home: Path) -> dict:
    """Token savings over every recorded session: totals, per tool and per project."""
    tools: dict[str, dict] = {}
    projects: dict[str, dict] = {}
    total = {"calls": 0, "raw": 0, "sent": 0, "filtered": 0, "model_seconds": 0.0, "bypass": 0, "fallback": 0}
    first_ts = None
    for path in sorted((home / "sessions").glob("*.jsonl")):
        sess = Session(path=path, root=Path("."))
        project = _display_project(sess)
        for ev in sess.events():
            if ev["kind"] not in ("run", "filter", "slice", "search") or "raw_tokens" not in ev:
                continue
            raw, sent = int(ev.get("raw_tokens") or 0), int(ev.get("out_tokens") or 0)
            filtered = ev.get("route") == "model"
            first_ts = min(first_ts or ev["ts"], ev["ts"])
            for row in (tools.setdefault(tool_name(ev), {"calls": 0, "raw": 0, "sent": 0, "filtered": 0}),
                        projects.setdefault(project, {"calls": 0, "raw": 0, "sent": 0, "filtered": 0}), total):
                row["calls"] += 1
                row["raw"] += raw
                row["sent"] += sent
                row["filtered"] += filtered
            total["model_seconds"] += float(ev.get("seconds") or 0) if filtered else 0.0
            if ev.get("route") in ("bypass", "fallback"):
                total[ev["route"]] += 1
            projects[project]["last"] = max(projects[project].get("last", 0), ev["ts"])
    for row in [*tools.values(), *projects.values(), total]:
        row["saved"] = row["raw"] - row["sent"]
        row["saved_pct"] = round(100 * row["saved"] / row["raw"], 1) if row["raw"] else 0.0
    for row in projects.values():
        row["last"] = time.strftime("%Y-%m-%d %H:%M", time.localtime(row["last"]))
    order = lambda d: dict(sorted(d.items(), key=lambda kv: -kv[1]["saved"]))  # noqa: E731
    return {"since": time.strftime("%Y-%m-%d", time.localtime(first_ts)) if first_ts else None,
            "total": total, "by_tool": order(tools), "by_project": order(projects)}


def cmd_dashboard(args) -> int:
    """Write a self-contained HTML page with the savings across projects and open it in the browser."""
    import webbrowser

    from .dashboard import render
    from .session import ordo_home

    home = ordo_home()
    path = home / "dashboard.html"
    home.mkdir(parents=True, exist_ok=True)
    mem: dict[str, list[dict]] = {}
    for f in sorted((home / "sessions").glob("*.jsonl")):
        sess = Session(path=f, root=Path("."))
        mem.setdefault(_display_project(sess), []).extend(
            {"id": m.id, "kind": m.kind, "text": m.text, "reason": m.reason} for m in memory(sess))
    path.write_text(render(savings_report(home), mem), encoding="utf-8")
    _print(f"# ordo dashboard: {path}")
    if not args.no_open and not webbrowser.open(path.resolve().as_uri()):
        _print("# ordo dashboard: no browser available here; open the file above in one")
    return 0


def cmd_savings(args) -> int:
    """Tokens Ordo kept out of the agent's context, across all projects, per tool and per project."""
    from .session import ordo_home

    rep = savings_report(ordo_home())
    if args.json:
        _print(json.dumps(rep, ensure_ascii=False, indent=1))
        return 0
    t = rep["total"]
    if not t["calls"]:
        _print("# ordo savings: no tool output recorded yet (run `ordo run -i ... -- <command>` first)")
        return 0
    head = (f"Ordo kept {t['saved']:,} tokens out of your agent's context ({t['saved_pct']:.0f}% of "
            f"{t['raw']:,}) across {t['calls']} tool calls in {len(rep['by_project'])} project(s) since {rep['since']}.")
    note = (f"{t['filtered']} call(s) were condensed by the model ({t['model_seconds'] / 60:.1f} min of model "
            "time); the rest were short and passed through. Token counts are estimates (~3.5 characters per token).")
    k = args.top
    sections = [("by tool", list(rep["by_tool"].items())[:k]), ("by project", list(rep["by_project"].items())[:k])]
    if _SINK.get() is None and sys.stdout.isatty():
        from rich.console import Console
        from rich.table import Table
        from rich.text import Text

        con = Console(highlight=False)
        con.print(Text(head, style="bold"))
        for title, rows in sections:
            tb = Table(title=title, title_justify="left", title_style="bold yellow", box=None, padding=(0, 2))
            for col in (title.split()[-1], "calls", "raw tokens", "sent", "saved"):
                tb.add_column(col, justify="left" if col in ("tool", "project") else "right",
                              style="cyan" if col in ("tool", "project") else ("green" if col == "saved" else ""))
            for name, r in rows:
                tb.add_row(name, str(r["calls"]), f"{r['raw']:,}", f"{r['sent']:,}",
                           f"{r['saved']:,} ({r['saved_pct']:.0f}%)")
            con.print(tb)
        con.print(Text(note, style="dim"))
        return 0
    lines = [head, ""]
    for title, rows in sections:
        w = max([len(n) for n, _ in rows] + [len(title.split()[-1])])
        lines.append(f"{title}:")
        lines.append(f"  {title.split()[-1]:<{w}}  {'calls':>6}  {'raw tokens':>11}  {'sent':>9}  saved")
        lines += [f"  {n:<{w}}  {r['calls']:>6}  {r['raw']:>11,}  {r['sent']:>9,}  {r['saved']:,} ({r['saved_pct']:.0f}%)"
                  for n, r in rows]
        lines.append("")
    _print("\n".join(lines + [note]))
    return 0


def project_sessions() -> list[dict]:
    """Every recorded session: project, root, last activity, saved outputs, and whether the root still exists."""
    from .session import ordo_home

    out = []
    for spath in sorted((ordo_home() / "sessions").glob("*.jsonl")):
        sess = Session(path=spath, root=Path("."))
        last = max((ev.get("ts", 0) for ev in sess.events()), default=spath.stat().st_mtime)
        runs = sess.runs_dir()
        root = sess.root_recorded()
        out.append({"session": sess.key, "project": sess.project, "root": root,
                    "root_exists": None if root is None else Path(root).exists(),
                    "last_activity": time.strftime("%Y-%m-%d", time.localtime(last)), "_last": last,
                    "outputs": len(list(runs.glob("*.log"))) if runs.exists() else 0,
                    "kb": round((spath.stat().st_size + sum(f.stat().st_size for f in runs.glob("*.log"))
                                 if runs.exists() else spath.stat().st_size) / 1024)})
    return out


def cmd_prune(args) -> int:
    """List projects with recorded memory; remove those whose folder is gone or that are inactive."""
    items = project_sessions()
    cutoff = time.time() - args.older_than * 86400 if args.older_than else None
    for it in items:
        reasons = []
        if args.missing and it["root_exists"] is False:
            reasons.append("folder no longer exists")
        if cutoff is not None and it["_last"] < cutoff:
            reasons.append(f"inactive for more than {args.older_than} days")
        it["remove"] = reasons
    targets = [it for it in items if it["remove"]]
    criteria = args.missing or args.older_than
    if args.json and not args.yes:
        _print(json.dumps({"projects": [{k: v for k, v in it.items() if not k.startswith("_")} for it in items],
                           "would_remove": [it["session"] for it in targets]}, ensure_ascii=False, indent=1))
        return 0
    if not args.json:
        _print(f"# ordo prune | {len(items)} project(s) with recorded memory")
        for it in items:
            where = it["root"] or "(root not recorded yet)"
            gone = " [folder missing]" if it["root_exists"] is False else ""
            mark = "  -> remove: " + "; ".join(it["remove"]) if it["remove"] else ""
            _print(f"{it['session']}  last {it['last_activity']}  {it['outputs']} output(s)  {it['kb']} KB  "
                   f"{where}{gone}{mark}")
    if not criteria:
        if not args.json:
            _print("  remove with --missing (folder gone) and/or --older-than DAYS; nothing is removed without them")
        return 0
    if not targets:
        if not args.json:
            _print("  nothing matches")
        return 0
    if not (args.yes or _confirm(f"Remove the memory and saved outputs of {len(targets)} project(s)?")):
        if not args.json:
            _print("  nothing removed (dry run); add --yes to remove")
        return 0
    from .session import ordo_home

    for it in targets:
        (ordo_home() / "sessions" / f"{it['session']}.jsonl").unlink(missing_ok=True)
        runs = ordo_home() / "runs" / it["session"]
        if runs.exists():
            shutil.rmtree(runs, ignore_errors=True)
    msg = f"removed {len(targets)} project(s): " + ", ".join(it["session"] for it in targets)
    _print(json.dumps({"removed": [it["session"] for it in targets]}) if args.json else f"# ordo: {msg}")
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
    target, logs = find_raw(args.session_obj, args.id)
    if target is None and args.id in (None, "last"):
        print("ordo show: no saved output in this project yet (run, filter and search save theirs)", file=sys.stderr)
        return 1
    if target is None:
        recent = ", ".join(p.stem for p in logs[-5:]) or "none in this project"
        print(f"ordo show: unknown id {args.id!r}; recent: {recent}", file=sys.stderr)
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


# what an agent needs on every turn (agents resend AGENTS.md with each step, so every line here is paid many times)
GUIDE_SHORT = """## Ordo: tool-output filter

Run verbose commands (tests, builds, linters) through `ordo run -i "<what you need to know>" -- <command>`:
it prints a short YAML (status, summary, file and line, snippets) instead of the full log. Run short commands
directly.

- `ordo show <raw_id> --grep "<regex>"` (or `--lines A-B`) prints exact original lines; use it instead of
  re-running the command.
- `# ordo: unverified: ...` marks a citation not found in the output: confirm it with `ordo show` first.
- If ordo says it is showing raw output, read the body as a normal log.
- Memory across sessions: `ordo note "<fact>"`, `ordo dead-end "<attempt>" "<why it failed>"`,
  `ordo compact -i "<goal>"` to read it back; `ordo memory`, `ordo edit`, `ordo forget` to keep it true.

`ordo guide --full` lists every command and the output contract.
"""

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
| List the memory with stable ids | `ordo memory` (or `ordo memory --grep "<regex>"`) |
| Correct a note or dead end | `ordo edit <id> "<new text>"` (dead ends: `--reason "<new reason>"`) |
| Retract a wrong or obsolete one | `ordo forget <id>` |
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
   Keep the memory true: when a fact or dead end changes, `ordo edit` it; when it stops applying,
   `ordo forget` it. Ids are stable, so an id read earlier still points to the same item.
4. Before a long task, after many steps, or when resuming work, read `ordo compact` and do not repeat
   any listed dead end.

### MCP

Agents with MCP support can register `ordo mcp` (command `ordo`, args `["mcp"]`; add `--no-run` for
read-only tools). Tools: ordo_run, ordo_filter, ordo_slice, ordo_search, ordo_show, ordo_dead_end, ordo_note,
ordo_memory, ordo_edit, ordo_forget, ordo_compact,
ordo_stats. They return the same structure as `--json`.

### Configuration

`ordo setup` checks the installation. The model server is configured by environment, not by the agent:
Ollama by default (`OLLAMA_HOST`, `ORDO_MODEL`), or an OpenAI-compatible server (llama.cpp, LM Studio, vLLM)
via `ORDO_BASE_URL` and `ORDO_API_KEY`. Session memory is stored outside the repository, in `~/.ordo`.
"""


def cmd_mcp(args) -> int:
    try:
        from .mcp_server import serve
    except ImportError:
        print("ordo mcp: the MCP SDK is not installed; run: pip install \"ordo-harness[mcp]\"", file=sys.stderr)
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
    # a model the user named (--model or ORDO_MODEL) is used as is; otherwise the preference order picks one
    named = args.model if (args.model != DEFAULT_MODEL or os.environ.get("ORDO_MODEL")) else None
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
            return done(1, "start your server, e.g.: llama-server -m ordo-3b-Q4_K_M.gguf -c 16384 --port 8080",
                        f"the GGUF is in {HF_URL} (ordo-3b-Q4_K_M.gguf)")
        step(True, f"server: {base} (OpenAI-compatible, {len(names)} model(s) listed)")
        name = named if named in names else match_ordo(names) or (names[0] if len(names) == 1 else None)
        if not names:
            name = args.model
        if name is None:
            step(False, f"model: no Ordo model among {', '.join(names[:6])}")
            return done(1, f"load ordo-3b-Q4_K_M.gguf from {HF_URL}, or set ORDO_MODEL to the served name")
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
        name = named if named in names else find_ordo_model(host)
        if name is None:
            question = f"Download {PULL_NAME} (about 1.9 GB) with Ollama?"
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
    text = GUIDE if args.full else GUIDE_SHORT
    interactive = _SINK.get() is None and sys.stdout.isatty() and not args.raw
    if not interactive:
        _print(text)
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
    console.print(_Guide(text, code_theme="ansi_dark", inline_code_theme="ansi_dark"))
    console.print(Text("\nplain Markdown for AGENTS.md: ordo guide --raw >> AGENTS.md"
                       + ("" if args.full else "   (every command: ordo guide --full)"), style="dim"))
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
│  ├─ memory    [--grep RX]                list notes and dead ends with stable ids
│  ├─ edit      ID [TEXT] [--reason R]     rewrite a note or dead end, keeping its id
│  ├─ forget    ID|TEXT                    retract a wrong or obsolete note or dead end
│  ├─ compact   -i GOAL                    state ledger: goal, facts, dead ends, file states
│  ├─ stats                                token savings for this session
│  ├─ savings   [--top N]                  tokens saved across all projects, per tool
│  ├─ dashboard [--no-open]                page with the savings per project, in the browser
│  └─ prune     [--missing|--older-than D] memory of all projects; remove stale ones
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
  ORDO_MODEL (installed Ordo)   OLLAMA_HOST (127.0.0.1:11434)   ORDO_MAX_INPUT_TOKENS (2500)
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


# words people try that are not commands, and what they probably meant
_ALIASES = {"usage": ["savings", "stats"], "home": ["dashboard"], "ui": ["dashboard"], "status": ["stats"], "help": ["guide"], "install": ["setup"],
            "init": ["setup"], "doctor": ["setup"], "check": ["setup"], "list": ["memory"], "ls": ["memory"],
            "notes": ["memory"], "delete": ["forget"], "remove": ["forget"], "rm": ["forget"], "update": ["edit"],
            "log": ["show"], "logs": ["show"], "cat": ["show"], "clean": ["prune"], "test": ["run"],
            "exec": ["run"], "grep": ["search"], "find": ["search"], "summary": ["compact"], "server": ["mcp"]}


def _suggestions(message: str, choices: list[str]) -> tuple[str | None, list[str]]:
    """For an unknown command, the word typed and the closest real commands."""
    import difflib

    m = re.search(r"invalid choice: '([^']*)'", message)
    if not m:
        return None, []
    word = m.group(1)
    found = _ALIASES.get(word.lower(), []) + difflib.get_close_matches(word.lower(), choices, n=2, cutoff=0.6)
    return word, list(dict.fromkeys(c for c in found if c in choices))


class _ErrorStyle:
    """argparse errors: colored with a suggestion for a person at a terminal, plain text otherwise. Exit 2."""

    def error(self, message: str):  # noqa: D401
        choices = sorted(next((list(a.choices) for a in self._actions
                               if isinstance(a, argparse._SubParsersAction)), []))
        word, hints = _suggestions(message, choices)
        if word is not None:
            head = f"unknown command '{word}'"
            tips = [f"did you mean: {', '.join(f'ordo {h}' for h in hints)}"] if hints else []
            tips.append("run `ordo` to see all commands")
        elif self.prog == "ordo":  # e.g. an option the command does not take: argparse reports it here
            cmd = next((a for a in sys.argv[1:] if a in choices), None) if _SINK.get() is None else None
            head = message
            tips = [f"run `ordo {cmd} -h` for its options" if cmd else "run `ordo <command> -h` for its options"]
        else:
            head = message
            tips = [f"usage: {self.format_usage().replace('usage: ', '').strip()}",
                    f"run `{self.prog} -h` for the options"]
        if _SINK.get() is None and sys.stderr.isatty():
            from rich.console import Console
            from rich.text import Text

            con = Console(stderr=True, highlight=False, soft_wrap=True)
            con.print(Text.assemble(("ordo: ", "bold"), ("error: ", "bold red"), head))
            for t in tips:
                con.print(Text("  " + t, style="cyan" if t.startswith("did you mean") else "dim"))
        else:
            print(f"ordo: error: {head}", file=sys.stderr)
            for t in tips:
                print(f"  {t}", file=sys.stderr)
        raise SystemExit(2)


class _SubParser(_ErrorStyle, argparse.ArgumentParser):
    pass


class _Parser(_ErrorStyle, argparse.ArgumentParser):
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
    g.add_argument("--model", default=DEFAULT_MODEL,
                   help="model name (env ORDO_MODEL; default: the installed Ordo, preferring the Hugging Face pull)")
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
    sub = p.add_subparsers(dest="cmd", parser_class=_SubParser)
    fmt = argparse.RawDescriptionHelpFormatter

    usages = {"run": "-i INTENT [--cmd-timeout S] -- CMD...", "filter": "-i INTENT [--mode MODE] [FILE]",
              "slice": "-i INTENT FILE", "search": "-i INTENT [-C N] PATTERN [PATH...]",
              "dead-end": "ATTEMPT REASON", "note": "TEXT", "forget": "ID|TEXT [--all]", "memory": "[--grep RX] [--json]", "savings": "[--top N] [--json]", "dashboard": "[--no-open]", "prune": "[--missing] [--older-than DAYS] [--yes]", "edit": "ID [TEXT] [--reason R]", "compact": "-i GOAL [--from FILE] [--neural]",
              "stats": "", "mcp": "[--no-run]", "show": "[ID|last] [--lines A-B] [--grep RX] [-C N] [--tail N]", "guide": "[--full] [--raw]", "setup": "[--yes] [--no-test]"}

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

    s = add("memory", "list the session memory (notes and dead ends) with stable ids",
            ["ordo memory", 'ordo memory --grep "quota"', "ordo memory --json"], cmd_memory)
    s.add_argument("--grep", help="regex on the text and reason (smart case)")

    s = add("edit", "rewrite a note, or the attempt or reason of a dead end, keeping its id",
            ['ordo edit n3f9a1c "cache is keyed by item and model"', 'ordo edit d07b2e4 --reason "fixed in 85bcce4"'],
            cmd_edit)
    s.add_argument("id", help="id from `ordo memory`")
    s.add_argument("text", nargs="?", help="new text (the fact, or the attempt of a dead end)")
    s.add_argument("--reason", help="new reason (dead ends only)")

    s = add("forget", "retract a note or dead end that is wrong or obsolete",
            ["ordo forget n3f9a1c", 'ordo forget "Gemini quota"   # by text, if it matches one item'], cmd_forget)
    s.add_argument("item", nargs="?", help="id from `ordo memory` or text contained in exactly one item")
    s.add_argument("--all", action="store_true", help="retract every item that matches the text")

    s = add("compact", "state ledger of the session: goal, facts, dead ends, file states (MEMORY_COMPACT)",
            ['ordo compact -i "stabilize the payment client"',
             'ordo compact -i "fix the import" --from transcript.txt'], cmd_compact)
    s.add_argument("-i", "--intent", required=True, help="the active goal")
    s.add_argument("--from", dest="from_file", help="also compact an agent transcript with the model ('-' = stdin)")
    s.add_argument("--neural", action="store_true",
                   help="ask the model instead of building the ledger from session records (experimental)")

    s = add("stats", "token savings telemetry for this session", ["ordo stats", "ordo stats --json"], cmd_stats)

    s = add("savings", "tokens Ordo kept out of the agent's context, across all projects, per tool and project",
            ["ordo savings", "ordo savings --top 5", "ordo savings --json"], cmd_savings)
    s.add_argument("--top", type=int, default=10, help="rows per table (default 10)")

    s = add("dashboard", "write a self-contained HTML page with the token savings per project and open it",
            ["ordo dashboard", "ordo dashboard --no-open"], cmd_dashboard)
    s.add_argument("--no-open", action="store_true", help="only write the file (for SSH or WSL)")

    s = add("prune", "list projects with recorded memory; remove those whose folder is gone or that are inactive",
            ["ordo prune", "ordo prune --missing", "ordo prune --older-than 90 --yes"], cmd_prune)
    s.add_argument("--missing", action="store_true", help="select projects whose folder no longer exists")
    s.add_argument("--older-than", type=int, metavar="DAYS", help="select projects inactive for more than DAYS")
    s.add_argument("-y", "--yes", action="store_true", help="remove without asking (otherwise: ask, or dry run)")

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
            ["ordo guide >> AGENTS.md", "ordo guide --full   # every command and the output contract",
             "ordo guide --raw   # plain Markdown at a terminal too"], cmd_guide)
    s.add_argument("--full", action="store_true", help="the complete reference instead of the short guide")
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
