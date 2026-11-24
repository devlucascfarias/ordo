"""Session memory and telemetry, kept outside the user's repository.

Each repository (git root, or the current folder) gets a JSONL file in
`~/.ordo/sessions/<hash>.jsonl`; `--session NAME` or `ORDO_SESSION` use `<NAME>.jsonl`
(useful for parallel agents in the same repo). `ORDO_HOME` changes the base folder.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


def ordo_home() -> Path:
    return Path(os.environ.get("ORDO_HOME") or Path.home() / ".ordo")


def repo_root(cwd: Path | None = None) -> Path:
    cwd = cwd or Path.cwd()
    try:
        out = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=cwd, capture_output=True,
                             encoding="utf-8", errors="replace", timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return Path(out.stdout.strip()).resolve()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return cwd.resolve()


@dataclass
class Session:
    path: Path
    root: Path

    @classmethod
    def open(cls, name: str | None = None, cwd: Path | None = None) -> "Session":
        root = repo_root(cwd)
        name = name or os.environ.get("ORDO_SESSION")
        if name:
            key = re.sub(r"[^\w.-]+", "_", name)
        else:
            key = f"{root.name}-{hashlib.sha1(str(root).lower().encode()).hexdigest()[:10]}"
        path = ordo_home() / "sessions" / f"{key}.jsonl"
        return cls(path=path, root=root)

    def append(self, kind: str, **data: Any) -> dict:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        ev = {"ts": round(time.time(), 3), "kind": kind, **data}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        return ev

    def events(self, kind: str | None = None) -> Iterator[dict]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if kind is None or ev.get("kind") == kind:
                    yield ev

    def since_last_compact(self) -> list[dict]:
        evs = list(self.events())
        last = max((i for i, e in enumerate(evs) if e["kind"] == "compact"), default=-1)
        return evs[last + 1:] if last >= 0 else evs

    def last_compact(self) -> dict | None:
        found = None
        for e in self.events("compact"):
            found = e
        return found
