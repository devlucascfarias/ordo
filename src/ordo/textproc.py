"""Input preparation before the model: cleanup, token estimate and windowing.

The model was trained on short inputs; this makes sure nothing silently exceeds the budget
(Ollama would cut the start of the prompt) and that whatever is cut is marked in the text.
"""
from __future__ import annotations

import re

from .contract import number_lines

ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07]*(\x07|\x1b\\)|\x1b[@-Z\\-_]")
ERROR_HINT = re.compile(
    r"error|fail|exception|traceback|panic|fatal|assert|expected|undefined|cannot|refused|denied|"
    r"not found|missing|invalid|warning|E\d{3,4}\b|TS\d{4}|\bat .+:\d+|-->|^\s*E\s",
    re.IGNORECASE,
)


def clean(text: str) -> str:
    """Strip ANSI codes, normalize line breaks and \\r redraws (progress bars)."""
    text = ANSI.sub("", text).replace("\r\n", "\n")
    lines = [l.rsplit("\r", 1)[-1] for l in text.split("\n")]
    return "\n".join(lines).strip("\n")


def est_tokens(text: str) -> int:
    """Cheap estimate (code and logs are about 3.5 characters per token with the Qwen tokenizer)."""
    return int(len(text) / 3.5) + 1


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1] + 1:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _render(lines: list[str], spans: list[tuple[int, int]], numbered: bool) -> str:
    parts, prev = [], 0
    for a, b in spans:
        if a > prev:
            parts.append(f"[... ordo: {a - prev} lines omitted ...]")
        chunk = lines[a: b + 1]
        parts.append(number_lines("\n".join(chunk), start=a + 1) if numbered else "\n".join(chunk))
        prev = b + 1
    if prev < len(lines):
        parts.append(f"[... ordo: {len(lines) - prev} lines omitted ...]")
    return "\n".join(parts)


def _fit(lines: list[str], spans: list[tuple[int, int]], budget: int) -> list[tuple[int, int]]:
    """Keep spans in priority order until the token budget runs out."""
    covered: set[int] = set()
    used = 0
    for a, b in spans:
        new = [i for i in range(a, b + 1) if i not in covered]
        cost = sum(est_tokens(lines[i]) for i in new)
        if used + cost > budget:
            for i in new:  # cut the span from its end
                c = est_tokens(lines[i])
                if used + c > budget:
                    break
                covered.add(i)
                used += c
            break
        covered.update(new)
        used += cost
    return _merge([(i, i) for i in sorted(covered)])


def shorten_line(line: str, max_chars: int) -> str:
    """A huge line (minified bundle, JSON dump, snapshot) becomes start + marker + end."""
    if len(line) <= max_chars:
        return line
    keep = max_chars // 2
    return f"{line[:keep]} [... ordo: {len(line) - 2 * keep} chars omitted ...] {line[-keep:]}"


def window_log(text: str, budget: int, *, head: int = 30, tail: int = 60, ctx: int = 12) -> tuple[str, bool]:
    """Reduce a log to the budget, keeping the end, the start and blocks around error lines."""
    if est_tokens(text) <= budget:
        return text, False
    max_chars = max(int(budget * 3.5 / 8), 200)  # no single line takes more than 1/8 of the budget
    lines = [shorten_line(l, max_chars) for l in text.split("\n")]
    n = len(lines)
    errs = [i for i, l in enumerate(lines) if ERROR_HINT.search(l)]
    around = lambda i: (max(i - ctx, 0), min(i + ctx, n - 1))  # noqa: E731
    # priority: first error, last error, short tail (summary), short head, remaining errors from last to first
    spans = [around(i) for i in (errs[:1] + errs[-1:])]
    spans += [(max(n - min(tail, 20), 0), n - 1), (0, min(head // 3, n) - 1)]
    spans += [around(i) for i in errs[1:3] + errs[::-1]]
    spans += [(max(n - tail, 0), n - 1), (0, min(head, n) - 1)]
    return _render(lines, _fit(lines, spans, budget), numbered=False), True


_STOP = set("""the and for with from that this into where what which when find show check list all any
uses used using call calls caller callers function functions method class file files code line lines
defined definition implementation where's does how why fix bug error test tests verify""".split())


def intent_terms(intent: str) -> tuple[list[str], list[str]]:
    """Strong terms (code identifiers, quoted or backticked text) and weak terms (common words)."""
    quoted = re.findall(r"[`'\"]([^`'\"]{2,60})[`'\"]", intent)
    idents = re.findall(r"[A-Za-z_][A-Za-z0-9_]*(?:[.:#]{1,2}[A-Za-z_][A-Za-z0-9_]*)*", intent)
    strong = quoted + [w for w in idents if "_" in w or re.search(r"[a-z][A-Z]", w) or re.search(r"[.:#]", w)]
    # `Context.lookup_default` does not appear like that in code (it is `def lookup_default`): search the parts too
    for w in list(strong):
        parts = [p for p in re.split(r"[.:#]+", w) if len(p) > 2]
        if len(parts) > 1:
            strong += parts
    weak = [w for w in idents if len(w) >= 4 and w.lower() not in _STOP and w not in strong]
    seen: set[str] = set()
    return [s for s in strong if not (s in seen or seen.add(s))], weak


def window_file(text: str, intent: str, budget: int, *, ctx: int = 40) -> tuple[str, bool]:
    """Numbered file (1-based); if it does not fit, only the regions that match the intent."""
    full = number_lines(text)
    if est_tokens(full) <= budget:
        return full, False
    max_chars = max(int(budget * 3.5 / 8), 200)  # minified lines must not use up the window
    lines = [shorten_line(l, max_chars) for l in text.split("\n")]
    strong, weak = intent_terms(intent)
    n = len(lines)
    scores = []
    for i, l in enumerate(lines):
        s = sum(3 for t in strong if t in l) + sum(1 for t in weak if t.lower() in l.lower())
        if s and re.match(r"\s*(def|class|func|fn|function|public|private|protected|export|async|static|module|impl|struct|interface|type)\b", l):
            s += 2  # definitions first
        if s:
            scores.append((s, i))
    scores.sort(key=lambda x: (-x[0], x[1]))
    spans = [(max(i - 5, 0), min(i + ctx, n - 1)) for _, i in scores]
    if not spans:
        spans = [(0, n - 1)]
    return _render(lines, _fit(lines, spans, budget), numbered=True), True
