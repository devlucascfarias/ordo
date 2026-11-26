"""`ordo dashboard`: a single self-contained HTML page with the token savings across projects.

No server and no external assets: the data is embedded at generation time, styles and icons are inline, and
the page follows the system light/dark theme. The typeface is Fira Code when installed (no download; falls back
to Cascadia Code, JetBrains Mono or the system monospace). Projects are shown by folder name only.
"""
from __future__ import annotations

import html
import time
from urllib.parse import quote

from . import __version__
from ._assets import BARUCHLABS_PNG


def _n(x: int) -> str:
    return f"{x:,}"


def _icon(paths: str, size: int = 15) -> str:
    return (f'<svg class="ic" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths}</svg>')


# 24px line icons, stroke only
FOLDER = _icon('<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>')
HOMEDIR = _icon('<path d="M4 11l8-7 8 7"/><path d="M6 10v9h12v-9"/>')
CLOCK = _icon('<circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/>')
TERMINAL = _icon('<rect x="3" y="4.5" width="18" height="15" rx="2"/><path d="M7 10l3 2.5L7 15"/><path d="M12.5 15H17"/>')
CHIP = _icon('<rect x="6.5" y="6.5" width="11" height="11" rx="1.5"/><path d="M9.5 3v3.5M14.5 3v3.5M9.5 17.5V21'
             'M14.5 17.5V21M3 9.5h3.5M3 14.5h3.5M17.5 9.5H21M17.5 14.5H21"/>')
PASS = _icon('<path d="M4 12h12"/><path d="M12 7l5 5-5 5"/><path d="M20 5v14"/>')
ALERT = _icon('<path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17.2v.1"/>')
LOCK = _icon('<rect x="5" y="10.5" width="14" height="9.5" rx="2"/><path d="M8.5 10.5V8a3.5 3.5 0 0 1 7 0v2.5"/>')
NOTE = _icon('<circle cx="12" cy="12" r="8.5"/><path d="M8.5 12.2l2.4 2.4 4.6-4.8"/>')
DEAD = _icon('<circle cx="12" cy="12" r="8.5"/><path d="M6 6l12 12"/>')
BOOK = _icon('<path d="M5 4.5h10a3 3 0 0 1 3 3V20H8a3 3 0 0 1-3-3z"/><path d="M5 17a3 3 0 0 1 3-3h10"/>')
MODEL = _icon('<path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"/><path d="M4 7.5l8 4.5 8-4.5"/><path d="M12 12v9"/>')
GITHUB = ('<svg class="ic" width="15" height="15" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">'
          '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37'
          '-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33'
          '.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21'
          ' 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27'
          '.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0'
          ' 0 0 16 8c0-4.42-3.58-8-8-8z"/></svg>')
GITHUB_URL = "https://github.com/devlucascfarias/ordo"
HF_URL = "https://huggingface.co/devlucascfarias/ordo-3b"
BARUCHLABS_URL = "https://baruchlabs.com/"


def _level(pct: float) -> str:
    """Colour class for a reduction: the higher, the greener. No warning colours: a low figure is not an error."""
    return "r-hi" if pct >= 90 else "r-good" if pct >= 70 else "r-mid" if pct >= 40 else "r-low"
MARK = ('<svg class="mark" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M3 5h18"/><path d="M6 10h12"/>'
        '<path d="M9 15h6"/><path d="M11.5 20h1"/></svg>')  # a sieve: wide input, narrow output

# the header mark as the tab icon: inline data URI (no file, no request), ink colour following the system theme
_FAVICON_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke-width="2.2" '
                'stroke-linecap="round"><style>path{stroke:#18202b}@media (prefers-color-scheme:dark){path{'
                'stroke:#e4e9f0}}</style><path d="M3 5h18"/><path d="M6 10h12"/><path d="M9 15h6"/>'
                '<path d="M11.5 20h1"/></svg>')
FAVICON = "data:image/svg+xml," + quote(_FAVICON_SVG, safe=" :;=,/")

CSS = """
:root {
  color-scheme: light dark;
  --paper: #eef1f5; --ink: #18202b; --graphite: #5e6a78; --rule: #d5dbe3; --raw: #c3ccd7; --signal: #2e46c8; --warn: #a1500f; --tint: #dfe5fb;
  --r-hi: #177a45; --r-good: #0e7490; --r-mid: #2e46c8; --r-low: #5e6a78;
  --mono: "Fira Code", "Cascadia Code", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --paper: #10141a; --ink: #e4e9f0; --graphite: #8a95a4; --rule: #232a34; --raw: #2c3541; --signal: #9aaeff; --warn: #f0b26b; --tint: #1d2540;
          --r-hi: #5fd394; --r-good: #56c7dd; --r-mid: #9aaeff; --r-low: #8a95a4; }
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--paper); color: var(--ink); }
body {
  font: 400 15px/1.6 var(--mono); font-feature-settings: "calt" 1, "liga" 1; font-variant-numeric: tabular-nums;
  -webkit-font-smoothing: antialiased; display: flex; justify-content: center; min-height: 100vh;
  padding: 10vh 28px 8vh;
}
main { width: 100%; max-width: 680px; }
.ic { flex: none; vertical-align: -2px; }
header { display: flex; align-items: center; gap: 10px; color: var(--graphite); margin-bottom: 9vh; }
header .word { font-size: 16px; font-weight: 500; color: var(--ink); letter-spacing: .02em; }
header .ver { font-size: 13px; }
header nav { margin-left: auto; display: flex; gap: 18px; }
a { color: inherit; text-decoration: none; }
header nav a { display: inline-flex; align-items: center; gap: 7px; font-size: 13px; color: var(--graphite);
               border-bottom: 1px solid transparent; }
header nav a:hover { color: var(--ink); border-bottom-color: var(--rule); }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--signal); outline-offset: 3px; border-radius: 3px; }

.hero h1 { color: var(--signal); margin: 0; font-weight: 300; font-size: clamp(44px, 9vw, 76px); line-height: 1; letter-spacing: -.04em; }
.hero p { margin: 14px 0 0; color: var(--graphite); max-width: 52ch; }
.flow { margin: 36px 0 0; display: grid; gap: 10px; }
.flow .row { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 16px; font-size: 13px;
             color: var(--graphite); }
.flow .row span { min-width: 15ch; }
.flow .row b { color: var(--ink); font-weight: 500; }
.flow .row.sent b { color: var(--signal); }
.track { height: 10px; border-radius: 5px; }
.fill { height: 100%; border-radius: 5px; transform-origin: left; animation: grow .9s cubic-bezier(.2,.7,.2,1) both; }
.fill.raw { background: repeating-linear-gradient(135deg, var(--raw) 0 6px, transparent 6px 9px); }
.fill.sent { background: var(--signal); min-width: 4px; animation-delay: .25s; }

.projects { margin-top: 10vh; }
.projects h2 { margin: 0 0 6px; font-size: 15px; font-weight: 500; }
.projects .sub, .memory .sub { margin: 0 0 22px; color: var(--graphite); font-size: 13px; }
.hl { color: var(--signal); font-weight: 500; }
.project { padding: 18px 0 16px; border-top: 1px solid var(--rule); }
.project:last-child { border-bottom: 1px solid var(--rule); }
.project .top { display: flex; align-items: baseline; gap: 10px; }
.project .name { font-weight: 500; display: flex; align-items: center; gap: 8px; min-width: 0; }
.project .name span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.project .saved { margin-left: auto; font-weight: 500; white-space: nowrap; color: var(--signal); }
.project .saved .unit { color: var(--graphite); font-weight: 400; font-size: 13px; margin-left: 8px; }
.pct-l { font-weight: 500; }
.r-hi { color: var(--r-hi); } .r-good { color: var(--r-good); } .r-mid { color: var(--r-mid); } .r-low { color: var(--r-low); }
.project .bars { margin: 12px 0 10px; display: grid; gap: 4px; }
.project .bars .track { height: 4px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 20px; color: var(--graphite); font-size: 13px; }
.meta span { display: inline-flex; align-items: center; gap: 6px; }
.meta b, footer b { color: var(--ink); font-weight: 500; }

.memory { margin-top: 9vh; }
.memory h2 { margin: 0 0 6px; font-size: 15px; font-weight: 500; }
details { border-top: 1px solid var(--rule); }
details:last-of-type { border-bottom: 1px solid var(--rule); }
summary { list-style: none; cursor: pointer; display: flex; align-items: center; gap: 10px; padding: 14px 0; }
summary::-webkit-details-marker { display: none; }
summary .name { font-weight: 500; display: flex; align-items: center; gap: 8px; }
summary .counts { margin-left: auto; display: flex; gap: 14px; font-size: 13px; color: var(--graphite); }
summary .counts .n { color: var(--signal); font-weight: 500; }
summary .counts .d { color: var(--warn); font-weight: 500; }
summary .chev { transition: transform .15s; color: var(--graphite); }
details[open] summary .chev { transform: rotate(90deg); }
.items { margin: 0 0 16px; padding: 0; list-style: none; display: grid; gap: 10px; font-size: 13px; }
.items li { display: grid; grid-template-columns: 18px 1fr; gap: 8px; }
.items li .ic { margin-top: 3px; }
.items .note .ic { color: var(--signal); }
.items .dead .ic { color: var(--warn); }
.items .id { color: var(--graphite); margin-right: 8px; }
.items .why { display: block; color: var(--graphite); }
footer { margin-top: 7vh; display: grid; gap: 10px; color: var(--graphite); font-size: 13px; }
footer .routes { display: flex; flex-wrap: wrap; gap: 6px 22px; }
footer span { display: inline-flex; align-items: center; gap: 7px; }
.credit { justify-content: center; text-align: center; margin-top: 18px; padding-top: 18px;
          border-top: 1px solid var(--rule); }
.credit a { display: inline-flex; align-items: center; gap: 7px; color: var(--ink); font-weight: 500;
            border-bottom: 1px solid var(--rule); }
.credit a:hover { border-bottom-color: var(--ink); }
.credit img { width: 22px; height: 22px; }
@media (prefers-color-scheme: dark) {  /* keep the logo's own colours; a thin light edge separates the black hair */
  .credit img { filter: drop-shadow(0 0 .6px rgba(228, 233, 240, .9)) drop-shadow(0 0 .6px rgba(228, 233, 240, .6)); }
}
.empty { color: var(--graphite); max-width: 52ch; }
.empty code { color: var(--ink); }

@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .fill { animation: none; } }
@media (max-width: 520px) {
  body { padding: 6vh 20px 6vh; }
  header { margin-bottom: 6vh; }
  header nav a span { display: none; }
}
"""


def _bar(kind: str, pct: float) -> str:
    return f'<div class="track"><div class="fill {kind}" style="width:{max(0.0, min(100.0, pct)):.2f}%"></div></div>'


CHEVRON = _icon('<path d="M9 6l6 6-6 6"/>', 14)


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} notes and dead ends"


def _memory(memory: dict) -> str:
    """Notes and dead ends per project, collapsed by default. `memory`: project -> list of item dicts."""
    projects = {k: v for k, v in memory.items() if v}
    if not projects:
        return ('<section class="memory"><h2>Memory</h2><p class="sub">No notes or dead ends recorded yet. Agents add '
                "them with <code>ordo note</code> and <code>ordo dead-end</code>.</p></section>")
    blocks = []
    for name, items in projects.items():
        notes = [m for m in items if m["kind"] == "note"]
        dead = [m for m in items if m["kind"] == "dead_end"]
        lis = "".join(
            f'<li class="note">{NOTE}<span><span class="id">{html.escape(m["id"])}</span>{html.escape(m["text"])}</span></li>'
            for m in notes) + "".join(
            f'<li class="dead">{DEAD}<span><span class="id">{html.escape(m["id"])}</span>{html.escape(m["text"])}'
            f'<span class="why">failed: {html.escape(m.get("reason", ""))}</span></span></li>' for m in dead)
        icon = HOMEDIR if name == "~" else FOLDER
        blocks.append(
            f'<details><summary>{CHEVRON}<span class="name">{icon}{html.escape(name)}</span>'
            f'<span class="counts"><span><span class="n">{len(notes)}</span> {"note" if len(notes) == 1 else "notes"}'
            f'</span><span><span class="d">{len(dead)}</span> {"dead end" if len(dead) == 1 else "dead ends"}</span>'
            f'</span></summary><ul class="items">{lis}</ul></details>')
    total = sum(len(v) for v in projects.values())
    return ('<section class="memory"><h2>Memory</h2>'
            f'<p class="sub">{_plural(total, "note or dead end", )} recorded by your agents, kept across sessions. '
            "Fix one with <code>ordo edit</code>, retract it with <code>ordo forget</code>.</p>"
            + "".join(blocks) + "</section>")


def render(rep: dict, memory: dict | None = None) -> str:
    t = rep["total"]
    projects = rep["by_project"]
    generated = time.strftime("%Y-%m-%d %H:%M")
    if not t["calls"]:
        body = ('<section class="hero"><h1>0</h1><p class="empty">No tool output recorded yet. In a project, run '
                '<code>ordo run -i "why does it fail?" -- pytest -q</code>, then open this page again with '
                "<code>ordo dashboard</code>.</p></section>")
    else:
        rows = []
        for name, r in projects.items():
            icon = HOMEDIR if name == "~" else FOLDER
            rows.append(
                '<article class="project"><div class="top">'
                f'<div class="name">{icon}<span>{html.escape(name)}</span></div>'
                f'<div class="saved">{_n(r["saved"])}<span class="unit">tokens saved '
                f'(<span class="pct-l {_level(r["saved_pct"])}">{r["saved_pct"]:.0f}%</span>)</span></div></div>'
                f'<div class="bars">{_bar("raw", 100)}{_bar("sent", 100 * r["sent"] / r["raw"] if r["raw"] else 0)}</div>'
                f'<div class="meta"><span>{TERMINAL}<b>{_n(r["calls"])}</b> calls</span>'
                f'<span>{CLOCK}last used {html.escape(r.get("last", ""))}</span></div></article>')
        sent_pct = 100 * t["sent"] / t["raw"] if t["raw"] else 0
        body = (
            '<section class="hero">'
            f'<h1>{_n(t["saved"])}</h1>'
            f"<p>tokens kept out of your agent's context across {_n(t['calls'])} tool calls since "
            f'{html.escape(rep["since"] or "")}</p>'
            '<div class="flow">'
            f'<div class="row">{_bar("raw", 100)}<span><b>{_n(t["raw"])}</b> read</span></div>'
            f'<div class="row sent">{_bar("sent", sent_pct)}<span><b>{_n(t["sent"])}</b> sent</span></div>'
            "</div></section>"
            '<section class="projects"><h2>Projects</h2>'
            f'<p class="sub">{len(projects)} with recorded tool output, most saved first; '
            f'<b class="{_level(t["saved_pct"])}">{t["saved_pct"]:.0f}% less context</b> overall</p>' + "".join(rows) + "</section>"
            + _memory(memory or {}) +
            '<footer><div class="routes">'
            f'<span>{CHIP}<b>{_n(t["filtered"])}</b> condensed by the model</span>'
            + (f'<span>{NOTE}<b>{_n(t["rule"])}</b> passing suite (rule)</span>' if t.get("rule") else "")
            + (f'<span>{NOTE}<b>{_n(t["index"])}</b> failures indexed without the model</span>' if t.get("index") else "")
            + f'<span>{PASS}<b>{_n(t.get("bypass", 0))}</b> short enough to pass through</span>'
            f'<span>{ALERT}<b>{_n(t.get("fallback", 0))}</b> fallback</span></div>'
            f"<span>{LOCK}Generated {generated} from local session files. Nothing on this page leaves your machine. "
            "Token counts are estimates.</span>"
            f'<span class="credit">Ordo CLI and model developed by <a href="{BARUCHLABS_URL}" target="_blank" '
            f'rel="noopener noreferrer"><img alt="" width="22" height="22" '
            f'src="data:image/png;base64,{BARUCHLABS_PNG}">BaruchLabs</a></span></footer>')
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f'<title>Ordo</title><link rel="icon" type="image/svg+xml" href="{FAVICON}">'
        f"<style>{CSS}</style></head><body><main>"
        f'<header>{MARK}<span class="word">ordo</span><span class="ver">v{html.escape(__version__)}</span>'
        f'<nav><a href="{GITHUB_URL}" rel="noopener" aria-label="GitHub">{GITHUB}<span>GitHub</span></a>'
        f'<a href="{HF_URL}" rel="noopener" aria-label="Hugging Face">{MODEL}<span>Hugging Face</span></a></nav>'
        "</header>"
        f"{body}</main></body></html>\n")
