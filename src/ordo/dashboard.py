"""`ordo dashboard`: a single self-contained HTML page with the token savings across projects.

No server and no external assets: the data is embedded at generation time, styles and icons are inline, and
the page follows the system light/dark theme. The typeface is Fira Code when installed (no download; falls back
to Cascadia Code, JetBrains Mono or the system monospace). Projects are shown by folder name only.
"""
from __future__ import annotations

import html
import time

from . import __version__


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
MARK = ('<svg class="mark" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" aria-hidden="true"><path d="M3 5h18"/><path d="M6 10h12"/>'
        '<path d="M9 15h6"/><path d="M11.5 20h1"/></svg>')  # a sieve: wide input, narrow output

CSS = """
:root {
  color-scheme: light dark;
  --paper: #eef1f5; --ink: #18202b; --graphite: #5e6a78; --rule: #d5dbe3; --raw: #c3ccd7; --signal: #2e46c8;
  --mono: "Fira Code", "Cascadia Code", "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root { --paper: #10141a; --ink: #e4e9f0; --graphite: #8a95a4; --rule: #232a34; --raw: #2c3541; --signal: #9aaeff; }
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
header .ver { margin-left: auto; font-size: 13px; }

.hero h1 { margin: 0; font-weight: 300; font-size: clamp(44px, 9vw, 76px); line-height: 1; letter-spacing: -.04em; }
.hero p { margin: 14px 0 0; color: var(--graphite); max-width: 52ch; }
.flow { margin: 36px 0 0; display: grid; gap: 10px; }
.flow .row { display: grid; grid-template-columns: 1fr auto; align-items: center; gap: 16px; font-size: 13px;
             color: var(--graphite); }
.flow .row span { min-width: 15ch; }
.flow .row b { color: var(--ink); font-weight: 500; }
.track { height: 10px; border-radius: 5px; }
.fill { height: 100%; border-radius: 5px; transform-origin: left; animation: grow .9s cubic-bezier(.2,.7,.2,1) both; }
.fill.raw { background: repeating-linear-gradient(135deg, var(--raw) 0 6px, transparent 6px 9px); }
.fill.sent { background: var(--signal); min-width: 4px; animation-delay: .25s; }

.projects { margin-top: 10vh; }
.projects h2 { margin: 0 0 6px; font-size: 15px; font-weight: 500; }
.projects .sub { margin: 0 0 22px; color: var(--graphite); font-size: 13px; }
.project { padding: 18px 0 16px; border-top: 1px solid var(--rule); }
.project:last-child { border-bottom: 1px solid var(--rule); }
.project .top { display: flex; align-items: baseline; gap: 10px; }
.project .name { font-weight: 500; display: flex; align-items: center; gap: 8px; min-width: 0; }
.project .name span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.project .saved { margin-left: auto; font-weight: 500; white-space: nowrap; }
.project .saved small { color: var(--graphite); font-weight: 400; font-size: 13px; margin-left: 6px; }
.project .bars { margin: 12px 0 10px; display: grid; gap: 4px; }
.project .bars .track { height: 4px; }
.meta { display: flex; flex-wrap: wrap; gap: 6px 20px; color: var(--graphite); font-size: 13px; }
.meta span { display: inline-flex; align-items: center; gap: 6px; }

footer { margin-top: 7vh; display: grid; gap: 10px; color: var(--graphite); font-size: 13px; }
footer .routes { display: flex; flex-wrap: wrap; gap: 6px 22px; }
footer span { display: inline-flex; align-items: center; gap: 7px; }
.empty { color: var(--graphite); max-width: 52ch; }
.empty code { color: var(--ink); }

@keyframes grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@media (prefers-reduced-motion: reduce) { .fill { animation: none; } }
@media (max-width: 520px) {
  body { padding: 6vh 20px 6vh; }
  header { margin-bottom: 6vh; }
  .project .saved small { display: none; }
}
"""


def _bar(kind: str, pct: float) -> str:
    return f'<div class="track"><div class="fill {kind}" style="width:{max(0.0, min(100.0, pct)):.2f}%"></div></div>'


def render(rep: dict) -> str:
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
                f'<div class="saved">{_n(r["saved"])}<small>{r["saved_pct"]:.0f}%</small></div></div>'
                f'<div class="bars">{_bar("raw", 100)}{_bar("sent", 100 * r["sent"] / r["raw"] if r["raw"] else 0)}</div>'
                f'<div class="meta"><span>{TERMINAL}{_n(r["calls"])} calls</span>'
                f'<span>{CLOCK}last used {html.escape(r.get("last", ""))}</span></div></article>')
        sent_pct = 100 * t["sent"] / t["raw"] if t["raw"] else 0
        body = (
            '<section class="hero">'
            f'<h1>{_n(t["saved"])}</h1>'
            f"<p>tokens kept out of your agent's context across {_n(t['calls'])} tool calls since "
            f'{html.escape(rep["since"] or "")}</p>'
            '<div class="flow">'
            f'<div class="row">{_bar("raw", 100)}<span><b>{_n(t["raw"])}</b> read</span></div>'
            f'<div class="row">{_bar("sent", sent_pct)}<span><b>{_n(t["sent"])}</b> sent</span></div>'
            "</div></section>"
            '<section class="projects"><h2>Projects</h2>'
            f'<p class="sub">{len(projects)} with recorded tool output, most saved first; '
            f'{t["saved_pct"]:.0f}% less context overall</p>' + "".join(rows) + "</section>"
            '<footer><div class="routes">'
            f'<span>{CHIP}{_n(t["filtered"])} condensed by the model</span>'
            f'<span>{PASS}{_n(t.get("bypass", 0))} short enough to pass through</span>'
            f'<span>{ALERT}{_n(t.get("fallback", 0))} fallback</span></div>'
            f"<span>{LOCK}Generated {generated} from local session files. Nothing on this page leaves your machine. "
            "Token counts are estimates.</span></footer>")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Ordo</title><style>{CSS}</style></head><body><main>"
        f'<header>{MARK}<span class="word">ordo</span><span class="ver">v{html.escape(__version__)}</span></header>'
        f"{body}</main></body></html>\n")
