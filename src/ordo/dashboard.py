"""`ordo dashboard`: a single self-contained HTML page with the token savings across projects.

No server and no external assets: the data is embedded at generation time, styles are inline, and the page
follows the system light/dark theme. Projects are shown by folder name only, never by full path.
"""
from __future__ import annotations

import html
import time

from . import __version__


def _n(x: int) -> str:
    return f"{x:,}"


def _short(x: int) -> str:
    """1234 -> 1.2k; 1234567 -> 1.2M (for the headline number)."""
    for div, suf in ((1_000_000, "M"), (1_000, "k")):
        if abs(x) >= div:
            v = x / div
            return f"{v:.1f}{suf}" if v < 100 else f"{v:.0f}{suf}"
    return str(x)


CSS = """
:root { color-scheme: light dark; --bg: #fafaf9; --fg: #1c1917; --muted: #78716c; --line: #e7e5e4;
        --bar: #d6d3d1; --accent: #15803d; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #0c0a09; --fg: #f5f5f4; --muted: #a8a29e; --line: #292524; --bar: #44403c; --accent: #4ade80; }
}
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--fg); }
body { font: 15px/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
       min-height: 100vh; display: flex; justify-content: center; padding: 12vh 24px 8vh; }
main { width: 100%; max-width: 640px; }
.brand { font-size: 13px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin: 0 0 40px; }
.hero { text-align: center; margin-bottom: 56px; }
.hero .big { font-size: 72px; font-weight: 600; letter-spacing: -.03em; line-height: 1;
             font-variant-numeric: tabular-nums; }
.hero .label { margin-top: 12px; color: var(--muted); }
.hero .label b { color: var(--accent); font-weight: 600; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th { text-align: left; font-weight: 500; font-size: 12px; letter-spacing: .06em; text-transform: uppercase;
     color: var(--muted); padding: 0 0 10px; border-bottom: 1px solid var(--line); }
td { padding: 14px 0 6px; border-bottom: 1px solid var(--line); vertical-align: top; }
th.num, td.num { text-align: right; padding-left: 16px; white-space: nowrap; }
td .name { font-weight: 500; }
td .when { color: var(--muted); font-size: 13px; }
.share { height: 3px; background: var(--line); border-radius: 2px; margin-top: 8px; overflow: hidden; }
.share span { display: block; height: 100%; background: var(--accent); opacity: .75; }
td.saved { color: var(--accent); font-weight: 500; }
.foot { margin-top: 32px; color: var(--muted); font-size: 13px; text-align: center; }
.empty { text-align: center; color: var(--muted); }
@media (max-width: 520px) {
  body { padding-top: 7vh; }
  .hero .big { font-size: 56px; }
  .pct { display: none; }
  th.num, td.num { padding-left: 10px; }
}
code { font: 13px ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
"""


def render(rep: dict) -> str:
    t = rep["total"]
    projects = rep["by_project"]
    generated = time.strftime("%Y-%m-%d %H:%M")
    if not t["calls"]:
        body = ('<p class="empty">No tool output recorded yet.<br>Run <code>ordo run -i "..." -- &lt;command&gt;</code> '
                "in a project, then open this page again.</p>")
    else:
        top = max((r["saved"] for r in projects.values()), default=0) or 1
        rows = []
        for name, r in projects.items():
            share = max(0.0, min(100.0, 100 * r["saved"] / top))
            rows.append(
                "<tr>"
                f'<td><div class="name">{html.escape(name)}</div>'
                f'<div class="when">last used {html.escape(r.get("last", ""))}</div>'
                f'<div class="share"><span style="width:{share:.1f}%"></span></div></td>'
                f'<td class="num">{_n(r["calls"])}</td>'
                f'<td class="num saved">{_n(r["saved"])}</td>'
                f'<td class="num pct">{r["saved_pct"]:.0f}%</td>'
                "</tr>")
        passed = t.get("bypass", 0)
        fallback = t.get("fallback", 0)
        body = (
            '<section class="hero">'
            f'<div class="big">{_short(t["saved"])}</div>'
            f'<div class="label">tokens kept out of your agent\'s context, <b>{t["saved_pct"]:.0f}%</b> of '
            f'{_n(t["raw"])}, across {_n(t["calls"])} tool calls since {html.escape(rep["since"] or "")}</div>'
            "</section>"
            '<table><thead><tr><th>Project</th><th class="num">Calls</th><th class="num">Tokens saved</th>'
            '<th class="num pct">Reduction</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
            f'<p class="foot">{_n(t["filtered"])} condensed by the model, {_n(passed)} short enough to pass through, '
            f"{_n(fallback)} fallback. Token counts are estimates.</p>")
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>Ordo</title><style>{CSS}</style></head><body><main>"
        '<p class="brand">Ordo</p>'
        f"{body}"
        f'<p class="foot">Generated {generated} by ordo {html.escape(__version__)}. Nothing on this page leaves '
        "your machine.</p>"
        "</main></body></html>\n")
