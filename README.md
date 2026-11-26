# ordo

<p align="center">
 <img width="812" height="469" alt="image" src="https://github.com/user-attachments/assets/7bf372cf-6e6e-45b9-9eae-c62d58f6301e" />

</p>

Command-line harness for **[Ordo 3B](https://huggingface.co/devlucascfarias/ordo-3b)**, a small local
model that filters noisy tool output (test runs, builds, linters, large files) into compact causal
YAML for coding agents, so the agent reads the failure cause instead of 10,000 lines of logs.
The CLI and the model are developed by [BaruchLabs](https://baruchlabs.com/).

```
$ ordo run -i "Check why the multi-currency settlement test fails" pytest -v
# ordo run | exit=1 | ~933 -> 249 tokens, 11.1s
status: FAIL
summary: 'KeyError: ''BRL'' in src/ledger/fx.py:5 (fx_rate) when evaluating settle(batch) in src/ledger/settlement.py:7.'
causal_trace:
- file: src/ledger/fx.py
  lines: '5'
  symbol: fx_rate
  issue: 'KeyError: ''BRL'''
...
# ordo: unverified: snippet not verbatim: src/ledger/fx.py:5 (...)
```

## Use cases

**Where Ordo helps**

- **Debugging with long test output**: large suites (pytest, jest, go test, cargo test) where one failure is
  buried among hundreds of passing tests, warnings and verbose output. The agent gets the failing file, line and
  assertion in a few hundred tokens.
- **Builds and linters with many messages**: tsc, eslint, mypy, maven or dotnet output where the real error sits
  between warnings.
- **CI logs**: pass a failed CI log to `ordo filter` and get the cause instead of scrolling.
- **Large source files**: `ordo slice` returns the symbol you asked about with exact line ranges, instead of the
  whole file.
- **Long sessions**: notes and dead ends persist per project, so an agent resuming work does not retry what
  already failed.
- **Small context windows and local agents**: agents on local models or tight budgets, where every thousand tokens
  matters.
- **Private code**: everything runs on your machine; logs and code never leave it.

**Where it does not**

- **Short output** (under ~40 lines) is passed through unchanged; there is nothing to save.
- **Writing code from scratch**: tasks that run few commands give Ordo nothing to filter.
- **Agents that already trim command output, when the cause is at the start or end of the log.** Some agents keep
  only the head and tail of long output (Antigravity keeps about 5 KB of each). If the failure is there, the agent
  already sees it: Ordo still trims some tokens but costs time. It matters most when the cause lands in the part
  the agent drops, as in monorepo runs where the failing package sits between passing ones.
- **Slow hardware**: on a modest GPU each condensed call takes several seconds; on CPU only, more.

**Measured** (Gemini 4 Neon through Antigravity, fixing one bug; 2 runs per condition, AMD RX 5700 XT):

| Log | Neon alone | Neon + Ordo |
|---|---|---|
| 120-test pytest log, failure at the end | 31k tokens, 45 s | 22k tokens, 67-71 s |
| 350-test node:test log, failure at the end | 31k tokens, 32 s | 18k tokens, 51-60 s |
| 8-package Python monorepo, failure in the middle | 38k tokens, 75-97 s | **21k tokens**, 83-92 s |
| 8-package Node monorepo, failure in the middle | 38k tokens, 41-43 s | 18k / 50k tokens, 55-107 s |

Every run fixed the bug without editing the tests. Ordo cut the agent's tokens by 30-50% in most runs but added
10-40 s of wall time on this GPU (one condensed call of 6-16 s, plus the agent waiting on it). When the agent
backgrounds a slow command and polls it, the polling can eat the savings (the 50k run above).

## Install

```bash
pip install "ordo-harness[mcp]"
ordo setup
```

`pip` installs the `ordo` command (the `[mcp]` extra adds the MCP server; drop it if you only use the CLI). `ordo setup` checks the model server, downloads the Ordo model through Ollama
if it is missing (about 1.9 GB, after asking; `--yes` skips the question) and runs one test call:

```text
[ok] server: Ollama at http://127.0.0.1:11434 (3 model(s) installed)
[ok] model: hf.co/devlucascfarias/ordo-3b:latest
[ok] test call: 1.3s, valid YAML, status FAIL, cites src/fx.py
```

Requires Python 3.10+ and a running [Ollama](https://ollama.com), or any OpenAI-compatible server
(see [Other model servers](#other-model-servers)). Without `ORDO_MODEL`, the CLI uses the installed
`hf.co/devlucascfarias/ordo-3b` (any tag), then a local `ordo:3b` build. Manual install:
`ollama pull hf.co/devlucascfarias/ordo-3b` (Q4_K_M; add `:Q8_0` for the larger variant). Run the same
command again to update.

## Connect your agent

Give your coding agent the Ordo tools over MCP with one line:

```bash
claude mcp add ordo -- ordo mcp        # Claude Code
codex mcp add ordo -- ordo mcp         # Codex CLI
```

[![Add to Cursor](https://cursor.com/deeplink/mcp-install-dark.svg)](https://cursor.com/en/install-mcp?name=ordo&config=eyJjb21tYW5kIjoib3JkbyIsImFyZ3MiOlsibWNwIl19)

Antigravity (`.agents/mcp_config.json` in the workspace, or `~/.gemini/config/mcp_config.json` for all projects), Cursor, Windsurf and other clients take the same entry in their MCP config file:

```json
{ "mcpServers": { "ordo": { "command": "ordo", "args": ["mcp"] } } }
```

Agents without MCP can call the CLI directly: `ordo guide >> AGENTS.md` adds the instructions they need.

## Commands

| Command | What it does | Mode |
|---|---|---|
| `ordo run -i INTENT [--] CMD...` | runs the command, filters its output, **exits with the command's exit code** | `TOOL_OUTPUT` |
| `ordo filter -i INTENT [FILE]` | filters text from a file or stdin (`cmd \| ordo filter -i ...`) | any (`--mode`) |
| `ordo slice -i INTENT FILE` | finds the symbols/lines relevant to the intent, with real 1-based line numbers | `CODE_SLICE` |
| `ordo search -i INTENT PATTERN [PATHS]` | `rg` (or a built-in fallback) filtered by intent | `TOOL_OUTPUT` |
| `ordo dead-end ATTEMPT REASON` | records an approach that failed, so the agent does not retry it | |
| `ordo note TEXT` | records a confirmed fact | |
| `ordo memory [--grep RX]` | lists notes and dead ends with stable ids | |
| `ordo edit ID [TEXT] [--reason R]` | rewrites a note or dead end, keeping its id | |
| `ordo forget ID\|TEXT` | retracts a wrong or obsolete note or dead end | |
| `ordo compact -i GOAL` | writes the session's state ledger (goal, facts, dead ends, modified files) | `MEMORY_COMPACT` |
| `ordo stats [--json]` | token savings telemetry for the session | |
| `ordo savings [--top N] [--json]` | tokens kept out of the agent's context across all projects, per tool (pytest, tsc, cargo test...) and per project | |
| `ordo dashboard [--no-open]` | writes a self-contained HTML page with the tokens saved per project and opens it in the browser (no server; nothing leaves the machine) | |
| `ordo prune [--missing] [--older-than DAYS] [--yes]` | lists every project with recorded memory; removes those whose folder is gone or that are inactive (asks first) | |
| `ordo show [ID] [--grep RX] [--lines A-B]` | exact lines of a saved run/filter/search output (`raw=<id>` in the header) | |
| `ordo diff [OLD] [NEW]` | failures gone, new and still there between two runs (default: the last two runs of a command) | |
| `ordo setup [--yes] [--no-test]` | checks the model server, downloads the model if missing, runs a test call | |
| `ordo guide` | instructions for coding agents, to paste into `AGENTS.md` | |
| `ordo mcp [--no-run]` | serve the tools over MCP for any MCP-capable agent | |

Use `--` before the command when it has its own flags that could clash (`ordo run -i "..." -- pytest -v -k foo`).

### How an output is read

Failures are read **deterministically first** when that is safe; the local model answers the rest.

1. **Short output** (≤40 lines) is passed through unchanged (route `bypass`).
2. **Passing suites** are summarized from the runner's own last line (route `rule`, no model).
3. **Recognized and pinned-down failures** (route `index`, no model, milliseconds): compiler and linter lines
   (tsc, rustc, go, javac and Maven, dotnet, gcc, mypy, ruff, ESLint), GLSL shader errors, browser console
   errors forwarded by Playwright, exceptions with their stack, Python tracebacks, pytest, node:test, jest,
   vitest, go test, cargo test and rspec failures. Each failure is listed once, causes first and failing tests
   after, at the project's own code (the innermost own frame, with the test that called it in `called_from`;
   package-relative paths resolved in the tree), and linked to the source when the repository can tell:
   - a cascade of tsc errors about one member becomes a single cause at the type's definition, with the field it
     may have been renamed to;
   - an undeclared GLSL identifier shows where it is declared and used, under which `#ifdef`, and where that
     feature is switched on or off (`switch for USE_FRESNEL at src/config/materials.ts:8: fresnel: false`);
   - NaN errors point to the earlier log lines where NaN first shows up;
   - a missing asset (404, ENOENT) gets the closest existing file names;
   - when the question names a part ("only the stock tests"), failures elsewhere move to `other_failures`.

   The index is used only when every cause is located (or linked) and, if the question names something
   specific (an identifier such as `test_fmt` or a quoted text), one of the failures is about it.
4. **The model** (route `model`) for everything else: logs nothing above recognizes, failures it cannot pin down,
   questions about something the failures do not mention, questions about warnings, timing, skips or coverage,
   or `--deep`. The failures the model leaves out are listed after its answer.

Measured against the model alone (Ordo 3B v2, Q4_K_M), counting the file the answer must cite:

| Set | Model | Ordo 2.4 (index when confident, model otherwise) | Answered without the model |
|---|---|---|---|
| 36 held-out validation logs (teacher labels, never used to tune the index) | 83.3% | 77.8% | 83% |
| 45 failing benchmark cases (human-reviewed labels; part of them used to tune the index) | 86.7% | 95.6% | 76% |

So the gain is mostly speed: three quarters or more of the failing logs come back in milliseconds instead of 5 to
30 seconds. On logs unlike the ones the index was built on it cites the right file a little less often than the
model (5 points); use `--deep` when the answer needs the model's reading.

**Between runs.** `ordo run` compares each run with the previous run of the same command (`# ordo: since the
last run of this command: 2 failure(s) gone, 0 new, 1 still failing`). When a failure disappears although none
of the files it pointed to changed, it says so: the fix may have hidden the symptom instead of addressing the
cause. `ordo diff` lists the failures gone, new and still there.

### Behaviour that agents can rely on

- **Output is always YAML** (or raw text with a `# ordo:` header when the model is unavailable). Warnings
  are YAML comments starting with `# ordo:`.
- **Grounding check.** Every file, line, symbol and snippet the model cites is checked against the input
  (and against the file on disk for `slice`). Anything that is not literally there is flagged with
  `# ordo: unverified: ...`. Treat flagged snippets as untrusted.
- **Never blocks the agent.** If Ollama is down or the YAML is unrecoverable, the raw output is printed
  (windowed around the errors). Use `--no-fallback` to get exit code 3 instead.
- **Short output is passed through** without calling the model (≤40 lines); use `--force` to always filter.
- **Passing suites skip the model.** When a command exits 0 and the runner's own summary reports no failures
  (pytest, node:test, jest, vitest, go test, cargo test, unittest), `run` answers from that line in well under a
  second (route `rule`). Playwright too. Any failing block in the output, or a question about warnings, timing, skips or
  coverage, goes on to the index or the model.
- **Monorepo logs.** Lines like `fail 0` do not count as errors, the window keeps the package header
  (`rootdir:`, `> pkg:test`) above each failure, and a package-relative path the model cites gets a note with
  the full path when exactly one file in the tree matches (`# ordo: path: tests/x.test.js is packages/x/...`).
- **Nothing is lost.** Every `run`/`filter`/`search` saves the full output and prints `raw=<id>` in the header;
  `ordo show <id> --grep "KeyError"` returns the exact original lines when the summary is not enough.
- **`--json`** on `run`/`filter`/`slice`/`search`/`stats` prints an envelope with the parsed result, exit code,
  warnings, tokens and `raw_id`.
- **Large input is windowed** to `--budget` tokens (default 2500) around error lines; the cut is marked in the text.
- **`compact` is deterministic by default**: it is built only from what the session recorded: `dead-end`,
  `note` (minus what `ordo forget` retracted), the exit code of each test, build or lint command from the
  last `ORDO_LEDGER_HOURS` (never the model's summary), and the
  modified files from `git status`. A file is `PASS` or `FAIL` only if the last test run started after its
  last change; otherwise it is `UNTESTED`. `--neural` asks the model instead, falling back
  to the deterministic ledger if the output fails the schema (the published Ordo 3B is not reliable
  in this mode yet).
- Everything is kept per project, outside your repo: memory in `~/.ordo/sessions/<project>.jsonl`, saved
  outputs in `~/.ordo/runs/<project>/` (the newest 300 per project). `ordo show last` only sees the
  current project; a `raw_id` works from anywhere. `ordo savings` and `ordo prune` look across projects.
  Notes and dead ends get a stable id when recorded (`ordo memory` lists them); an agent can correct one
  with `ordo edit` or retract it with `ordo forget`. The session file is append-only: edits and
  retractions are new records, so the history stays auditable.
  Use `--session NAME` (or `ORDO_SESSION`) for parallel agents in the same repo; names are scoped to the
  repository, so the same name in two projects never mixes.

### Configuration

| Variable | Default |
|---|---|
| `ORDO_MODEL` | installed Ordo (`hf.co/devlucascfarias/ordo-3b`, then `ordo:3b`) |
| `OLLAMA_HOST` | `http://127.0.0.1:11434` (server-style values such as `0.0.0.0:11434` are accepted) |
| `ORDO_MAX_INPUT_TOKENS` | `2500` |
| `ORDO_SESSION` | one session per repository |
| `ORDO_KEEP_ALIVE` | `30m` (how long Ollama keeps the model loaded) |
| `ORDO_HOME` | `~/.ordo` |
| `ORDO_LEDGER_HOURS` | `12` (command results older than this are left out of `compact`) |
| `ORDO_BASE_URL` | unset (Ollama). Set it to use an OpenAI-compatible server instead |
| `ORDO_API_KEY` | unset (sent as a Bearer token when the server needs one) |

Exit codes: `run` returns the command's code (127 if not found, 124 on `--cmd-timeout`); `2` for usage
errors; `3` with `--no-fallback` when the model fails; `search` returns `1` when nothing matches.

## Other model servers

Ordo runs on any server with an OpenAI-compatible `/v1/chat/completions` endpoint: llama.cpp
(`llama-server`), LM Studio, vLLM, LocalAI. Point `ORDO_BASE_URL` (or `--base-url`) at it:

```bash
llama-server -m ordo-3b-Q4_K_M.gguf -c 16384 --port 8080      # context of 16k tokens, as Ordo expects
export ORDO_BASE_URL=http://127.0.0.1:8080              # /v1 is added when missing
ordo run -i "why does the build fail?" -- npm run build
```

The model is picked from the server's `/v1/models` list: the configured `ORDO_MODEL` if served, else a
model whose name contains `ordo`, else the only model served. Set the context size on the server; Ollama's
own `/v1` endpoint ignores it, so with Ollama keep the default native backend.

## Use as an MCP server

`ordo mcp` serves the same tools over MCP (stdio), so MCP-capable coding agents call them natively
(`ordo_run`, `ordo_filter`, `ordo_slice`, `ordo_search`, `ordo_show`, `ordo_dead_end`, `ordo_note`,
`ordo_memory`, `ordo_edit`, `ordo_forget`, `ordo_compact`, `ordo_stats`) and get structured results. Setup is
in [Connect your agent](#connect-your-agent); add `--no-run` (`ordo mcp --no-run`) to expose read-only tools.

The server keeps the model loaded between calls (`ORDO_KEEP_ALIVE`, default 30m) and runs one tool call at a time.

### Python API

```python
from ordo.client import call_ollama_ordo
from ordo.schema import parse_output, repair_yaml_output
from ordo.verify import verify

res = call_ollama_ordo("TOOL_OUTPUT", "Why does the build fail?", log_text)
parsed = parse_output(res.text, "TOOL_OUTPUT")      # YAML (+ repair) and schema validation
report = verify(parsed.obj, log_text, "TOOL_OUTPUT") # grounding check
```

## Known limitations of the published model (Ordo 3B v2)

- **Line numbers** are the weakest field: about two thirds of required lines are cited exactly. File and
  status are right in over 80% and 95% of cases. The grounding check flags any citation not in the input.
- **Go packages in subdirectories** are often cited by file name only (`lru.go` instead of `lru/lru.go`).
- `MEMORY_COMPACT` follows its schema in the Q8_0 variant but less reliably in Q4_K_M, which is one reason
  `compact` is deterministic by default.

Measured results and the comparison with v1 are on the
[model card](https://huggingface.co/devlucascfarias/ordo-3b).

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## License

MIT, see [LICENSE](https://github.com/devlucascfarias/ordo/blob/main/LICENSE). Developed by [BaruchLabs](https://baruchlabs.com/).
