# texflow — Build Plan

> Talk into your LaTeX project. It reads your setup (packages, citation style, macros, labels, bib, writing style), listens to free speech, and writes clean, academic LaTeX into your file that compiles in **your** template.
>
> Working name: `texflow` (check PyPI availability before publishing; fallbacks: `speaktex`, `talktex`).

## How to use this plan (instructions for Claude Code)

- Build **milestone by milestone**. After each milestone: run the tests, show me what works, and **stop for my review** before continuing.
- Keep it simple: a single Python CLI, no servers, no editor extensions, no agent frameworks.
- Before Milestone 1, ask me for the path to my real thesis project so we can test against it (in addition to the fixture project).
- Prefer deterministic code checks over prompt instructions wherever possible.

---

## 1. Goals and non-goals

**Goals**
- `texflow` in a LaTeX folder → it learns the project → listens → writes LaTeX at a marker in the file.
- Free speech, **no fixed commands**. The LLM interprets intent: write, correct, rewrite, delete, or ignore (thinking out loud).
- Output is rewritten to be clean (default: academic register matched to my existing chapters), never raw transcription.
- Fast: ~1–2 seconds from end of speech to text in the file (Groq backend).
- Reliable: every output is validated before touching the file; everything is undoable.
- Free/cheap: Groq free tier by default, local (faster-whisper + Ollama) optional.
- Speak in Greek or English (or mixed); output follows the document's language.

**Non-goals (for v0.1)**
- No VS Code extension, MCP server, Claude Code plugin, or OpenAI proxy.
- No GUI. No Overleaf integration (local files only).
- No general-purpose dictation outside LaTeX.

---

## 2. Stack

- Python 3.11+, managed with **uv**
- CLI: `typer` + `rich` (status line, colored output)
- Config: TOML (`tomllib` to read, `tomli-w` to write), stored at `~/.config/texflow/config.toml`, with optional project-level `.texflow.toml`
- LLM + STT client: the official `openai` SDK with a configurable `base_url` (Groq, Ollama, OpenRouter and OpenAI all speak this API)
- Audio: `sounddevice` + `numpy`; voice activity detection with `webrtcvad-wheels` (or a simple energy-based VAD if that's simpler)
- Bib parsing: `bibtexparser`
- Structured output: `pydantic` models for the LLM's JSON decisions
- Optional extra `[local]`: `faster-whisper`
- Tests: `pytest`, with a mocked LLM client

---

## 3. Repo layout

```
texflow/
├── pyproject.toml              # [project.scripts] texflow = "texflow.cli:main"
├── README.md
├── texflow.example.toml
├── src/texflow/
│   ├── cli.py                  # typer app: listen (default), init, doctor, undo, index, text
│   ├── config.py               # load/merge global + project config, defaults
│   ├── project/
│   │   ├── discover.py         # find main .tex (\documentclass), follow \input/\include
│   │   ├── preamble.py         # packages + options, \newcommand/\DeclareMathOperator/etc.
│   │   ├── conventions.py      # derive rules: cite cmds, ref cmds, table style, units
│   │   ├── bib.py              # parse .bib → key, authors, title, year
│   │   ├── labels.py           # all \label{}, detect prefix pattern (fig:, tab:, sec:, eq:)
│   │   ├── style.py            # pick 2–3 representative paragraphs + 1 table + 1 figure
│   │   └── index.py            # build + cache ProjectIndex in .texflow/index.json (mtime-based)
│   ├── engine/
│   │   ├── buffer.py           # rolling transcript buffer + record of recent written blocks
│   │   ├── retrieve.py         # per-request: fuzzy-match bib entries, relevant labels
│   │   ├── prompts.py          # system prompt builder (conventions, macros, style samples)
│   │   ├── decide.py           # one LLM call → Decision (pydantic); optional small tool loop
│   │   └── engine.py           # orchestrates: buffer → retrieve → decide → validate → retry
│   ├── validate/
│   │   ├── citations.py        # every \cite-like key exists in bib
│   │   ├── refs.py             # every \ref/\cref target exists (or is created in this output)
│   │   ├── numbers.py          # no invented numbers (see §5.4)
│   │   ├── syntax.py           # balanced braces/environments
│   │   └── compile.py          # optional: compile snippet inside the real preamble
│   ├── io/
│   │   ├── writer.py           # write at marker, move marker, backups
│   │   └── history.py          # undo stack (.texflow/history.jsonl)
│   ├── audio/
│   │   ├── capture.py          # mic stream + VAD → utterance chunks
│   │   └── stt.py              # Groq/OpenAI transcription, or local faster-whisper
│   └── providers.py            # build openai clients from config
└── tests/
    ├── fixtures/sample-thesis/ # tiny project: biblatex+cleveref+booktabs+siunitx, .bib, 2 chapters
    ├── test_project_*.py
    ├── test_validate_*.py
    └── test_engine.py          # mocked LLM decisions, golden outputs
```

---

## 4. CLI

| Command | What it does |
|---|---|
| `texflow` / `texflow listen [PATH]` | Main mode: index the project, listen, write at the marker. |
| `texflow text [PATH]` | Same engine, but reads typed lines from stdin instead of the mic (for testing and quiet places). |
| `texflow init` | First-run wizard: backend choice (Groq / Local / custom URL), API key (tested immediately), mic choice. Runs automatically if no config exists. |
| `texflow doctor` | Checks mic, backend reachability, model names, TeX availability, and project detection, with fixes for each failure. |
| `texflow undo [N]` | Reverts the last N writes. |
| `texflow index [PATH]` | Prints what it learned (packages, conventions, macros, bib count, labels). Great for debugging. |

Startup output should look like:
```
✓ main.tex — biblatex (authoryear), cleveref, booktabs, siunitx, 14 macros
✓ 87 bib entries · 42 labels (sec:, fig:, tab:, eq:)
✓ Style learned from chapters/introduction.tex, chapters/related-work.tex
🎙 Listening → chapters/methodology.tex at %> HERE   (style: academic, lang: auto)
```

---

## 5. Core pipeline

### 5.1 Project index (built once, cached)
- **Discover**: find the main file via `\documentclass`; follow `\input`/`\include`/`\subfile`; find `.bib` via `\addbibresource` or `\bibliography`.
- **Preamble**: list packages with options; extract macros (`\newcommand`, `\renewcommand`, `\DeclareMathOperator`, `\NewDocumentCommand`) with arity.
- **Conventions** (derived deterministically, passed to the LLM as explicit rules):
  - biblatex → `\parencite` / `\textcite` (detect `style=` option); natbib → `\citep` / `\citet`; otherwise `\cite`
  - cleveref → `\cref` / `\Cref`; else `Figure~\ref{}`
  - booktabs → `\toprule` / `\midrule` / `\bottomrule`, no vertical rules
  - siunitx → `\SI{}{}` / `\num{}`
  - label prefix pattern from existing labels
  - document language from babel/polyglossia options or the text itself
- **Style samples**: 2–3 representative paragraphs, 1 table and 1 figure from existing chapters (few-shot examples).
- **Cache** in `.texflow/index.json`, rebuilding only files whose mtime changed.

### 5.2 Listening (audio → utterances)
- Continuous mic stream at 16 kHz mono, VAD-based segmentation.
- A short pause (~0.7 s) closes an *utterance*; utterances go to STT immediately.
- STT default: Groq `whisper-large-v3-turbo` (make the model configurable and verify current model names in `doctor`). Language: auto-detect (Greek and English both supported).
- Pass a `prompt` hint to Whisper with key terms (macro names, frequent bib author surnames) to improve recognition.
- Discard empty or near-silent transcripts.

### 5.3 Deciding (transcript → action)
Each new utterance is appended to the **rolling buffer**. The engine sends one LLM call with:
- system prompt: conventions, macros, style level, style samples, rules (never invent facts, output only LaTeX in `latex`, match the document's language)
- context: the ~1500 characters of the file before the marker, the last 3 written blocks (with ids), pending (not yet written) transcript
- retrieved bib candidates (fuzzy match on author surnames, title words and years mentioned) and relevant labels

The model returns JSON validated by pydantic:
```json
{"action": "wait"}                                       // thought not finished
{"action": "ignore"}                                     // thinking out loud, not content
{"action": "insert", "latex": "..."}
{"action": "replace", "block_id": "b12", "latex": "..."}  // corrections / rewrites of recent blocks
{"action": "delete",  "block_id": "b12"}
{"action": "uncertain_insert", "latex": "...", "note": "did you mean chalkidis2019 or chalkidis2020?"}
```
- `wait`: keep the transcript pending; force a decision after ~20 s or a long pause (~2.5 s).
- `uncertain_insert`: write the best guess plus a `% texflow-check: <note>` comment. Never block the user.
- There are **no keyword commands**. "Scratch that", "no wait, it was 0.63", "make that more formal" and "όχι, σβήσ' το" are all interpreted by the model.
- Optional tiny tool loop (only if needed, max 3 steps) for edits beyond the last few blocks: `find_block(query)`, `search_bib(query)`, `find_label(query)`.

### 5.4 Validation (before anything touches the file)
1. **Citations**: every key in `\cite`-family commands exists in the bib.
2. **Refs**: every `\ref`/`\cref` target exists in the project or is defined in this same output.
3. **Labels**: new labels follow the detected prefix pattern and are unique.
4. **No invented numbers**: every numeric literal in the output's prose, table cells and `\num`/`\SI` arguments must appear in the user's transcript or in the block being replaced. Ignore digits inside command names, cite keys, label names and structural arguments.
5. **Syntax**: balanced braces, matching `\begin`/`\end`.
6. **Compile** (optional, only if `latexmk` or `tectonic` exists): wrap the snippet in the real preamble plus `\begin{document}…\end{document}` and compile in a temp dir with a timeout. Treat undefined-reference warnings as fine; treat errors as failures.

On failure: one automatic retry with the validator messages fed back. If it still fails, write it as `uncertain_insert` with the issue in the check comment.

### 5.5 Writing
- Target: the `%> HERE` marker in any project file. If none exists, ask once at startup which file to append to, then insert a marker.
- Insert above the marker so the marker stays below the latest text.
- Each write is a **block** with an id, wrapped in invisible bookkeeping only inside texflow's history (no markup left in the file).
- Before the first write of a session, copy the target file to `.texflow/backups/<timestamp>-<file>`.
- History in `.texflow/history.jsonl` stores each operation with before/after text, so `undo` and voice "undo that" work reliably.
- Writes are atomic (write to temp, then rename) so the editor and LaTeX Workshop never see half files.
- Add `.texflow/` to the project's `.gitignore` if a git repo is detected (ask first).

---

## 6. Config (`~/.config/texflow/config.toml`)

```toml
[backend]
provider = "groq"                       # groq | local | custom
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"            # or api_key stored by `init`

[models]
stt = "whisper-large-v3-turbo"
llm = "llama-3.3-70b-versatile"         # verify current names; configurable

[local]
whisper_model = "large-v3-turbo"        # faster-whisper
ollama_url = "http://localhost:11434/v1"
llm = "qwen2.5-coder:14b"

[writing]
style = "academic"                      # clean | academic
language = "auto"                       # auto | en | el

[listen]
utterance_pause = 0.7
decision_pause = 2.5
max_pending_seconds = 20
```

---

## 7. Milestones

**M0 — Scaffold (≈30 min)**
- uv project, src layout, typer CLI with stub commands, ruff, pytest, fixture project.
- ✅ `uv run texflow --help` works; `pytest` passes.

**M1 — Project index**
- discover, preamble, conventions, bib, labels, style samples, cache.
- `texflow index` prints a summary.
- ✅ Correct summary for the fixture project **and my real thesis**.

**M2 — Engine in text mode (no audio yet)**
- buffer, retrieve, prompts, decide, providers (Groq first).
- `texflow text` reads typed lines and prints the decisions + LaTeX to the terminal (no file writes yet).
- ✅ Typing "add a table comparing legal-bert 0.71 and bert-base 0.63 on macro F1, cite the chalkidis paper" produces a booktabs table with correct cite command, real key and correct label prefix.
- ✅ A follow-up "no wait, the second one was 0.61" produces a `replace` of that block.

**M3 — Validators**
- citations, refs, labels, numbers, syntax, optional compile; retry loop.
- ✅ Unit tests for each validator; an LLM output with a fake cite key or invented number is rejected and retried (mocked).

**M4 — Writer + undo**
- marker writing, atomic writes, backups, history, `texflow undo`, voice-level undo via `replace`/`delete`.
- ✅ Text mode now writes into the real file; undo restores the exact previous content.

**M5 — Audio**
- capture + VAD, Groq STT, Whisper prompt hints, pending/decision timing.
- `texflow listen` with a live rich status line (listening / transcribing / writing / waiting).
- ✅ I can talk for 5 minutes into my thesis, with corrections and thinking-out-loud pauses, and the result compiles.
- ✅ Measured latency from end of speech to file write is ≤ 2 s on Groq for typical utterances (log timings).

**M6 — Setup, doctor, packaging**
- `init` wizard (runs on first use), `doctor`, local backend (`[local]` extra: faster-whisper + Ollama).
- README: GIF placeholder at the top, 3-line quickstart, how it works, config reference.
- GitHub Action: publish to PyPI on tag via trusted publishing.
- ✅ Fresh machine: `uv tool install texflow && texflow` gets to listening in under a minute.

---

## 8. Testing strategy
- **Fixture project** with known packages, bib entries (including two papers by the same author to test ambiguity), labels and a style sample.
- **Mocked LLM** returning canned decisions, so engine, validators and writer are tested deterministically.
- **Golden tests** for conventions detection (input preamble → expected rules).
- **Live smoke test** (skipped without `GROQ_API_KEY`): a few typed utterances against the fixture project, asserting validation passes.
- A latency log (`.texflow/timings.log`) for STT, LLM and validation per utterance.

---

## 9. Later (not v0.1)
- Watch mode (`%>` lines dictated by any external tool are converted on save)
- OpenAI-compatible proxy so other dictation apps' "AI polish" step can use texflow
- MCP server / Claude Code plugin
- VS Code extension with a hotkey and diff view for edits
- Homebrew tap / AUR package
