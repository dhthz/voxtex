<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/voxtex-lockup-on-dark.svg">
    <img alt="voxtex" src="assets/voxtex-lockup-on-light.svg" width="280">
  </picture>
</p>

<p align="center">
  <a href="pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg"></a>
  <img alt="Status: pre-release" src="https://img.shields.io/badge/status-pre--release-orange.svg">
  <a href="pyproject.toml"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

**Voxtex: talk to your thesis, get LaTeX that compiles.**

You point it at a LaTeX project and talk, in Greek, English or both. It reads
how your thesis is set up (packages, citation commands, macros, labels, the
bibliography, how you write) and turns what you say into clean academic LaTeX
at a `%> HERE` marker in your file. Corrections like "no, it was 0.61" or "όχι,
σβήσ' το" edit what it just wrote. Every write is checked first and can be undone.

<!-- Demo GIF goes here: assets/voxtex-demo.gif (voxtex listen writing into a thesis) -->

## Install

voxtex is not on PyPI yet. Install it from source with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/dhthz/voxtex.git
cd voxtex
uv tool install ".[gpu]"     # NVIDIA GPU; on other machines: uv tool install .
```

On Linux and WSL, the microphone needs three system packages:

```bash
sudo apt install -y libportaudio2 libasound2-plugins pulseaudio-utils
```

| Host | Speech-to-text it picks | Tested |
| --- | --- | --- |
| Linux or WSL2, NVIDIA GPU | local `large-v3-turbo` on the GPU | Yes (WSL2, Ubuntu 24.04, RTX 3060) |
| Linux or WSL2, no GPU | Groq Whisper if you have a key, else local `small` on the CPU | No |
| macOS, Apple silicon | Groq Whisper if you have a key, else local `small` on the CPU | No |
| Windows without WSL | — | No; use WSL2 |

A TeX installation (`pdflatex` or `xelatex`) is optional. With one, voxtex
compiles anything unusual it writes inside your real preamble before writing it.

## First run

Get a free key at [console.groq.com](https://console.groq.com) and put it in a
`.env` file in your thesis folder (or in `~/.config/voxtex/.env` for every project):

```bash
echo 'GROQ_API_KEY=gsk_...' >> ~/thesis/.env
```

Then, from the thesis folder:

```bash
echo '%> HERE' >> chapters/results.tex   # where new text goes
voxtex index                             # show what voxtex learned about the thesis
voxtex text --dry-run                    # type instead of talk; nothing is written
voxtex listen                            # talk; text appears above %> HERE
voxtex undo 3                            # revert the last three writes
```

`voxtex` with no command is `voxtex listen`. To dictate with another tool
instead (Voxtype, Windows `Win+H`, macOS dictation), run `voxtex watch`, dictate
onto lines that start with `%>`, and save: each block of `%>` lines becomes LaTeX.

## What you control

- **The files.** voxtex writes only above the `%> HERE` marker, or into the
  file you name with `--file`. Writes are atomic, so your editor never sees half
  a file. Before the first write of a session it copies the file to
  `.voxtex/backups/`. `voxtex undo` reverts writes exactly, even from earlier
  sessions, and text you have edited yourself is never overwritten.
- **What leaves your machine.** The AI model gets what you said, about 800
  characters of text before the marker, and the few bibliography entries and
  labels that match. Your audio stays local when speech-to-text runs locally.
  `--private` keeps everything on your machine: local Whisper and a local model
  through [Ollama](https://ollama.com).
- **What gets written.** Before anything touches a file, voxtex checks that every
  citation key is in your `.bib`, every reference points to a real label, new
  labels follow your naming pattern, every number was actually said, braces and
  environments balance, and (for unfamiliar commands) that it compiles. A
  failure gets one automatic retry. If that fails too, the text is written with
  a `% voxtex-check:` comment saying what to look at.
- **The cost.** The default setup is free: a chain of free tiers (Groq, then
  Gemini and OpenRouter if you add keys) that ends in a local model. When one
  runs out, the same request moves to the next. A decision uses about 1,300
  input tokens; `.voxtex/timings.log` records every call.

## Commands

| If you want to | Run |
| --- | --- |
| Talk into the thesis | `voxtex listen [PATH]` |
| Type instead of talk (same engine) | `voxtex text [PATH]` |
| Use another dictation app | `voxtex watch [PATH]` |
| Undo the last N writes | `voxtex undo [N]` |
| See what voxtex learned about the thesis | `voxtex index [PATH] [-v] [--json]` |
| Pick a microphone | `voxtex listen --list-mics`, then `--mic NAME` |
| Keep everything local | `--private` on `listen`, `text` or `watch` |
| Try without changing files | `--dry-run` on `listen` or `text` |

## Configuration

Settings come from `~/.config/voxtex/config.toml`, then `.voxtex.toml` in the
thesis folder. Everything is optional:

```toml
[llm]
chain = ["groq-gptoss", "groq-qwen", "gemini", "openrouter", "local"]

[providers.local]
model = "qwen2.5-coder:14b"          # any model you've pulled into Ollama

[writing]
style = "academic"                   # academic | clean
language = "auto"                    # auto | el | en

[stt]
mode = "auto"                        # auto | local | groq | openai
local_model = "auto"                 # auto | small | medium | large-v3-turbo

[listen]
utterance_pause = 0.7                # seconds of silence that end an utterance
decision_pause = 2.5                 # seconds of silence before a thought is written

[validate]
compile = "auto"                     # auto | always | off
```

## Project status

voxtex has not had a release, and anything can still change. It follows the
build plan in [docs/voxtext_plan.md](docs/voxtext_plan.md) and
[docs/PLAN-ADDENDUM.md](docs/PLAN-ADDENDUM.md). Milestones M0–M5 are done: the
project indexer, the engine and provider chain, validation, writing and undo,
watch mode, and listening. Next are `voxtex init`, `voxtex doctor` and a PyPI
release.

To work on it:

```bash
uv sync --extra gpu
uv run pytest            # tests run without a microphone, a GPU or an API key
uv run ruff check .
```

License: MIT.
