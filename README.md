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
bibliography, how you write) and turns what you say into clean academic LaTeX.
Your chapter opens in voxtex's terminal editor, which works like Wispr Flow:
press `ctrl+space`, talk, press it again. What you said is written at the cursor
in your own words, only formatted as LaTeX (punctuation, citations, `\en{}` terms,
maths). Select text first and what you say is an instruction for it ("make this
more formal", "add the Broekstra citation"). The mic records only between the two
presses, so typing and clicks never end up in the thesis, and nothing you say is
ever dropped. Every write is checked first.

<!-- Demo GIF goes here: assets/voxtex-demo.gif (dictating into a chapter in the voxtex editor) -->

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

```bash
voxtex init                              # speech, free AI keys (tested on the spot), privacy, mic
voxtex doctor                            # check providers, models, speech speed, mic and TeX
```

`voxtex init` also runs by itself the first time you use voxtex. Then, from the
thesis folder:

```bash
voxtex index                             # show what voxtex learned about the thesis
voxtex                                   # open your last chapter in the editor and talk
voxtex edit chapters/results.tex         # open a specific chapter
```

In the editor, `ctrl+space` starts and stops recording (`esc` cancels); the text goes
at the cursor, or replaces the selection if you selected something. `ctrl+z` undoes,
`ctrl+o` opens another chapter, `ctrl+q` quits. voxtex saves after every voice edit (with a backup in
`.voxtex/backups/` before the first save), so your PDF viewer and other editors see
the changes right away.

To keep using your own editor, there are two other ways in. `voxtex listen` writes
at a `%> HERE` line in the file. `voxtex watch` works with any dictation app
(Voxtype, Windows `Win+H`, macOS dictation): dictate onto lines that start with `%>`
and save, and each block of `%>` lines becomes LaTeX.

## What you control

- **The files.** voxtex writes where your cursor is (or rewrites what you
  selected) in the chapter you opened, and nowhere else. Saves are atomic, so
  other programs never see half a file. Before the first save of a session it
  copies the file to `.voxtex/backups/`. Voice edits are part of the editor's
  undo, and a change to a selection you've since edited is written beside it instead.
- **What leaves your machine.** The AI model gets what you said, about 800
  characters of text before the cursor (and the selection, if any), and the few bibliography entries and
  labels that match. Your audio stays local when speech-to-text runs locally.
  `--private` keeps everything on your machine: local Whisper and a local model
  through [Ollama](https://ollama.com).
- **What gets written.** Before anything touches a file, voxtex checks that every
  citation key is in your `.bib`, every reference points to a real label, new
  labels follow your naming pattern, every number was actually said, braces and
  environments balance, and (for unfamiliar commands) that it compiles. A
  failure gets one automatic retry. If that fails too, the text is still written,
  tinted yellow, with a short message saying what to look at (`listen` and `watch`
  add a `% voxtex-check:` comment instead). If no AI answers at all, your words go
  in exactly as heard.
- **The cost.** The default setup is free: a chain of free tiers (Groq, then
  Gemini and OpenRouter if you add keys) that ends in a local model. When one
  runs out, the same request moves to the next. A decision uses about 1,300
  input tokens; `.voxtex/timings.log` records every call.

## Commands

| If you want to | Run |
| --- | --- |
| Set up, or change the setup | `voxtex init` |
| Check that everything works | `voxtex doctor [PATH] [--quick]` |
| Talk into the thesis | `voxtex`, or `voxtex edit [CHAPTER]` |
| Talk into a file open in your own editor | `voxtex listen [PATH]`, at a `%> HERE` line |
| Use another dictation app | `voxtex watch [PATH]` |
| Undo `listen`/`watch` writes, even from earlier sessions | `voxtex undo [N]` |
| See what voxtex learned about the thesis | `voxtex index [PATH] [-v] [--json]` |
| Pick a microphone | `voxtex listen --list-mics`, then `--mic NAME` |
| Keep everything local | `--private` on `edit`, `listen` or `watch` |
| Try without changing files | `voxtex listen --dry-run` |

## Configuration

`voxtex init` writes `~/.config/voxtex/config.toml`, and keys go in
`~/.config/voxtex/.env`. A `.voxtex.toml` (or `.env`) in the thesis folder
overrides them for that thesis. Everything is optional:

```toml
[llm]
chain = ["groq-gptoss", "groq-qwen", "gemini", "openrouter", "local"]

[providers.local]
model = "qwen2.5-coder:14b"          # any model you've pulled into Ollama

[writing]
style = "academic"                   # academic | clean
language = "auto"                    # auto | el | en

[stt]
mode = "auto"                        # auto | local | groq | openai | external
local_model = "auto"                 # auto | small | medium | large-v3-turbo

[listen]
utterance_pause = 0.9                # seconds of silence that end an utterance
decision_pause = 2.5                 # seconds of silence before a thought is written

[validate]
compile = "auto"                     # auto | always | off
```

## Project status

voxtex has not had a release, and anything can still change. It follows the
build plan in [docs/voxtext_plan.md](docs/voxtext_plan.md) and
[docs/PLAN-ADDENDUM.md](docs/PLAN-ADDENDUM.md). Milestones M0–M5 are done: the
project indexer, the engine and provider chain, validation, writing and undo,
watch mode, listening, `voxtex init` and `voxtex doctor`, and the editor. Next is a
PyPI release.

To work on it:

```bash
uv sync --extra gpu
uv run pytest            # tests run without a microphone, a GPU or an API key
uv run ruff check .
```

License: MIT.
