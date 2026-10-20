<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/voxtex-lockup-on-dark.svg">
    <img alt="voxtex" src="assets/voxtex-lockup-on-light.svg" width="420">
  </picture>
</p>

<p align="center">
  <a href="pyproject.toml"><img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-blue.svg"></a>
  <a href="https://pypi.org/project/voxtex/"><img alt="PyPI" src="https://img.shields.io/pypi/v/voxtex.svg"></a>
  <a href="https://github.com/dhthz/voxtex/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/dhthz/voxtex/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
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

With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install "voxtex[gpu]"     # NVIDIA GPU; on other machines: uv tool install voxtex
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
cd my-thesis                             # the folder with your main .tex file
voxtex                                   # first time: a one-minute setup, then the editor
```

Setup looks at your machine first (GPU, Ollama, keys you already have) and only asks
what it can't decide: whether everything should stay on this machine, and a free
[Groq key](https://console.groq.com/keys), tested on the spot. Run `voxtex init` to
change it later and `voxtex doctor` to check everything.

```bash
voxtex edit chapters/results.tex         # open a specific chapter
voxtex index                             # show what voxtex learned about the thesis
```

In the editor, `ctrl+space` starts and stops recording (`esc` cancels); the text goes
at the cursor, or replaces the selection if you selected something. Say punctuation
("κόμμα", "full stop") and it becomes the mark; say "undo" ("αναίρεση", "undo twice") to
take back the last edit. If Whisper only heard noise, or wasn't sure of the language and
another one fits better, nothing odd is written: a message shows what it heard so you
can say it again. `ctrl+t` switches to unfiltered: what you say goes in exactly as
heard, with no AI (press it again for AI formatting). `ctrl+p` picks a theme (charcoal by default, voxtex,
catppuccin, nord, gruvbox, dracula, tokyo-night, solarized, rose-pine, ...); the
background always stays your terminal's. `ctrl+g` shows or hides small
graphs of what's in use (AI calls and tokens, speech time, and the GPU when speech
or the local model runs on it). `ctrl+z` undoes, `ctrl+o` opens any `.tex` file in the
thesis folder (type to filter, `esc` to go back), `ctrl+q` quits. voxtex saves after every voice edit, a moment after you stop typing, and on `ctrl+s` (with a backup in
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

voxtex is in beta (0.1). It's built and tested on a Greek thesis on WSL2 with an NVIDIA
GPU; other setups are less tested, so issues are welcome. The original build plan is
in [docs/voxtext_plan.md](docs/voxtext_plan.md) and
[docs/PLAN-ADDENDUM.md](docs/PLAN-ADDENDUM.md).

To work on it:

```bash
git clone https://github.com/dhthz/voxtex.git && cd voxtex
uv sync --extra gpu
uv run pytest            # tests run without a microphone, a GPU or an API key
uv run ruff check .
```

License: [MIT](LICENSE).
