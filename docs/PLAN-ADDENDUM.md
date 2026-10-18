# Voxtex — Plan Addendum: Free-Tier Setup

> Read this **together with PLAN.md**. It only adds or changes things; anything not mentioned here stays as PLAN.md describes it.
> Where this file conflicts with PLAN.md, **this file wins**.

## How to use this addendum (instructions for Claude Code)

- Apply these changes **as part of the milestones they're attached to** (see §7), not as a separate project at the end.
- Same rule as before: stop for my review after each milestone.

---

## 1. Name: `voxtex`

- Package, CLI command, config paths and project folder all use `voxtex`:
  - CLI: `voxtex`, `voxtex listen`, `voxtex doctor`…
  - Global config: `~/.config/voxtex/config.toml`
  - Project config: `.voxtex.toml`; project state folder: `.voxtex/`
- README headline: **"Voxtex: talk to your thesis, get LaTeX that compiles."**
- Before publishing, check that `voxtex` is free on PyPI and GitHub.

---

## 2. Goal: $0 by default

The default setup must work with **no paid accounts at all**:
- Speech → **local** (unlimited, private, free)
- LLM → **chain of free tiers**, ending in a **local model**, so it never stops working

Paid keys are optional and just get added to the chain.

---

## 3. Speech: local by default

### 3.1 Built-in local STT (new default)
- `faster-whisper` moves from the optional `[local]` extra into the **default install**. Cloud STT (Groq/OpenAI) becomes optional.
- Default model: a **multilingual** model (`small` for speed, `large-v3-turbo` if a GPU or Apple Silicon is detected). Never an English-only (`.en`) model, because I dictate in Greek and English.
- Model is downloaded on first run with a progress bar; size shown before download.
- Auto-select device: CUDA → Apple Silicon (Metal/CoreML where supported) → CPU with int8.
- `voxtex doctor` reports transcription speed (real-time factor) on a 5-second test clip and suggests a smaller model if it's too slow.

### 3.2 External dictation mode (Voxtype, Omarchy, or any dictation app)
New command:
```
voxtex watch [PATH]
```
- I dictate with **any** tool that types at the cursor (Voxtype on Omarchy via `F9` / `Super+Ctrl+X`, macOS dictation, Wispr Flow, whisrs…) onto lines starting with `%>` in my `.tex` file.
- On file save, Voxtex finds the `%>` lines, treats them as the transcript, runs the normal engine, and replaces them in place with validated LaTeX.
- Consecutive `%>` lines are one utterance block. A blank line between them separates blocks.
- Because `%>` is a LaTeX comment, the document still compiles while I'm dictating.
- Writes are atomic, backed up and undoable exactly like `listen` mode.
- File watching via `watchdog`; debounce saves by ~300 ms.
- In this mode Voxtex uses **no microphone and no STT at all**.

### 3.3 Config
```toml
[stt]
mode = "local"                    # local | groq | openai | external (watch mode)
local_model = "auto"              # auto | small | medium | large-v3-turbo
language = "auto"                 # auto | el | en
```

---

## 4. LLM: free-tier fallback chain

### 4.1 Chain
Replace the single LLM backend with an **ordered chain** of OpenAI-compatible providers:

```toml
[llm]
chain = ["groq-gptoss", "groq-llama", "gemini", "openrouter", "local"]

[providers.groq-gptoss]
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"
model = "openai/gpt-oss-120b"

[providers.groq-llama]
base_url = "https://api.groq.com/openai/v1"
api_key_env = "GROQ_API_KEY"
model = "llama-3.3-70b-versatile"

[providers.gemini]
base_url = "https://generativelanguage.googleapis.com/v1beta/openai/"
api_key_env = "GEMINI_API_KEY"
model = "gemini-flash"            # verify current name

[providers.openrouter]
base_url = "https://openrouter.ai/api/v1"
api_key_env = "OPENROUTER_API_KEY"
model = "<a :free model>"         # verify current free models

[providers.local]
base_url = "http://localhost:11434/v1"
model = "qwen2.5-coder:14b"
```
- Model names and free limits change often: `voxtex doctor` must **verify each provider's model exists** (list models endpoint) and warn instead of crashing.
- Providers without a configured key are **skipped silently**.

### 4.2 Failover rules
- On **429 / quota exceeded**: mark the provider exhausted until its reset time (from `retry-after` or rate-limit headers, else 1 hour), move to the next provider **for the same request**, with no visible error.
- On **timeout (>8 s) or 5xx**: one retry, then move on.
- On **invalid JSON from the model**: one retry on the same provider, then move on.
- If structured-output support differs per provider, fall back to "JSON in the prompt + pydantic parse".
- Exhausted state persists in `.voxtex/quota.json` so restarts don't hammer a dead provider.

### 4.3 Quota visibility
- Read rate-limit headers where available (Groq exposes remaining requests/tokens).
- Status line shows the active provider and remaining quota, e.g.:
  `🎙 listening · groq-gptoss · ~142K tok left today`
- When falling back, show one short notice: `↪ switched to gemini (groq quota reached)`.

### 4.4 Privacy ordering
- Each provider entry gets `private = true | false` (local is `true`; free cloud tiers default to `false` unless I mark them).
- Option `--private` (and config `privacy = "strict"`) restricts the chain to `private = true` providers only, for unpublished thesis material.
- `voxtex init` explains this in one sentence and asks which mode I want.

---

## 5. Spend fewer tokens (so free tiers last)

Target: **≤ 2,000 input tokens per typical call** (down from ~3–4K).

1. **Decide less often**: utterances accumulate; the LLM is called only on a decision pause or when the pending transcript is large enough. Remove calls that would just return `wait`.
2. **Cheap pre-filter**: skip the LLM entirely for empty or filler-only transcripts ("um", "hmm", "let me think") using a small local word list.
3. **Compact conventions**: send rules as short lines (`cite: \parencite/\textcite`, `ref: \cref`, `tables: booktabs`, `units: siunitx`), not prose.
4. **Macros**: only names + arity, and only those whose names relate to the transcript or nearby text (plus a short always-include list).
5. **Style samples**: 1 paragraph by default; include the table/figure sample only when the transcript mentions a table/figure.
6. **Bib**: only the top 3 fuzzy matches, as `key | first author | year | short title`.
7. **Nearby text**: ~800 characters before the marker instead of ~1,500.
8. **Stable prefix first**: put the static part of the prompt (rules, conventions) first and keep it byte-identical across calls, so providers with prompt caching can reuse it.
9. Log input/output tokens per call to `.voxtex/timings.log` and print a per-session total on exit.

---

## 6. Setup wizard changes (`voxtex init`)

New flow:
1. **Speech**: "Local (free, private) / I use an external dictation app (watch mode) / Groq cloud".
   - If Omarchy + Voxtype is detected (`voxtype` in PATH), suggest watch mode and remind me to pick a multilingual Voxtype model (`voxtype setup model`).
2. **AI providers**: offer each free provider with a link to get a key; every key is **tested immediately**; skipping is fine.
3. **Local fallback**: detect Ollama; if missing, show the one-line install command and the `ollama pull` command for the default model.
4. **Privacy**: normal vs strict (local-only).
5. Print the final chain in order, e.g. `groq-gptoss → gemini → local`.

---

## 7. Where these changes go in the milestones

| PLAN.md milestone | Add from this addendum |
|---|---|
| **M0** | Rename to `voxtex` (§1). |
| **M2** (engine, text mode) | Provider chain + failover (§4.1–4.2), token-saving prompt rules (§5.3–5.8). |
| **M3** (validators) | Nothing new. |
| **M4** (writer + undo) | `voxtex watch` external dictation mode (§3.2). |
| **M5** (audio) | Local faster-whisper as default (§3.1), filler pre-filter and fewer decision calls (§5.1–5.2), quota status line (§4.3). |
| **M6** (setup + packaging) | New `init` flow (§6), privacy mode (§4.4), doctor checks for models, STT speed and providers. |

### New acceptance criteria
- ✅ With **no API keys at all** (only Ollama installed), `voxtex listen` works end to end.
- ✅ With only a Groq key, exhausting the quota (simulated 429 in tests) switches to the next provider mid-session without losing the utterance.
- ✅ `voxtex watch` converts `%>` lines dictated by Voxtype into valid LaTeX on save, and `voxtex undo` restores them.
- ✅ Average input tokens per call ≤ 2,000 on my real thesis (from the timings log).
- ✅ `--private` never sends anything to a non-private provider (unit test asserts no network calls to cloud base URLs).
