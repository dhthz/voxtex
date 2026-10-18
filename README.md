# voxtex

**Voxtex: talk to your thesis, get LaTeX that compiles.**

Talk into your LaTeX project. voxtex reads your setup (packages, citation style,
macros, labels, bib, writing style), listens to free speech, and writes clean,
academic LaTeX into your file that compiles in **your** template.

> Status: early development (milestone M0, scaffold). See `docs/voxtext_plan.md` and `docs/PLAN-ADDENDUM.md`.

## Development

```bash
uv sync
uv run voxtex --help
uv run pytest
uv run ruff check .
```
