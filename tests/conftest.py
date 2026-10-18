import shutil
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_thesis(tmp_path) -> Path:
    """A private copy of the fixture project, so tests can write .voxtex/ into it."""
    dst = tmp_path / "sample-thesis"
    shutil.copytree(FIXTURES / "sample-thesis", dst, ignore=shutil.ignore_patterns(".voxtex"))
    return dst


@pytest.fixture
def make_project(tmp_path):
    """Write {relative path: content} into a fresh project folder and return it."""

    def _make(files: dict[str, str]) -> Path:
        root = tmp_path / "proj"
        for rel, content in files.items():
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        return root

    return _make


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test from an empty directory, so a developer's .env never leaks in."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
