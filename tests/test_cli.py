from typer.testing import CliRunner

from voxtex import __version__
from voxtex.cli import app

runner = CliRunner()


def test_help_lists_all_commands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("listen", "text", "init", "doctor", "undo", "index"):
        assert cmd in result.output


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_undo_rejects_zero():
    result = runner.invoke(app, ["undo", "0"])
    assert result.exit_code != 0
