"""T01 scaffold smoke test: the package imports and the CLI help renders."""

from typer.testing import CliRunner

from caliper import __version__
from caliper.cli import app

runner = CliRunner()


def test_version_is_defined() -> None:
    assert __version__ == "0.1.0"


def test_cli_help_exits_zero() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.output
