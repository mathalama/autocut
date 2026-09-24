"""Stage 1 CLI contract tests."""

from typer.testing import CliRunner

from autocut.cli import app


def test_cli_exposes_stage_one_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "ingest" in result.stdout
    assert "vad" in result.stdout
    assert "transcribe" in result.stdout
    assert "run" in result.stdout
