from typer.testing import CliRunner
from autocut.cli import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "devices" in result.output
    assert "live" in result.output


def test_cli_devices():
    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 0
    assert "Available Audio Input Devices" in result.output


def test_cli_live_help():
    result = runner.invoke(app, ["live", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--device" in result.output
    assert "--port" in result.output
