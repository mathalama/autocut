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
    assert "--engine" in result.output


def test_cli_live_mock_start(monkeypatch):
    from unittest.mock import Mock, patch
    with patch("autocut.cli.OverlayServer") as mock_server, \
         patch("autocut.cli.RealtimeSubtitleEngine") as mock_engine, \
         patch("time.sleep", side_effect=KeyboardInterrupt):
        result = runner.invoke(app, ["live", "--port", "9999", "--hotkey", ""])
        assert result.exit_code == 0
        assert mock_server.return_value.start.called
        assert mock_engine.return_value.start.called


def test_cli_record_help():
    result = runner.invoke(app, ["record", "--help"])
    assert result.exit_code == 0
    assert "--engine" in result.output
    assert "--output" in result.output
    assert "--clipboard" in result.output


def test_cli_transcribe_help():
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--engine" in result.output
    assert "--output" in result.output


def test_cli_daemon_help():
    result = runner.invoke(app, ["daemon", "--help"])
    assert result.exit_code == 0
    assert "--port" in result.output
    assert "--engine" in result.output


