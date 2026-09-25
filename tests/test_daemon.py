from unittest.mock import Mock, patch
import numpy as np
from autocut.daemon import DaemonClient
from autocut.stt.base import STTSegment


def test_daemon_client_health():
    client = DaemonClient("http://127.0.0.1:8766")
    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = Mock()
        mock_resp.status = 200
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_resp

        assert client.is_running() is True


def test_daemon_client_transcribe():
    client = DaemonClient("http://127.0.0.1:8766")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_resp = Mock()
        mock_resp.read.return_value = b'{"segments": [{"index": 1, "start": 0.0, "end": 1.0, "text": "test audio"}]}'
        mock_resp.__enter__ = Mock(return_value=mock_resp)
        mock_resp.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_resp

        segments = client.transcribe(dummy_audio)
        assert len(segments) == 1
        assert segments[0].text == "test audio"
        assert segments[0].start == 0.0
