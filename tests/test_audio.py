import numpy as np
from autocut.audio import CircularAudioBuffer, list_input_devices
from autocut.vad import VoiceActivityDetector, calculate_rms


def test_list_input_devices():
    devices = list_input_devices()
    assert isinstance(devices, list)
    assert len(devices) > 0
    assert "name" in devices[0]
    assert "index" in devices[0]


def test_circular_audio_buffer_basic():
    # 1 second buffer at 16000 Hz
    buf = CircularAudioBuffer(capacity_seconds=1.0, sample_rate=16000)
    data = np.ones(8000, dtype=np.float32) * 0.5
    buf.write(data)

    recent = buf.get_recent(0.5)
    assert len(recent) == 8000
    assert np.allclose(recent, 0.5)


def test_circular_audio_buffer_wrap_around():
    # Buffer capacity 1000 samples
    buf = CircularAudioBuffer(capacity_seconds=0.1, sample_rate=10000)
    # Write 800 samples of 1.0
    buf.write(np.ones(800, dtype=np.float32) * 1.0)
    # Write 400 samples of 2.0 (should wrap around and overwrite oldest 200)
    buf.write(np.ones(400, dtype=np.float32) * 2.0)

    # Total stored should be max 1000 samples
    recent = buf.get_recent(0.1)
    assert len(recent) == 1000
    # The last 400 should be 2.0
    assert np.allclose(recent[-400:], 2.0)
    # The first 600 should be 1.0
    assert np.allclose(recent[:600], 1.0)


def test_calculate_rms():
    silent = np.zeros(1600, dtype=np.float32)
    assert calculate_rms(silent) == 0.0

    sine = np.sin(np.linspace(0, 2 * np.pi, 1600)).astype(np.float32)
    rms = calculate_rms(sine)
    assert rms > 0.5


def test_vad_silence():
    vad = VoiceActivityDetector()
    silent = np.zeros(1600, dtype=np.float32)
    assert not vad.is_speech_active(silent)
