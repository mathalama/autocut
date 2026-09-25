from autocut.engine import clean_hallucinations


def test_clean_hallucinations_filters_ghosts_on_weak_audio():
    # Whisper hallucination on quiet mic (RMS < 0.006)
    text = "Thank you for watching."
    assert clean_hallucinations(text, rms_energy=0.004) == ""

    text = "Subtitles by..."
    assert clean_hallucinations(text, rms_energy=0.003) == ""


def test_clean_hallucinations_keeps_legitimate_speech():
    # Real speech should not be filtered
    text = "Hello and welcome to this screencast."
    assert clean_hallucinations(text, rms_energy=0.04) == text

    # "Thank you" with real speech energy should be preserved
    assert clean_hallucinations("Thank you", rms_energy=0.03) == "Thank you"


def test_clean_hallucinations_filters_repetition_loops():
    # Pathological repetition loops (4 or more words)
    text = "you you you you"
    assert clean_hallucinations(text, rms_energy=0.03) == ""

    text = "test test test test"
    assert clean_hallucinations(text, rms_energy=0.03) == ""


def test_is_valid_segment():
    from autocut.engine import is_valid_segment
    from unittest.mock import Mock

    # High probability of no speech -> reject
    bad_seg = Mock(no_speech_prob=0.7, avg_logprob=-0.2, compression_ratio=1.0)
    assert not is_valid_segment(bad_seg)

    # Low log probability (whisper guessing/hallucinating) -> reject
    bad_logprob = Mock(no_speech_prob=0.1, avg_logprob=-1.3, compression_ratio=1.0)
    assert not is_valid_segment(bad_logprob)

    # Repetition loop compression -> reject
    bad_comp = Mock(no_speech_prob=0.1, avg_logprob=-0.3, compression_ratio=2.8)
    assert not is_valid_segment(bad_comp)

    # Confident real speech segment -> accept
    good_seg = Mock(no_speech_prob=0.05, avg_logprob=-0.3, compression_ratio=1.1)
    assert is_valid_segment(good_seg)


def test_engine_single_pass_transcription():
    from autocut.engine import RealtimeSubtitleEngine
    from unittest.mock import Mock
    import numpy as np

    engine = RealtimeSubtitleEngine(engine_type="whisper")
    engine.model = Mock()
    mock_seg = Mock(text=" Hello world", no_speech_prob=0.01, avg_logprob=-0.2, compression_ratio=1.1)
    engine.model.transcribe.return_value = ([mock_seg], None)

    dummy_audio = np.zeros(16000, dtype=np.float32)
    result = engine._transcribe_audio(dummy_audio, active_language="en")
    assert result == "Hello world"
    assert engine.model.transcribe.call_count == 1


def test_engine_higgs_import_guidance(monkeypatch):
    import pytest
    import sys
    from autocut.higgs import HiggsSTTModel

    # When torch is not available, HiggsSTTModel must raise an actionable ImportError
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(ImportError) as exc_info:
        HiggsSTTModel()
    assert "Higgs STT requires PyTorch" in str(exc_info.value)

