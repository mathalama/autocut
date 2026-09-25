from autocut.cut import compute_keep_intervals, format_duration


def test_format_duration():
    assert format_duration(45) == "00:45"
    assert format_duration(125) == "02:05"
    assert format_duration(3665) == "01:01:05"


def test_compute_keep_intervals_merges_close_segments():
    # Two speech intervals separated by small gap (0.2s) should merge with margin=0.1
    speech = [(1.0, 2.0), (2.2, 3.5)]
    keep = compute_keep_intervals(speech, total_duration=10.0, min_pause=0.5, margin=0.1)
    assert len(keep) == 1
    assert keep[0][0] == 0.9  # 1.0 - 0.1
    assert keep[0][1] == 3.6  # 3.5 + 0.1


def test_compute_keep_intervals_cuts_large_silence():
    # Two speech intervals separated by 4.0s gap -> should produce 2 separate keep intervals
    speech = [(1.0, 3.0), (7.0, 9.0)]
    keep = compute_keep_intervals(speech, total_duration=10.0, min_pause=0.6, margin=0.15)
    assert len(keep) == 2
    # First interval
    assert keep[0][0] == 0.85
    assert keep[0][1] == 3.15
    # Second interval
    assert keep[1][0] == 6.85
    assert keep[1][1] == 9.15
