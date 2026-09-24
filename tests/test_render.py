"""Stage 2 contract tests for filter script and render logic."""

from pathlib import Path

import pytest

from autocut.models import KeepRange
from autocut.stages.render import filter_script, invert_keep_ranges, render_ranges


def test_filter_script_single_range() -> None:
    keep = [KeepRange(start=0.5, end=2.5)]
    script = filter_script(keep, audio_input=0)

    assert "[0:v]trim=start=0.500000:end=2.500000,setpts=PTS-STARTPTS[v0]" in script
    assert "[0:a]atrim=start=0.500000:end=2.500000,asetpts=PTS-STARTPTS[a0]" in script
    assert "[v0]null[vout]" in script
    assert "[a0]anull[aout]" in script


def test_filter_script_multiple_ranges_acrossfade() -> None:
    keep = [KeepRange(start=0.0, end=1.0), KeepRange(start=2.0, end=3.5)]
    script = filter_script(keep, audio_input=1)

    assert "[0:v]trim=start=0.000000:end=1.000000,setpts=PTS-STARTPTS[v0]" in script
    assert "[1:a]atrim=start=0.000000:end=1.000000,asetpts=PTS-STARTPTS[a0]" in script
    assert "[v0][v1]concat=n=2:v=1:a=0[vout]" in script
    assert "acrossfade=d=0.02:o=0[aout]" in script


def test_filter_script_empty_raises_value_error() -> None:
    with pytest.raises(ValueError):
        filter_script([], audio_input=0)


def test_invert_keep_ranges() -> None:
    keep = [KeepRange(start=1.0, end=3.0), KeepRange(start=4.0, end=6.5)]
    removed = invert_keep_ranges(keep, duration=10.0)

    assert [(r.start, r.end) for r in removed] == [(0.0, 1.0), (3.0, 4.0), (6.5, 10.0)]


def test_render_ranges_dry_run(tmp_path: Path) -> None:
    video = tmp_path / "video.mp4"
    output = tmp_path / "edited.mp4"
    script_path = tmp_path / "script.txt"
    keep = [KeepRange(start=0.0, end=2.0)]

    cmd = render_ranges(video, None, keep, output, script_path, dry_run=True)

    assert script_path.exists()
    assert not output.exists()
    assert cmd[0] == "ffmpeg"
    assert "-filter_complex_script" in cmd
    assert "h264_nvenc" in cmd
