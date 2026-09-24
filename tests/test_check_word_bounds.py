"""Contract test for the standalone word-boundary listening aid."""

from pathlib import Path
import subprocess
import sys


def test_check_word_bounds_exposes_required_arguments() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/check_word_bounds.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
        cwd=Path(__file__).parents[1],
    )

    assert result.returncode == 0
    assert "--transcript" in result.stdout
    assert "--audio" in result.stdout
