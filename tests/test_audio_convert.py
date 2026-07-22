import subprocess
from unittest.mock import patch

import pytest

from src.audio_convert import convert_to_wav


def test_convert_to_wav_invokes_ffmpeg_and_returns_output_path(tmp_path):
    audio_path = tmp_path / "input.mp3"
    audio_path.write_bytes(b"fake audio")
    output_path = tmp_path / "output.wav"

    with patch("src.audio_convert.subprocess.run") as mock_run:
        result = convert_to_wav(audio_path, output_path)

    assert result == output_path
    args = mock_run.call_args.args[0]
    assert args[0] == "ffmpeg"
    assert str(audio_path) in args
    assert str(output_path) in args
    assert mock_run.call_args.kwargs["check"] is True


def test_convert_to_wav_raises_on_ffmpeg_failure(tmp_path):
    audio_path = tmp_path / "input.mp3"
    audio_path.write_bytes(b"fake audio")
    output_path = tmp_path / "output.wav"

    with patch(
        "src.audio_convert.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            convert_to_wav(audio_path, output_path)
