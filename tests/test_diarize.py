from unittest.mock import MagicMock

from src.diarize import diarize_audio


class FakeTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


def test_diarize_audio_extracts_speaker_turns(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [
        (FakeTurn(0.0, 3.0), None, "SPEAKER_00"),
        (FakeTurn(3.0, 6.0), None, "SPEAKER_01"),
    ]
    mock_pipeline = MagicMock(return_value=fake_diarization)

    result = diarize_audio(audio_path, hf_token="test-token", pipeline=mock_pipeline)

    assert result == [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
    ]
    mock_pipeline.assert_called_once_with(str(audio_path))
    fake_diarization.itertracks.assert_called_once_with(yield_label=True)
