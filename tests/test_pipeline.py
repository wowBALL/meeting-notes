from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.pipeline import process_file


def make_config(tmp_path: Path) -> Config:
    return Config(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
        anthropic_api_key="sk-ant-test",
        hf_token="hf-test-token",
        claude_model="claude-opus-4-8",
        whisper_model="small",
    )


def _mock_convert_to_wav():
    return patch("src.pipeline.convert_to_wav", side_effect=lambda src, dst: dst)


def test_process_file_saves_transcript_and_summary(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        ),
        patch(
            "src.pipeline.summarize_transcript",
            return_value="## ประเด็นสำคัญ\n- ทดสอบ",
        ),
    ):
        meeting_dir = process_file(audio_path, config)

    expected_dir = config.meetings_dir / f"{date.today().isoformat()}-weekly-standup"
    assert meeting_dir == expected_dir
    assert (meeting_dir / "transcript.md").exists()
    assert (meeting_dir / "summary.md").read_text(encoding="utf-8") == "## ประเด็นสำคัญ\n- ทดสอบ"
    assert (meeting_dir / "weekly-standup.mp3").exists()
    assert not audio_path.exists()


def test_process_file_continues_without_diarization_on_failure(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", side_effect=RuntimeError("model load failed")),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)

    transcript_text = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "ผู้พูด 1" in transcript_text


def test_process_file_moves_to_failed_when_conversion_fails(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        patch("src.pipeline.convert_to_wav", side_effect=RuntimeError("ffmpeg not found")),
        pytest.raises(RuntimeError, match="ffmpeg not found"),
    ):
        process_file(audio_path, config)

    assert not audio_path.exists()
    assert (config.failed_dir / "broken.mp3").exists()
    error_log = config.failed_dir / "broken.error.log"
    assert "Audio conversion failed" in error_log.read_text(encoding="utf-8")


def test_process_file_moves_to_failed_when_transcription_fails(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch("src.pipeline.transcribe_audio", side_effect=RuntimeError("network error")),
        patch("time.sleep"),
        pytest.raises(RuntimeError, match="network error"),
    ):
        process_file(audio_path, config)

    assert not audio_path.exists()
    assert (config.failed_dir / "broken.mp3").exists()
    error_log = config.failed_dir / "broken.error.log"
    assert "Transcription failed" in error_log.read_text(encoding="utf-8")


def test_process_file_moves_to_failed_when_summarization_fails(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        ),
        patch(
            "src.pipeline.summarize_transcript",
            side_effect=RuntimeError("claude api error"),
        ),
        patch("time.sleep"),
        pytest.raises(RuntimeError, match="claude api error"),
    ):
        process_file(audio_path, config)

    assert not audio_path.exists()
    assert (config.failed_dir / "broken.mp3").exists()
    error_log = config.failed_dir / "broken.error.log"
    assert "Summarization failed" in error_log.read_text(encoding="utf-8")


def test_process_file_moves_to_failed_when_rendering_fails(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        ),
        patch(
            "src.pipeline.render_transcript_markdown",
            side_effect=RuntimeError("render boom"),
        ),
        pytest.raises(RuntimeError, match="render boom"),
    ):
        process_file(audio_path, config)

    assert not audio_path.exists()
    assert (config.failed_dir / "broken.mp3").exists()
    error_log = config.failed_dir / "broken.error.log"
    assert "Rendering failed" in error_log.read_text(encoding="utf-8")


def test_process_file_moves_to_failed_when_save_fails(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
        patch("src.pipeline.save_outputs", side_effect=OSError("disk full")),
        pytest.raises(OSError, match="disk full"),
    ):
        process_file(audio_path, config)

    assert not audio_path.exists()
    assert (config.failed_dir / "broken.mp3").exists()
    error_log = config.failed_dir / "broken.error.log"
    assert "Save failed" in error_log.read_text(encoding="utf-8")


def test_process_file_notes_diarization_failure_in_transcript(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", side_effect=RuntimeError("model load failed")),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)

    transcript_text = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "ไม่สามารถแยกผู้พูดได้อัตโนมัติ" in transcript_text


def test_process_file_threads_diarization_pipeline_to_diarize_audio(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    sentinel_pipeline = object()
    mock_diarize = MagicMock(
        return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
    )

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", mock_diarize),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config, diarization_pipeline=sentinel_pipeline)

    assert mock_diarize.call_args.kwargs["pipeline"] is sentinel_pipeline


def test_process_file_threads_whisper_model_to_transcribe_audio(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    sentinel_model = object()
    mock_transcribe = MagicMock(
        return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}]
    )

    with (
        _mock_convert_to_wav(),
        patch("src.pipeline.transcribe_audio", mock_transcribe),
        patch(
            "src.pipeline.diarize_audio",
            return_value=[{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}],
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config, whisper_model=sentinel_model)

    assert mock_transcribe.call_args.kwargs["model"] is sentinel_model
    assert mock_transcribe.call_args.kwargs["model_size"] == config.whisper_model
