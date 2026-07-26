import subprocess
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import Config
from src.job import (
    JOB_SUFFIX,
    NO_SUMMARY_MODEL,
    read_model,
    read_transcript,
    record_transcript,
    write_job,
)
from src.pipeline import process_file
from src.segments import WAV_HEADER_ALLOWANCE, finish_session, part_filename, session_dir_for, write_manifest

# See tests/test_segments.py: finish_session decides which parts are "real" by
# size on disk, so a fixture part must be comfortably larger than the allowance.
_FAKE_WAV_BYTES = b"fake wav bytes " * (WAV_HEADER_ALLOWANCE // 16 + 2)


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

    expected_dir = config.meetings_dir / f"{date.today().isoformat()}_weekly-standup"
    assert meeting_dir == expected_dir
    assert (meeting_dir / "transcript.md").exists()
    summary = (meeting_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.startswith("## ประเด็นสำคัญ\n- ทดสอบ")
    assert summary.endswith(f"---\nสรุปด้วย {config.claude_model}\n")
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


def test_process_file_keeps_the_transcript_when_summarization_fails(tmp_path):
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
        pytest.raises(RuntimeError, match="claude api error"),
    ):
        process_file(audio_path, config)

    # the transcript costs a GPU pass over the whole recording; a failed summary
    # must never be what throws it away
    transcript_path = (
        config.meetings_dir / f"{date.today().isoformat()}_broken" / "transcript.md"
    )
    assert "สวัสดีครับ" in transcript_path.read_text(encoding="utf-8")
    assert (config.failed_dir / "broken.mp3").exists()
    error_log = (config.failed_dir / "broken.error.log").read_text(encoding="utf-8")
    assert "Summarization failed" in error_log
    assert str(transcript_path) in error_log


def test_process_file_does_not_retry_summarization_itself(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    mock_summarize = MagicMock(side_effect=RuntimeError("claude api error"))

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
        patch("src.pipeline.summarize_transcript", mock_summarize),
        patch("time.sleep"),
        pytest.raises(RuntimeError, match="claude api error"),
    ):
        process_file(audio_path, config)

    # summarize_transcript retries every API call internally; retrying it again
    # here would re-run an entire map-reduce for one permanently dead chunk
    assert mock_summarize.call_count == 1


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
        patch("src.pipeline.save_summary", side_effect=OSError("disk full")),
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


def test_process_file_uses_the_model_from_the_job_file(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(config.inbox_dir, "weekly-standup", "claude-sonnet-5")
    summarize = MagicMock(return_value="## สรุป")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        process_file(audio_path, config)

    assert summarize.call_args.kwargs["model"] == "claude-sonnet-5"


def test_process_file_falls_back_to_the_config_model_without_a_job_file(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "dropped.mp3"
    audio_path.write_bytes(b"fake audio")
    summarize = MagicMock(return_value="## สรุป")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        process_file(audio_path, config)

    assert summarize.call_args.kwargs["model"] == config.claude_model


def test_process_file_falls_back_when_the_job_file_is_corrupt(tmp_path):
    # the transcript costs a full GPU pass -- unreadable job bytes must not
    # throw that away
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    (config.inbox_dir / f"weekly-standup{JOB_SUFFIX}").write_text("{oops", encoding="utf-8")
    summarize = MagicMock(return_value="## สรุป")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        meeting_dir = process_file(audio_path, config)

    assert summarize.call_args.kwargs["model"] == config.claude_model
    assert (meeting_dir / "summary.md").exists()


def test_process_file_removes_the_job_file_when_it_succeeds(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(config.inbox_dir, "weekly-standup", "claude-sonnet-5")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    assert not (config.inbox_dir / f"weekly-standup{JOB_SUFFIX}").exists()


def test_process_file_sends_the_job_file_to_failed_when_summarizing_fails(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(config.inbox_dir, "weekly-standup", "claude-sonnet-5")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch("src.pipeline.summarize_transcript", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError),
    ):
        process_file(audio_path, config)

    assert not (config.inbox_dir / f"weekly-standup{JOB_SUFFIX}").exists()
    assert read_model(config.failed_dir / "weekly-standup.mp3") == "claude-sonnet-5"


def test_the_recorded_model_choice_survives_from_manifest_to_summary(tmp_path):
    # The feature's actual premise: the model the user picked at record time
    # (written into the session manifest) must survive session -> job sidecar ->
    # pipeline -> summarize_transcript call -> summary.md footer, with the sidecar
    # gone from inbox/ afterward. No hop in between (write_job, read_model,
    # finish_session) is mocked -- only the external boundaries are.
    config = make_config(tmp_path)
    inbox = config.inbox_dir
    inbox.mkdir(parents=True)

    session_dir = session_dir_for(inbox, "weekly-standup")
    session_dir.mkdir(parents=True)
    part_name = part_filename(1)
    (session_dir / part_name).write_bytes(_FAKE_WAV_BYTES)
    write_manifest(
        session_dir,
        "weekly-standup",
        "2026-07-24T14:30:05",
        48000,
        [part_name],
        "recording",
        claude_model="claude-sonnet-5",
    )

    def fake_ffmpeg_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with patch("src.segments.subprocess.run", side_effect=fake_ffmpeg_run):
        audio_path = finish_session(session_dir, inbox)

    summarize = MagicMock(return_value="## ประเด็นสำคัญ\n- ทดสอบ")

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
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        meeting_dir = process_file(audio_path, config)

    assert summarize.call_args.kwargs["model"] == "claude-sonnet-5"
    summary = (meeting_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.endswith("---\nสรุปด้วย claude-sonnet-5\n")
    assert not (inbox / f"weekly-standup{JOB_SUFFIX}").exists()


def _saved_transcript(config: Config, name: str, text: str) -> Path:
    """สภาพที่รอบก่อนทิ้งไว้: transcript เขียนครบแล้ว แต่ขั้นสรุปล้ม"""
    meeting_dir = config.meetings_dir / name
    meeting_dir.mkdir(parents=True)
    transcript_path = meeting_dir / "transcript.md"
    transcript_path.write_text(text, encoding="utf-8")
    return transcript_path


def test_process_file_reuses_the_transcript_saved_by_an_earlier_run(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    transcript_path = _saved_transcript(
        config, "2026-07-25_09-00-weekly-standup", "# Transcript\n\nของเดิม"
    )
    record_transcript(audio_path, transcript_path)

    with (
        _mock_convert_to_wav() as convert,
        patch("src.pipeline.transcribe_audio") as transcribe,
        patch("src.pipeline.diarize_audio") as diarize,
        patch(
            "src.pipeline.summarize_transcript", return_value="## ประเด็นสำคัญ\n- ใหม่"
        ) as summarize,
    ):
        meeting_dir = process_file(audio_path, config)

    # ถอดเสียงคือขั้นที่แพงที่สุดของ pipeline และผลลัพธ์ก็จะเหมือนเดิมเป๊ะ
    convert.assert_not_called()
    transcribe.assert_not_called()
    diarize.assert_not_called()
    assert meeting_dir == transcript_path.parent
    assert summarize.call_args.args[0] == "# Transcript\n\nของเดิม"
    assert (meeting_dir / "summary.md").exists()


def test_process_file_transcribes_again_when_the_saved_transcript_is_gone(tmp_path):
    # ผู้ใช้ลบหรือย้ายโฟลเดอร์ประชุมทิ้ง -- ต้องถอยกลับไปทำแบบเต็มไม่ใช่ล้ม
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    transcript_path = _saved_transcript(config, "2026-07-25_09-00-weekly-standup", "เดิม")
    record_transcript(audio_path, transcript_path)
    transcript_path.unlink()

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "ถอดใหม่"}],
        ) as transcribe,
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch("src.pipeline.summarize_transcript", return_value="## ประเด็นสำคัญ"),
    ):
        meeting_dir = process_file(audio_path, config)

    transcribe.assert_called_once()
    assert "ถอดใหม่" in (meeting_dir / "transcript.md").read_text(encoding="utf-8")


def test_process_file_records_the_transcript_path_for_a_later_retry(tmp_path):
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
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch(
            "src.pipeline.summarize_transcript",
            side_effect=RuntimeError("เครดิตไม่พอ"),
        ),
    ):
        with pytest.raises(RuntimeError):
            process_file(audio_path, config)

    # ตัวชี้เดินทางไปพร้อมไฟล์เสียง คนกู้จึงลากทั้งคู่กลับ inbox/ แล้วได้ของเดิมต่อ
    moved_audio = config.failed_dir / "weekly-standup.mp3"
    assert moved_audio.exists()
    assert read_transcript(moved_audio) == config.meetings_dir.joinpath(
        f"{date.today().isoformat()}_weekly-standup", "transcript.md"
    )


def test_process_file_skips_summarizing_in_transcript_only_mode(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(config.inbox_dir, "weekly-standup", NO_SUMMARY_MODEL)
    summarize = MagicMock(return_value="## สรุป")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        meeting_dir = process_file(audio_path, config)

    # ผู้ใช้เลือกโหมดนี้เพื่อไม่ให้เสียเงิน การเรียกแม้ครั้งเดียวคือการผิดสัญญานั้น
    summarize.assert_not_called()
    assert "สวัสดีครับ" in (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert not (meeting_dir / "summary.md").exists()
    # ที่เหลือของงานต้องจบเหมือนประชุมปกติ ไม่ใช่ค้างอยู่กลางทาง
    assert (meeting_dir / "weekly-standup.mp3").exists()
    assert not audio_path.exists()
    assert not (config.inbox_dir / f"weekly-standup{JOB_SUFFIX}").exists()


def test_process_file_skips_summarizing_when_reusing_a_saved_transcript(tmp_path):
    # เส้นทาง reuse ไม่ผ่าน process_file ท่อนบนเลย ถ้าเช็ค sentinel ไปวางผิดที่
    # ไฟล์ที่กลับมาจาก failed/ จะถูกสรุปทั้งที่ผู้ใช้สั่งว่าไม่ต้อง
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    # write_job ก่อน record_transcript: record_transcript อ่านของเดิมขึ้นมาเติม field
    # แต่ write_job เขียนทับทั้งไฟล์ สลับลำดับแล้ว transcript_path จะหายไป
    write_job(config.inbox_dir, "weekly-standup", NO_SUMMARY_MODEL)
    transcript_path = _saved_transcript(
        config, "2026-07-25_09-00-weekly-standup", "# Transcript\n\nของเดิม"
    )
    record_transcript(audio_path, transcript_path)
    summarize = MagicMock(return_value="## สรุป")

    with (
        _mock_convert_to_wav() as convert,
        patch("src.pipeline.transcribe_audio") as transcribe,
        patch("src.pipeline.diarize_audio"),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        meeting_dir = process_file(audio_path, config)

    summarize.assert_not_called()
    convert.assert_not_called()
    transcribe.assert_not_called()
    assert meeting_dir == transcript_path.parent
    assert not (meeting_dir / "summary.md").exists()
    assert (meeting_dir / "weekly-standup.mp3").exists()


def test_transcript_only_survives_from_the_manifest_to_the_meeting_folder(tmp_path):
    # ท่อทั้งสาย (write_manifest -> finish_session -> write_job -> read_model)
    # ไม่รู้จัก sentinel ตัวนี้เลย เทสต์นี้พิสูจน์ว่ามันไม่จำเป็นต้องรู้ -- ไม่มี
    # hop ไหนถูก mock มีแต่ ffmpeg กับโมเดลที่เป็นขอบนอกเท่านั้น
    config = make_config(tmp_path)
    inbox = config.inbox_dir
    inbox.mkdir(parents=True)

    session_dir = session_dir_for(inbox, "weekly-standup")
    session_dir.mkdir(parents=True)
    part_name = part_filename(1)
    (session_dir / part_name).write_bytes(_FAKE_WAV_BYTES)
    write_manifest(
        session_dir,
        "weekly-standup",
        "2026-07-26T14:30:05",
        48000,
        [part_name],
        "recording",
        claude_model=NO_SUMMARY_MODEL,
    )

    def fake_ffmpeg_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with patch("src.segments.subprocess.run", side_effect=fake_ffmpeg_run):
        audio_path = finish_session(session_dir, inbox)

    summarize = MagicMock(return_value="## สรุป")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=[]),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        meeting_dir = process_file(audio_path, config)

    summarize.assert_not_called()
    assert (meeting_dir / "transcript.md").exists()
    assert not (meeting_dir / "summary.md").exists()
    assert not (inbox / f"weekly-standup{JOB_SUFFIX}").exists()
