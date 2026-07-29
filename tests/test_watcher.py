import json
from unittest.mock import patch

from src import enroll
from src.config import Config
from src.watcher import is_file_stable, scan_inbox, watch_loop


def make_config(tmp_path) -> Config:
    return Config(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
        hf_token="hf-test-token",
    )


def test_scan_inbox_returns_only_audio_files_sorted(tmp_path):
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    (inbox_dir / "b.mp3").write_bytes(b"x")
    (inbox_dir / "a.wav").write_bytes(b"x")
    (inbox_dir / "notes.txt").write_bytes(b"x")

    result = scan_inbox(inbox_dir)

    assert result == [inbox_dir / "a.wav", inbox_dir / "b.mp3"]


def test_scan_inbox_returns_empty_list_when_dir_missing(tmp_path):
    assert scan_inbox(tmp_path / "does-not-exist") == []


def test_scan_inbox_ignores_session_directories_and_accepts_ogg(tmp_path):
    session = tmp_path / ".session-meet1"
    session.mkdir()
    (session / "part0001.wav").write_bytes(b"x")
    (tmp_path / "done.ogg").write_bytes(b"x")

    assert scan_inbox(tmp_path) == [tmp_path / "done.ogg"]


def test_is_file_stable_true_for_unchanging_file(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with patch("time.sleep"):
        assert is_file_stable(audio_path, check_interval=0) is True


def test_is_file_stable_false_for_empty_file(tmp_path):
    audio_path = tmp_path / "empty.mp3"
    audio_path.write_bytes(b"")

    with patch("time.sleep"):
        assert is_file_stable(audio_path, check_interval=0) is False


def test_watch_loop_processes_stable_files_once_with_single_pass(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=None, whisper_model=None
    )


def test_watch_loop_threads_diarization_pipeline_to_process_file(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    sentinel_pipeline = object()

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True, diarization_pipeline=sentinel_pipeline)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=sentinel_pipeline, whisper_model=None
    )


def test_watch_loop_threads_whisper_model_to_process_file(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    sentinel_model = object()

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True, whisper_model=sentinel_model)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=None, whisper_model=sentinel_model
    )


def test_watch_loop_skips_unstable_files(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with (
        patch("src.watcher.is_file_stable", return_value=False),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True)

    mock_process_file.assert_not_called()


def make_enroll_audio(tmp_path, name="สมชาย.ogg"):
    directory = tmp_path / "enroll"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"fake audio")
    return path


def test_watch_loop_analyzes_a_requested_enrollment_clip(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    sentinel_pipeline = object()

    analyzed = {"status": "ok", "embedding": [0.5], "speaker_count": 1}
    with patch("src.watcher.enroll.analyze", return_value=analyzed) as mock_analyze:
        watch_loop(
            config, single_pass=True, diarization_pipeline=sentinel_pipeline
        )

    assert mock_analyze.call_count == 1
    assert mock_analyze.call_args.kwargs["pipeline"] is sentinel_pipeline
    written = json.loads(
        (tmp_path / "enroll" / "สมชาย.ogg.result.json").read_text(encoding="utf-8")
    )
    assert written["status"] == "ok"
    assert written["audio_file"] == "สมชาย.ogg"


def test_watch_loop_does_not_analyze_a_clip_nobody_requested(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    make_enroll_audio(tmp_path)

    with patch("src.watcher.enroll.analyze") as mock_analyze:
        watch_loop(config, single_pass=True)

    mock_analyze.assert_not_called()


def test_watch_loop_never_writes_the_speaker_registry(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")

    with patch(
        "src.watcher.enroll.analyze",
        return_value={"status": "ok", "embedding": [0.5]},
    ):
        watch_loop(config, single_pass=True)

    # ทะเบียนถูกเขียนโดย session_service เมื่อมีคนกดยืนยันเท่านั้น การจับคู่ที่ผิด
    # ต้องไม่ฝังตัวอย่างเสียงผิดคนลงโปรไฟล์ถาวรโดยไม่มีมนุษย์เห็นเลยสักครั้ง
    assert not (tmp_path / "speakers" / "registry.json").exists()


def test_a_failing_enroll_job_does_not_stop_the_inbox_from_being_processed(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "meeting.ogg"
    audio_path.write_bytes(b"fake audio data")
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
        patch("src.watcher.enroll.analyze", side_effect=OSError("disk on fire")),
    ):
        watch_loop(config, single_pass=True)

    # ประชุมที่อัดซ้ำไม่ได้ต้องไม่โดนงาน enroll ที่ทำใหม่ได้เสมอทำให้พัง
    mock_process_file.assert_called_once()


def test_the_inbox_is_processed_before_any_enroll_work(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    (config.inbox_dir / "meeting.ogg").write_bytes(b"fake audio data")
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    order = []

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file", side_effect=lambda *a, **k: order.append("inbox")),
        patch(
            "src.watcher.enroll.analyze",
            side_effect=lambda *a, **k: order.append("enroll") or {"status": "ok"},
        ),
    ):
        watch_loop(config, single_pass=True)

    # ลำดับนี้ load-bearing: การประชุมที่อัดซ้ำไม่ได้ต้องได้ GPU ก่อนงานที่ทำใหม่ได้
    assert order == ["inbox", "enroll"]
