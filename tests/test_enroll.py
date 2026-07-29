from pathlib import Path

from src import enroll
from src.diarize import DiarizationResult


def test_enroll_dir_and_done_dir_are_derived_from_base_dir(tmp_path):
    assert enroll.enroll_dir(tmp_path) == tmp_path / "enroll"
    assert enroll.done_dir(tmp_path) == tmp_path / "enroll" / "done"


def test_scan_audio_returns_only_audio_files_sorted(tmp_path):
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "b.wav").write_bytes(b"x")
    (directory / "a.ogg").write_bytes(b"x")
    (directory / "a.request.json").write_text("{}", encoding="utf-8")
    (directory / "notes.txt").write_bytes(b"x")

    assert enroll.scan_audio(tmp_path) == [directory / "a.ogg", directory / "b.wav"]


def test_scan_audio_ignores_the_done_subfolder(tmp_path):
    directory = tmp_path / "enroll"
    (directory / "done").mkdir(parents=True)
    (directory / "done" / "archived.ogg").write_bytes(b"x")
    (directory / "live.ogg").write_bytes(b"x")

    assert enroll.scan_audio(tmp_path) == [directory / "live.ogg"]


def test_scan_audio_returns_empty_list_when_dir_missing(tmp_path):
    assert enroll.scan_audio(tmp_path) == []


def test_is_safe_filename_rejects_paths_that_escape_the_folder():
    assert enroll.is_safe_filename("สมชาย.ogg") is True
    assert enroll.is_safe_filename("../../evil.ogg") is False
    assert enroll.is_safe_filename("sub/dir.ogg") is False
    assert enroll.is_safe_filename("C:\\Windows\\evil.ogg") is False
    assert enroll.is_safe_filename("") is False
    assert enroll.is_safe_filename(".") is False
    assert enroll.is_safe_filename("..") is False
    assert enroll.is_safe_filename(None) is False


def test_suggested_name_strips_extension_and_markdown_characters():
    assert enroll.suggested_name_from("สมชาย.ogg") == "สมชาย"
    assert enroll.suggested_name_from("พี่ *เอ* [1].wav") == "พี่ เอ 1"
    assert enroll.suggested_name_from("  a   b .m4a") == "a b"


def test_suggested_name_is_empty_when_nothing_usable_remains():
    assert enroll.suggested_name_from("***.ogg") == ""


def fake_diarize(result=None, error=None):
    """แทนที่ src.enroll.diarize_audio ด้วยฟังก์ชันที่คืนผลที่เราคุมได้"""

    def _call(audio_path, hf_token, pipeline):
        if error is not None:
            raise error
        return result

    return _call


def test_analyze_accepts_a_single_speaker_who_talks_long_enough(tmp_path, monkeypatch):
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    result = DiarizationResult(
        turns=[
            {"start": 0.0, "end": 40.0, "speaker": "SPEAKER_00"},
            {"start": 45.0, "end": 73.5, "speaker": "SPEAKER_00"},
        ],
        embeddings={"SPEAKER_00": [0.1, 0.2, 0.3]},
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    analyzed = enroll.analyze(audio_path, pipeline=object())

    assert analyzed["status"] == "ok"
    assert analyzed["speaker_count"] == 1
    assert analyzed["speaking_seconds"] == 68.5
    assert analyzed["suggested_name"] == "สมชาย"
    assert analyzed["embedding"] == [0.1, 0.2, 0.3]
    assert "reason" not in analyzed


def test_analyze_rejects_a_file_with_more_than_one_speaker(tmp_path, monkeypatch):
    audio_path = tmp_path / "ประชุมย่อย.m4a"
    audio_path.write_bytes(b"x")
    result = DiarizationResult(
        turns=[
            {"start": 0.0, "end": 60.0, "speaker": "SPEAKER_00"},
            {"start": 60.0, "end": 140.0, "speaker": "SPEAKER_01"},
        ],
        embeddings={"SPEAKER_00": [0.1, 0.2], "SPEAKER_01": [0.3, 0.4]},
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    analyzed = enroll.analyze(audio_path, pipeline=object())

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "multiple_speakers"
    assert analyzed["speaker_count"] == 2
    # ตัวเลขต้องยังอยู่ให้หน้าเว็บอธิบายเหตุผลได้
    assert analyzed["speaking_seconds"] == 140.0
    # เวกเตอร์ที่ไม่ควรถูกใช้ ต้องไม่มีอยู่ในไฟล์ให้ใครหยิบไปใช้ผิด
    assert "embedding" not in analyzed


def test_analyze_rejects_a_long_file_holding_only_a_few_seconds_of_speech(
    tmp_path, monkeypatch
):
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"x")
    # ไฟล์ยาว 5 นาที แต่มีเสียงพูดจริง 6 วินาที -- ต้องดูเวลาพูด ไม่ใช่ความยาวไฟล์
    result = DiarizationResult(
        turns=[{"start": 200.0, "end": 206.0, "speaker": "SPEAKER_00"}],
        embeddings={"SPEAKER_00": [0.1, 0.2]},
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    analyzed = enroll.analyze(audio_path, pipeline=object())

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "too_short"
    assert analyzed["speaking_seconds"] == 6.0
    assert "embedding" not in analyzed


def test_analyze_rejects_a_file_with_no_speech_at_all(tmp_path, monkeypatch):
    audio_path = tmp_path / "silence.wav"
    audio_path.write_bytes(b"x")
    monkeypatch.setattr(
        "src.enroll.diarize_audio",
        fake_diarize(DiarizationResult(turns=[], embeddings={})),
    )

    analyzed = enroll.analyze(audio_path, pipeline=object())

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "too_short"
    assert analyzed["speaker_count"] == 0


def test_analyze_rejects_a_zero_vector_pyannote_padded_in(tmp_path, monkeypatch):
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    # pyannote pad ศูนย์เข้ามาเมื่อจำนวน label มากกว่าจำนวน centroid -- เวกเตอร์ศูนย์
    # "เหมือน" กับเวกเตอร์ศูนย์อื่นทุกตัว ปล่อยเข้าทะเบียนไม่ได้
    result = DiarizationResult(
        turns=[{"start": 0.0, "end": 40.0, "speaker": "SPEAKER_00"}],
        embeddings={"SPEAKER_00": [0.0, 0.0, 0.0]},
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    analyzed = enroll.analyze(audio_path, pipeline=object())

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "unusable_embedding"
    assert "embedding" not in analyzed


def test_analyze_rejects_when_the_label_has_no_embedding_at_all(tmp_path, monkeypatch):
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    # diarize กลืน exception ตอนอ่าน speaker_embeddings แล้วคืน {} ได้ (ดู
    # diarize._speaker_embeddings) turns ยังมาครบ แต่ไม่มีเวกเตอร์ให้เก็บ
    result = DiarizationResult(
        turns=[{"start": 0.0, "end": 40.0, "speaker": "SPEAKER_00"}],
        embeddings={},
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    analyzed = enroll.analyze(audio_path, pipeline=object())

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "unusable_embedding"


def test_analyze_turns_a_pipeline_crash_into_a_rejected_result(tmp_path, monkeypatch):
    audio_path = tmp_path / "broken.ogg"
    audio_path.write_bytes(b"x")
    monkeypatch.setattr(
        "src.enroll.diarize_audio", fake_diarize(error=RuntimeError("cuda oom"))
    )

    analyzed = enroll.analyze(audio_path, pipeline=object())

    # เงียบไปเฉย ๆ = หน้าจอค้างที่ "กำลังวิเคราะห์" ตลอดกาล ต้องได้ผลกลับมาเสมอ
    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "analysis_failed"
    assert "cuda oom" in analyzed["detail"]
    assert "embedding" not in analyzed
    assert "speaker_count" not in analyzed


def test_analyze_passes_the_pipeline_through_without_loading_one(tmp_path, monkeypatch):
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    sentinel = object()
    seen = {}

    def _spy(path, hf_token, pipeline):
        seen["pipeline"] = pipeline
        return DiarizationResult(
            turns=[{"start": 0.0, "end": 40.0, "speaker": "SPEAKER_00"}],
            embeddings={"SPEAKER_00": [0.5]},
        )

    monkeypatch.setattr("src.enroll.diarize_audio", _spy)

    enroll.analyze(audio_path, pipeline=sentinel)

    assert seen["pipeline"] is sentinel


import json
from datetime import datetime


def make_audio(tmp_path, name="สมชาย.ogg"):
    directory = tmp_path / "enroll"
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_bytes(b"fake audio")
    return path


def test_write_request_creates_a_sidecar_next_to_the_audio(tmp_path):
    make_audio(tmp_path)

    path = enroll.write_request(
        tmp_path, "สมชาย.ogg", now=datetime(2026, 7, 29, 10, 31, 0)
    )

    assert path == tmp_path / "enroll" / "สมชาย.request.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["audio_file"] == "สมชาย.ogg"
    assert payload["requested"] == "2026-07-29T10:31:00"


def test_write_request_refuses_a_filename_that_escapes_the_folder(tmp_path):
    (tmp_path / "enroll").mkdir()

    assert enroll.write_request(tmp_path, "../../evil.ogg") is None
    assert not (tmp_path / "evil.request.json").exists()


def test_write_request_refuses_a_file_that_is_not_there(tmp_path):
    (tmp_path / "enroll").mkdir()

    assert enroll.write_request(tmp_path, "ghost.ogg") is None


def test_pending_requests_lists_requests_that_have_no_result_yet(tmp_path):
    make_audio(tmp_path, "a.ogg")
    make_audio(tmp_path, "b.ogg")
    make_audio(tmp_path, "c.ogg")
    enroll.write_request(tmp_path, "a.ogg")
    enroll.write_request(tmp_path, "b.ogg")
    enroll.write_result(tmp_path, "b.ogg", {"status": "ok"})

    # a สั่งแล้วยังไม่มีผล, b มีผลแล้ว, c ยังไม่ได้สั่ง
    assert enroll.pending_requests(tmp_path) == ["a.ogg"]


def test_pending_requests_is_empty_when_the_folder_does_not_exist(tmp_path):
    assert enroll.pending_requests(tmp_path) == []


def test_write_result_stamps_the_file_and_keeps_the_embedding(tmp_path):
    make_audio(tmp_path)

    path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2], "speaker_count": 1},
        now=datetime(2026, 7, 29, 10, 33, 12),
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["audio_file"] == "สมชาย.ogg"
    assert payload["analyzed"] == "2026-07-29T10:33:12"
    assert payload["embedding"] == [0.1, 0.2]


def test_read_result_returns_none_for_a_corrupt_file(tmp_path):
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "สมชาย.result.json").write_text("{ not json", encoding="utf-8")

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None


def test_read_result_returns_none_when_there_is_no_result(tmp_path):
    (tmp_path / "enroll").mkdir()

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None


def test_clear_removes_both_sidecars_and_leaves_the_audio(tmp_path):
    audio_path = make_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok"})

    enroll.clear(tmp_path, "สมชาย.ogg")

    assert not (tmp_path / "enroll" / "สมชาย.request.json").exists()
    assert not (tmp_path / "enroll" / "สมชาย.result.json").exists()
    assert audio_path.exists()


def test_clear_is_silent_when_there_is_nothing_to_remove(tmp_path):
    (tmp_path / "enroll").mkdir()

    enroll.clear(tmp_path, "สมชาย.ogg")  # ต้องไม่ raise


def test_archive_moves_the_audio_into_done_and_clears_sidecars(tmp_path):
    make_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok"})

    destination = enroll.archive(tmp_path, "สมชาย.ogg")

    assert destination == tmp_path / "enroll" / "done" / "สมชาย.ogg"
    assert destination.is_file()
    assert not (tmp_path / "enroll" / "สมชาย.ogg").exists()
    assert not (tmp_path / "enroll" / "สมชาย.request.json").exists()
    assert enroll.scan_audio(tmp_path) == []


def test_archive_never_overwrites_an_earlier_file_of_the_same_name(tmp_path):
    done = tmp_path / "enroll" / "done"
    done.mkdir(parents=True)
    (done / "สมชาย.ogg").write_bytes(b"the first one")
    make_audio(tmp_path)

    destination = enroll.archive(tmp_path, "สมชาย.ogg")

    assert destination == done / "สมชาย-2.ogg"
    assert (done / "สมชาย.ogg").read_bytes() == b"the first one"


def test_archive_refuses_a_filename_that_escapes_the_folder(tmp_path):
    (tmp_path / "enroll").mkdir()

    assert enroll.archive(tmp_path, "../../evil.ogg") is None


def test_list_entries_reports_status_per_file_without_leaking_vectors(tmp_path):
    make_audio(tmp_path, "waiting.ogg")
    make_audio(tmp_path, "queued.ogg")
    make_audio(tmp_path, "ready.ogg")
    enroll.write_request(tmp_path, "queued.ogg")
    enroll.write_request(tmp_path, "ready.ogg")
    enroll.write_result(
        tmp_path,
        "ready.ogg",
        {
            "status": "ok",
            "embedding": [0.1, 0.2],
            "speaker_count": 1,
            "speaking_seconds": 68.5,
            "suggested_name": "ready",
        },
    )

    entries = {entry["audio_file"]: entry for entry in enroll.list_entries(tmp_path)}

    assert entries["waiting.ogg"]["state"] == "idle"
    assert entries["queued.ogg"]["state"] == "queued"
    assert entries["ready.ogg"]["state"] == "done"
    assert entries["ready.ogg"]["status"] == "ok"
    assert entries["ready.ogg"]["speaking_seconds"] == 68.5
    assert entries["ready.ogg"]["suggested_name"] == "ready"
    # เวกเตอร์เป็นข้อมูล biometric และหน้าเว็บไม่ได้ใช้ -- ห้ามหลุดออกไป
    assert all("embedding" not in entry for entry in entries.values())


def test_list_entries_suggests_a_name_even_before_analysis(tmp_path):
    make_audio(tmp_path, "สมหญิง.wav")

    entry = enroll.list_entries(tmp_path)[0]

    assert entry["suggested_name"] == "สมหญิง"
    assert entry["size_bytes"] == len(b"fake audio")
