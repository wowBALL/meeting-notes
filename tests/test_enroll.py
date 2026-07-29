import json
import os
from datetime import datetime
from pathlib import Path

import pytest

from src import enroll
from src.diarize import DiarizationResult
from src.speakers import MIN_SPEAKING_SECONDS


@pytest.fixture(autouse=True)
def _stub_convert_to_wav(monkeypatch):
    """analyze() ผ่าน convert_to_wav (ffmpeg จริง) เสมอตอนนี้ (finding 4) -- เทสต์ในไฟล์นี้
    ต้องไม่พึ่ง ffmpeg ตัวจริงหรือไฟล์เสียงจริง จึง stub เป็นค่าเริ่มต้นของทุกเทสต์
    เทสต์ที่ตั้งใจทดสอบ path การแปลงเองจะ monkeypatch ทับอีกทีในตัวเทสต์นั้น ๆ
    """
    monkeypatch.setattr("src.enroll.convert_to_wav", lambda src, dst: dst)


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


def test_scan_audio_accepts_an_uppercase_extension(tmp_path):
    # ผู้ใช้ลากไฟล์จากมือถือ/กล้องมาบ่อย ๆ นามสกุลตัวใหญ่ทั้งดุ้นเจอได้ทั่วไป -- .lower()
    # ต้องทำงานจริง ไม่ใช่แค่มีอยู่ในโค้ดเฉย ๆ
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "A.WAV").write_bytes(b"x")

    assert enroll.scan_audio(tmp_path) == [directory / "A.WAV"]


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


def test_analyze_accepts_speech_at_exactly_the_minimum_boundary(tmp_path, monkeypatch):
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    # เงื่อนไขจริงคือ speaking_seconds < MIN_SPEAKING_SECONDS -- เท่ากับพอดีต้องผ่าน
    # ไม่ใช่ถูกปฏิเสธ จึงล็อกค่านี้ไว้แทนการฝัง 10.0 ตรง ๆ เผื่อค่าคงที่เปลี่ยนในอนาคต
    result = DiarizationResult(
        turns=[{"start": 0.0, "end": MIN_SPEAKING_SECONDS, "speaker": "SPEAKER_00"}],
        embeddings={"SPEAKER_00": [0.1, 0.2]},
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    analyzed = enroll.analyze(audio_path, pipeline=object())

    assert analyzed["status"] == "ok"
    assert analyzed["speaking_seconds"] == MIN_SPEAKING_SECONDS


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


def test_analyze_converts_to_wav_before_diarizing(tmp_path, monkeypatch):
    """finding 4: enroll ต้อง normalize เสียงด้วย convert_to_wav แบบเดียวกับ
    pipeline.process_file ก่อนส่งเข้า diarize_audio -- ไม่งั้น pyannote ได้ container ดิบ
    ที่ decode ด้วย backend ที่เดาไม่ได้ (soundfile ถอด AAC ไม่ได้) แทนที่จะผ่าน ffmpeg
    ซึ่งเป็นตัวถอดรหัสเดียวที่โปรเจกต์นี้รับประกัน
    """
    audio_path = tmp_path / "สมชาย.m4a"
    audio_path.write_bytes(b"x")
    seen = {}

    def fake_convert(src, dst):
        seen["src"] = src
        seen["dst"] = dst
        return dst

    def fake_diarize(path, hf_token, pipeline):
        seen["diarized_path"] = path
        return DiarizationResult(
            turns=[{"start": 0.0, "end": 40.0, "speaker": "SPEAKER_00"}],
            embeddings={"SPEAKER_00": [0.1]},
        )

    monkeypatch.setattr("src.enroll.convert_to_wav", fake_convert)
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize)

    enroll.analyze(audio_path, pipeline=object())

    assert seen["src"] == audio_path
    assert seen["dst"].suffix == ".wav"
    # diarize ต้องได้ไฟล์ wav ที่แปลงแล้ว ไม่ใช่ไฟล์ .m4a ดิบตัวเดิม
    assert seen["diarized_path"] == seen["dst"]


def test_analyze_turns_a_conversion_failure_into_a_rejected_result(tmp_path, monkeypatch):
    """finding 4: แปลงเป็น wav ไม่สำเร็จ (เช่น ffmpeg ถอดรหัส .m4a ไม่ได้) ต้องไม่ raise
    ออกจาก analyze() -- ต้องได้ rejected/analysis_failed เหมือนตอน diarize ล้มเหลว
    และ diarize ต้องไม่ถูกเรียกเลยเพราะไม่มีไฟล์ wav ให้ป้อน
    """
    audio_path = tmp_path / "broken.m4a"
    audio_path.write_bytes(b"x")
    diarize_calls = []

    def fail_convert(src, dst):
        raise RuntimeError("ffmpeg exited with code 1")

    monkeypatch.setattr("src.enroll.convert_to_wav", fail_convert)
    monkeypatch.setattr(
        "src.enroll.diarize_audio", lambda *a, **k: diarize_calls.append(1)
    )

    analyzed = enroll.analyze(audio_path, pipeline=object())

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "analysis_failed"
    assert "ffmpeg exited with code 1" in analyzed["detail"]
    assert "embedding" not in analyzed
    assert diarize_calls == []


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

    assert path == tmp_path / "enroll" / "สมชาย.ogg.request.json"
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


def test_write_result_records_the_audio_files_size_and_mtime(tmp_path):
    audio_path = make_audio(tmp_path)
    stat = audio_path.stat()

    path = enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["audio_size"] == stat.st_size
    assert payload["audio_mtime"] == pytest.approx(stat.st_mtime)


def test_read_result_returns_none_when_the_audio_file_was_replaced_after_analysis(tmp_path):
    """finding 1 (critical): ผู้ใช้ลบไฟล์เดิมทิ้งแล้ววางไฟล์ใหม่ชื่อเดียวกัน (เนื้อหา/ขนาด
    ต่างกัน) -- ผลวิเคราะห์เก่าต้องไม่ผูกเข้ากับไฟล์ใหม่ ไม่งั้นเวกเตอร์เสียงของคนเดิม
    จะถูกเสนอให้ยืนยันภายใต้ชื่อคนใหม่ และหลุดเข้าทะเบียนถาวรถ้ากดบันทึก
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1, 0.2]})

    audio_path.unlink()
    audio_path.write_bytes(b"a completely different recording body, much longer than before")

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None
    # ผลเก่าที่ผูกกับไฟล์เดิมต้องถูกล้างทิ้งไปด้วย ไม่งั้นการ์ดจะยังค้างสถานะ "done" ผิด ๆ
    assert not (tmp_path / "enroll" / "สมชาย.ogg.result.json").exists()
    assert not (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()


def test_read_result_tolerates_the_small_mtime_drift_a_file_copy_produces(tmp_path):
    """คัดลอกไฟล์ (เช่นย้ายด้วยเครื่องมือบางตัว) ขยับ mtime ได้เล็กน้อยโดยเนื้อหาเดิมทุก
    ไบต์ -- เทียบแบบ exact equality จะปฏิเสธไฟล์ที่ไม่ได้เปลี่ยนอะไรเลยผิด ๆ
    """
    audio_path = make_audio(tmp_path)
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
    stat = audio_path.stat()
    os.utime(audio_path, (stat.st_atime, stat.st_mtime + 1.0))

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is not None


def test_read_result_rejects_an_mtime_drift_beyond_the_tolerance(tmp_path):
    audio_path = make_audio(tmp_path)
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
    stat = audio_path.stat()
    os.utime(audio_path, (stat.st_atime, stat.st_mtime + 30.0))

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None


def test_list_entries_sweeps_orphaned_sidecars_whose_audio_is_gone(tmp_path):
    """finding 1: sidecar ที่ไฟล์เสียงต้นทางหายไปแล้ว (ผู้ใช้ลบผ่าน Explorer) ต้องถูกกวาด
    ทิ้งไม่ให้ค้างบนดิสก์ พร้อมสำหรับผูกผิดกับไฟล์ใหม่ชื่อเดียวกันที่วางเข้ามาทีหลัง
    """
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "หาย.ogg.request.json").write_text("{}", encoding="utf-8")
    (directory / "หาย.ogg.result.json").write_text("{}", encoding="utf-8")

    enroll.list_entries(tmp_path)

    assert not (directory / "หาย.ogg.request.json").exists()
    assert not (directory / "หาย.ogg.result.json").exists()


def test_list_entries_orphan_sweep_does_not_raise_when_deletion_fails(tmp_path, monkeypatch):
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "หาย.ogg.result.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(OSError("locked"))
    )

    enroll.list_entries(tmp_path)  # ต้องไม่ raise


def test_list_entries_skips_a_row_whose_file_was_archived_mid_request(tmp_path):
    """finding 5: path.stat() raise FileNotFoundError ได้ถ้าอีก request หนึ่ง archive
    ไฟล์นี้ไปพอดีระหว่างที่กำลังแจกแจงรายการ -- ต้องข้ามแถวนั้น ไม่ใช่ 500 ทั้งหน้า
    """
    make_audio(tmp_path, "ok.ogg")
    make_audio(tmp_path, "vanishes.ogg")
    real_stat = Path.stat
    calls = {"vanishes.ogg": 0}

    def flaky_stat(self, *args, **kwargs):
        # เรียกครั้งแรกคือตอน scan_audio กรองว่าเป็นไฟล์จริง (ยังอยู่) -- ต้อง
        # สำเร็จ ไฟล์นี้ค่อย "หาย" (ถูก archive โดย request อื่น) ก่อนถึง .stat()
        # รอบสองที่ list_entries เรียกเพื่อเอา size_bytes
        if self.name == "vanishes.ogg":
            calls["vanishes.ogg"] += 1
            if calls["vanishes.ogg"] > 1:
                raise FileNotFoundError(self)
        return real_stat(self, *args, **kwargs)

    import unittest.mock as mock

    with mock.patch.object(Path, "stat", flaky_stat):
        entries = enroll.list_entries(tmp_path)

    names = [entry["audio_file"] for entry in entries]
    assert names == ["ok.ogg"]


def test_write_result_downgrades_to_rejected_when_the_file_was_replaced_mid_analysis(
    tmp_path,
):
    """CRITICAL A: analyze() อ่านไบต์ ณ เวลา T1 แต่ write_result เดิม stat() ไฟล์ตอน T3
    (นาทีให้หลัง) เท่านั้น -- ระหว่างนั้นผู้ใช้แทนที่ enroll/สมชาย.ogg ด้วยการอัดคนละคน
    ผลที่ได้ต้องไม่ถูกผูกเข้ากับไฟล์ใหม่แล้วรับรองว่า "ok" เพราะ embedding เป็นของไบต์
    ชุดเดิมที่ไม่มีอยู่บนดิสก์แล้ว ผู้เรียก (watcher) ต้อง stat ไฟล์ก่อนส่งเข้า analyze()
    แล้วส่ง (size, mtime) นั้นมาที่นี่เป็น pre_analysis_stat
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    stat_before_analysis = audio_path.stat()
    pre_analysis_stat = (stat_before_analysis.st_size, stat_before_analysis.st_mtime)

    # จำลองผู้ใช้แทนที่ไฟล์ระหว่าง diarization กำลังทำงานอยู่ (เนื้อหา/ขนาดต่างไปจากเดิม)
    audio_path.unlink()
    audio_path.write_bytes(b"a completely different recording, replaced mid-analysis")

    path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2], "suggested_name": "สมชาย"},
        pre_analysis_stat=pre_analysis_stat,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["reason"] == "analysis_failed"
    assert "embedding" not in payload


def test_write_result_keeps_ok_when_the_file_matches_the_pre_analysis_stat(tmp_path):
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    stat_before_analysis = audio_path.stat()
    pre_analysis_stat = (stat_before_analysis.st_size, stat_before_analysis.st_mtime)

    path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2], "suggested_name": "สมชาย"},
        pre_analysis_stat=pre_analysis_stat,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "ok"
    assert payload["embedding"] == [0.1, 0.2]


def test_write_result_downgrades_to_rejected_when_it_cannot_stat_the_file_at_write_time(
    tmp_path,
):
    """CRITICAL B: เดิม write_result เขียน audio_size/audio_mtime เป็น None เมื่อ stat
    ไม่สำเร็จ แล้วปล่อย "ok" หลุดออกไปพร้อม embedding -- ต้องลดสถานะเป็น rejected แทน
    เพราะไม่มีทางผูกผลนี้กับไฟล์อะไรได้เลย (เช่น ผู้ใช้กด "เอาออกจากรายการ" ระหว่างที่
    watcher กำลังวิเคราะห์อยู่พอดี ไฟล์ถูกย้ายเข้า done/ ไปแล้วตอนเขียนผล)
    """
    (tmp_path / "enroll").mkdir()  # ไม่มีไฟล์เสียงอยู่จริงตอนเขียนผล

    path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2], "suggested_name": "สมชาย"},
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "rejected"
    assert payload["reason"] == "analysis_failed"
    assert "embedding" not in payload
    assert payload["audio_size"] is None
    assert payload["audio_mtime"] is None


def test_read_result_treats_a_null_binding_as_unverifiable_not_as_a_match(tmp_path):
    """CRITICAL B: การไม่มี binding (audio_size/audio_mtime เป็น null) ต้องแปลว่า
    "ยืนยันไม่ได้" ไม่ใช่ "ผ่านการเช็ค" -- ไม่งั้น sidecar ที่หลงเหลือจากบั๊กเดิม (หรือไฟล์
    ที่ถูกแก้มือ) จะเสนอ embedding ของคนหนึ่งให้ยืนยันภายใต้ไฟล์เสียงของอีกคนโดยไม่มีการ
    เช็คใด ๆ เลย
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    result_file = audio_path.with_name(audio_path.name + ".result.json")
    result_file.write_text(
        json.dumps(
            {
                "audio_file": "สมชาย.ogg",
                "analyzed": "2026-07-29T10:00:00",
                "audio_size": None,
                "audio_mtime": None,
                "status": "ok",
                "embedding": [0.9, 0.9],
                "suggested_name": "สมชาย",
            }
        ),
        encoding="utf-8",
    )

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None
    # ต้องเก็บกวาด sidecar ที่ยืนยันไม่ได้ทิ้งไปเลย ไม่ปล่อยค้างให้ผูกผิดซ้ำได้อีก
    assert not result_file.exists()


def test_read_result_returns_none_for_a_corrupt_file(tmp_path):
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "สมชาย.ogg.result.json").write_text("{ not json", encoding="utf-8")

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None


def test_read_result_returns_none_when_there_is_no_result(tmp_path):
    (tmp_path / "enroll").mkdir()

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None


def test_read_result_keeps_the_sidecar_on_a_transient_permission_error(tmp_path, monkeypatch):
    """Minor C: PermissionError ชั่วคราวจากโปรแกรมสแกนไวรัส/ตัวซิงก์ไฟล์ (ปัญหาที่โปรเจกต์
    นี้ยอมรับอยู่แล้ว ดู storage.replace_with_retry) ต้องไม่ถูกตีความว่าไฟล์เสียงหายไปแล้ว
    -- ไม่งั้นผลวิเคราะห์ GPU ที่เสร็จสมบูรณ์แล้วจะถูกลบทิ้งฟรี ๆ โดยไม่มีอะไรบอกผู้ใช้เลย
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
    result_file = audio_path.with_name(audio_path.name + ".result.json")
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self == audio_path:
            raise PermissionError("locked by antivirus")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None
    # ไม่รู้แน่ชัดว่าไฟล์หายจริงหรือแค่ล็อกชั่วคราว -- ต้องเก็บ sidecar ไว้ก่อน ไม่ล้างทิ้ง
    assert result_file.exists()


def test_read_result_clears_the_sidecar_when_the_file_is_genuinely_gone(tmp_path):
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
    result_file = audio_path.with_name(audio_path.name + ".result.json")

    audio_path.unlink()

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None
    assert not result_file.exists()


def test_sweep_orphan_sidecars_keeps_the_sidecar_on_a_transient_permission_error(
    tmp_path, monkeypatch
):
    """Minor C: _sweep_orphan_sidecars ใช้ Path.is_file() ซึ่งกลืน OSError ใด ๆ แล้วคืน
    False เงียบ ๆ -- PermissionError ชั่วคราวจึงถูกตีความว่า "ไฟล์เสียงไม่มีอยู่" ผิด ๆ
    แล้วลบ sidecar ของผลวิเคราะห์ที่เสร็จสมบูรณ์แล้วทิ้งไปฟรี ๆ
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
    result_file = audio_path.with_name(audio_path.name + ".result.json")
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self == audio_path:
            raise PermissionError("locked by antivirus")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    enroll.list_entries(tmp_path)  # เรียก _sweep_orphan_sidecars ภายใน

    assert result_file.exists()


def test_clear_removes_both_sidecars_and_leaves_the_audio(tmp_path):
    audio_path = make_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    enroll.write_result(tmp_path, "สมชาย.ogg", {"status": "ok"})

    enroll.clear(tmp_path, "สมชาย.ogg")

    assert not (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()
    assert not (tmp_path / "enroll" / "สมชาย.ogg.result.json").exists()
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
    assert not (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()
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


def test_sidecars_stay_independent_for_files_sharing_a_stem(tmp_path):
    # call.wav กับ call.mp3 มี stem เดียวกัน ถ้า sidecar ตัดนามสกุลทิ้ง ทั้งคู่จะชน
    # กันที่ call.request.json / call.result.json แล้วผลของไฟล์หนึ่งจะไปทับอีกไฟล์
    # -- คนหน้าเว็บจะเห็นเสียงของคนผิดถูกเสนอให้ยืนยันภายใต้ชื่อไฟล์อื่น
    make_audio(tmp_path, "call.wav")
    make_audio(tmp_path, "call.mp3")

    enroll.write_request(tmp_path, "call.wav")
    enroll.write_request(tmp_path, "call.mp3")
    enroll.write_result(
        tmp_path,
        "call.wav",
        {"status": "ok", "speaker_count": 1, "suggested_name": "call-wav-person"},
    )
    enroll.write_result(
        tmp_path,
        "call.mp3",
        {"status": "ok", "speaker_count": 2, "suggested_name": "call-mp3-person"},
    )

    wav_result = enroll.read_result(tmp_path, "call.wav")
    mp3_result = enroll.read_result(tmp_path, "call.mp3")

    assert wav_result["speaker_count"] == 1
    assert wav_result["suggested_name"] == "call-wav-person"
    assert mp3_result["speaker_count"] == 2
    assert mp3_result["suggested_name"] == "call-mp3-person"

    entries = {entry["audio_file"]: entry for entry in enroll.list_entries(tmp_path)}
    assert entries["call.wav"]["speaker_count"] == 1
    assert entries["call.wav"]["suggested_name"] == "call-wav-person"
    assert entries["call.mp3"]["speaker_count"] == 2
    assert entries["call.mp3"]["suggested_name"] == "call-mp3-person"
