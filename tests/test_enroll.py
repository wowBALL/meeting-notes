import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from src import enroll
from src.diarize import DiarizationResult
from src.enroll import analyze
from src.speakers import MIN_SPEAKING_SECONDS
from src.voiceprint import Voiceprint

MODEL = "pyannote/speaker-diarization-community-1"


@pytest.fixture(autouse=True)
def _stub_convert_to_wav(monkeypatch):
    """analyze() ผ่าน convert_to_wav (ffmpeg จริง) เสมอตอนนี้ (finding 4) -- เทสต์ในไฟล์นี้
    ต้องไม่พึ่ง ffmpeg ตัวจริง จึง stub เป็นค่าเริ่มต้นของทุกเทสต์ เทสต์ที่ตั้งใจทดสอบ
    path การแปลงเองจะ monkeypatch ทับอีกทีในตัวเทสต์นั้น ๆ

    Task 10: เดิม stub นี้เป็น no-op ที่คืน `dst` เฉย ๆ โดยไม่เขียนไฟล์ที่ path นั้นจริง --
    ใช้ได้ตอนที่ analyze() รู้จักแค่ diarize_audio ที่เทสต์ mock ทับอยู่แล้วเสมอ (ไม่มีใคร
    อ่าน dst จริง) แต่ตอนนี้ analyze() เรียก extract_voiceprints (voiceprint.py) ซึ่งโหลด
    ไฟล์ที่ dst จริงด้วย soundfile (ผ่าน load_waveform) เมื่อไม่ได้ mock ทับ (ดูเทสต์ที่ใช้
    _write_enroll_wav + pipeline ปลอมแทน diarize_audio ปลอม) -- no-op เดิมจะทำให้ path
    นั้นไม่มีไฟล์อยู่เลย จึง stub เป็นการคัดลอกไฟล์ตรง ๆ แทน (ไม่ใช่ ffmpeg จริง แต่ทำให้
    dst มีไบต์เสียงจริงรออยู่เสมอสำหรับเทสต์ที่ตั้งใจให้เดินไปถึง extract_voiceprints จริง)
    """

    def _copy(src, dst):
        shutil.copy(src, dst)
        return dst

    monkeypatch.setattr("src.enroll.convert_to_wav", _copy)


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


def test_scan_audio_skips_a_folder_thats_locked_instead_of_raising(tmp_path, monkeypatch, caplog):
    """finding 2: directory.is_dir() หลุด PermissionError ได้ (Minor C) -- ถ้าปล่อยลอยขึ้น
    ไปตรง ๆ /api/enroll ทั้งหน้าจะ 500 เพราะโฟลเดอร์ทั้งโฟลเดอร์ถูกล็อกชั่วคราว ต้องคืน []
    ให้รอบ poll นี้แทน (finding 3: ต้อง log ไว้ด้วย ไม่งั้นล็อกค้างถาวรไม่มีร่องรอยให้ตามหา)
    """
    (tmp_path / "enroll").mkdir()
    real_is_dir = Path.is_dir

    def flaky_is_dir(self, *args, **kwargs):
        if self == enroll.enroll_dir(tmp_path):
            raise PermissionError("locked by antivirus")
        return real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", flaky_is_dir)

    with caplog.at_level("WARNING"):
        assert enroll.scan_audio(tmp_path) == []
    assert "enroll" in caplog.text


def test_scan_audio_skips_and_logs_a_file_thats_locked_instead_of_raising(
    tmp_path, monkeypatch, caplog
):
    """finding 3: การข้ามไฟล์ที่ stat ไม่สำเร็จชั่วคราว (finding 2/Minor C) ต้อง log ไว้
    ไม่งั้นไฟล์ที่ถูกล็อกค้างถาวรจะไม่โผล่ใน /enroll ตลอดกาลโดยไม่มีอะไรบอกผู้ใช้เลย
    """
    directory = tmp_path / "enroll"
    directory.mkdir()
    locked_path = directory / "locked.ogg"
    locked_path.write_bytes(b"x")
    real_is_file = Path.is_file

    def flaky_is_file(self, *args, **kwargs):
        if self == locked_path:
            raise PermissionError("locked by antivirus")
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", flaky_is_file)

    with caplog.at_level("WARNING"):
        assert enroll.scan_audio(tmp_path) == []
    assert "locked.ogg" in caplog.text


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


class _StubEmbedder:
    """embedder ปลอมสำหรับเทสต์ที่ mock extract_voiceprints ไปแล้ว -- ไม่เคยถูกเรียกจริง
    มีไว้แค่ให้ analyze() ส่งต่อ (ต้องมี .checkpoint เพราะ analyze() อ่านมันตรง ๆ ตอน
    ประกอบผล "ok") ใช้ตัวเดียวกันซ้ำในเทสต์ plumbing ที่ไม่ได้ตั้งใจตรวจค่า embedding_model
    """

    checkpoint = "stub-checkpoint"


def _stub_extract_voiceprints(monkeypatch, mapping):
    """แทน src.enroll.extract_voiceprints ด้วยค่าที่กำหนดเอง

    เทสต์กลุ่มนี้ตรวจการเดินสายของ analyze() เอง (นับคนจาก turns / เหตุผล reject / ส่ง
    ต่อ pipeline/model) ไม่ใช่การคำนวณเวกเตอร์จริงซึ่งมีเทสต์ละเอียดของตัวเองอยู่แล้วใน
    tests/test_voiceprint.py -- ไฟล์เสียงปลอม (b"x") ที่เทสต์กลุ่มนี้ใช้ไม่ใช่ wav จริง
    extract_voiceprints ตัวจริงจะ load_waveform ไม่ขึ้นแล้วเงียบ ๆ คืน {} เสมอ (ดู
    docstring ของมันเอง) ทำให้ status "ok" ไปไม่ถึงเลยถ้าไม่ mock จุดนี้
    """
    monkeypatch.setattr("src.enroll.extract_voiceprints", lambda *a, **k: mapping)


def test_analyze_accepts_a_single_speaker_who_talks_long_enough(tmp_path, monkeypatch):
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    result = DiarizationResult(
        turns=[
            {"start": 0.0, "end": 40.0, "speaker": "SPEAKER_00"},
            {"start": 45.0, "end": 73.5, "speaker": "SPEAKER_00"},
        ],
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))
    _stub_extract_voiceprints(
        monkeypatch,
        {"SPEAKER_00": Voiceprint(embedding=[0.1, 0.2, 0.3], seconds=40.0, segment_count=1)},
    )

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

    assert analyzed["status"] == "ok"
    assert analyzed["speaker_count"] == 1
    assert analyzed["speaking_seconds"] == 68.5
    assert analyzed["suggested_name"] == "สมชาย"
    assert analyzed["embedding"] == [0.1, 0.2, 0.3]
    assert analyzed["embedding_seconds"] == 40.0
    assert analyzed["segment_count"] == 1
    # ผลค้างข้ามการสลับโมเดลได้ -- ป้ายนี้คือสิ่งเดียวที่บอกว่าเวกเตอร์นี้เป็นของ
    # พื้นที่ไหน ตอนที่ผู้ใช้กลับมากดยืนยันชื่อในภายหลัง
    assert analyzed["model"] == MODEL
    # embedding_model มาจาก embedder.checkpoint ที่คำนวณจริง ไม่ใช่ model (checkpoint
    # ของตัวแยกผู้พูด) -- สองโมเดลนี้เป็นคนละตัวกันโดยเจตนา (ดู docstring ของ analyze())
    assert analyzed["embedding_model"] == _StubEmbedder.checkpoint
    assert "reason" not in analyzed


def test_analyze_rejects_a_file_with_more_than_one_speaker(tmp_path, monkeypatch):
    audio_path = tmp_path / "ประชุมย่อย.m4a"
    audio_path.write_bytes(b"x")
    result = DiarizationResult(
        turns=[
            {"start": 0.0, "end": 60.0, "speaker": "SPEAKER_00"},
            {"start": 60.0, "end": 140.0, "speaker": "SPEAKER_01"},
        ],
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "multiple_speakers"
    assert analyzed["speaker_count"] == 2
    # ตัวเลขต้องยังอยู่ให้หน้าเว็บอธิบายเหตุผลได้
    assert analyzed["speaking_seconds"] == 140.0
    # เวกเตอร์ที่ไม่ควรถูกใช้ ต้องไม่มีอยู่ในไฟล์ให้ใครหยิบไปใช้ผิด -- ด่านนี้ reject
    # ก่อนจะเรียก extract_voiceprints เลยด้วยซ้ำ (ไม่มีการ mock มันในเทสต์นี้)
    assert "embedding" not in analyzed


def test_analyze_rejects_a_long_file_holding_only_a_few_seconds_of_speech(
    tmp_path, monkeypatch
):
    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"x")
    # ไฟล์ยาว 5 นาที แต่มีเสียงพูดจริง 6 วินาที -- ต้องดูเวลาพูด ไม่ใช่ความยาวไฟล์
    result = DiarizationResult(
        turns=[{"start": 200.0, "end": 206.0, "speaker": "SPEAKER_00"}],
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

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
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))
    _stub_extract_voiceprints(
        monkeypatch,
        {
            "SPEAKER_00": Voiceprint(
                embedding=[0.1, 0.2], seconds=MIN_SPEAKING_SECONDS, segment_count=1
            )
        },
    )

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

    assert analyzed["status"] == "ok"
    assert analyzed["speaking_seconds"] == MIN_SPEAKING_SECONDS


def test_analyze_rejects_a_file_with_no_speech_at_all(tmp_path, monkeypatch):
    audio_path = tmp_path / "silence.wav"
    audio_path.write_bytes(b"x")
    monkeypatch.setattr(
        "src.enroll.diarize_audio",
        fake_diarize(DiarizationResult(turns=[])),
    )

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "too_short"
    assert analyzed["speaker_count"] == 0


def test_analyze_turns_a_pipeline_crash_into_a_rejected_result(tmp_path, monkeypatch):
    audio_path = tmp_path / "broken.ogg"
    audio_path.write_bytes(b"x")
    monkeypatch.setattr(
        "src.enroll.diarize_audio", fake_diarize(error=RuntimeError("cuda oom"))
    )

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

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
        # สั้นกว่า MIN_SPEAKING_SECONDS โดยตั้งใจ: เทสต์นี้สนใจแค่ว่า pipeline ที่ส่งเข้า
        # มาถูกส่งต่อให้ diarize_audio ตัวเดิมโดยไม่ถูกแทนที่ -- ให้ผลจบที่ too_short เร็ว
        # ที่สุดพอ ไม่ต้องพึ่ง extract_voiceprints/embedder เลยเพื่อตัดความซับซ้อนที่ไม่
        # เกี่ยวกับสิ่งที่เทสต์นี้ตรวจ
        return DiarizationResult(
            turns=[{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
        )

    monkeypatch.setattr("src.enroll.diarize_audio", _spy)

    enroll.analyze(audio_path, pipeline=sentinel, model=MODEL, embedder=_StubEmbedder())

    assert seen["pipeline"] is sentinel


def test_analyze_converts_to_wav_before_diarizing(tmp_path, monkeypatch):
    """finding 4: enroll ต้อง normalize เสียงด้วย convert_to_wav แบบเดียวกับ
    pipeline.process_file ก่อนส่งเข้า diarize_audio -- ไม่งั้น pyannote ได้ container ดิบ
    ที่ decode ด้วย backend ที่เดาไม่ได้ (soundfile ถอด AAC ไม่ได้) แทนที่จะผ่าน ffmpeg
    ซึ่งเป็นตัวถอดรหัสเดียวที่โปรเจกต์นี้รับประกัน

    turns สั้นกว่า MIN_SPEAKING_SECONDS โดยตั้งใจ (Task 10): analyze() เวอร์ชันนี้แปลง
    เป็น wav สองครั้ง (ครั้งที่สองก่อนสร้าง voiceprint) -- ถ้า turns ยาวพอจนถึง "ok" การ
    แปลงครั้งที่สองจะเรียก fake_convert ซ้ำแล้วทับ seen["dst"] เดิม ทำให้เทียบกับ
    seen["diarized_path"] (จับค่าไว้ตอนแปลงครั้งแรก) ไม่ตรงกันอย่างผิดจุดประสงค์ของเทสต์
    นี้ ซึ่งต้องการตรวจแค่การแปลงครั้งแรกก่อน diarize เท่านั้น
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
            turns=[{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}],
        )

    monkeypatch.setattr("src.enroll.convert_to_wav", fake_convert)
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize)

    enroll.analyze(audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder())

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

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "analysis_failed"
    assert "ffmpeg exited with code 1" in analyzed["detail"]
    assert "embedding" not in analyzed
    assert diarize_calls == []


# --- รีวิวรอบนี้: finding 1/2/3 ---


def test_analyze_rejects_when_the_embedding_rests_on_less_than_the_minimum(
    tmp_path, monkeypatch
):
    """finding 1 (ตัดสินใจแล้วโดยเจ้าของโปรเจกต์): เวกเตอร์มาจากเฉพาะช่วงเสียงสะอาดที่
    select_intervals คัดแล้วเท่านั้น ซึ่งเป็นเซตย่อยของเวลาพูดทั้งหมด -- ก่อนหน้านี้เวกเตอร์
    มาจาก centroid ที่ครอบคลุมเสียงพูดทั้งหมดของคน ๆ นั้น MIN_SPEAKING_SECONDS จึงเคยผูกกับ
    เวกเตอร์ได้จริง ตอนนี้ไฟล์ที่มี speaking_seconds ผ่านด่าน (>=10) แต่เวกเตอร์จริงมาจาก
    เสียงแค่ไม่กี่วินาทีต้องยังถูกปฏิเสธ (ทำซ้ำจาก: turns เก้าท่อน 1.0 วิ บวกอีกหนึ่งท่อน 2.0
    วิ -- speaking_seconds 11.0 ผ่านด่าน แต่ embedding_seconds เหลือแค่ 2.0)
    """
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    turns = [{"start": float(i), "end": float(i) + 1.0, "speaker": "SPEAKER_00"} for i in range(9)]
    turns.append({"start": 20.0, "end": 22.0, "speaker": "SPEAKER_00"})
    result = DiarizationResult(turns=turns)
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))
    _stub_extract_voiceprints(
        monkeypatch,
        {"SPEAKER_00": Voiceprint(embedding=[0.1, 0.2], seconds=2.0, segment_count=1)},
    )

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "too_short"
    # ผลนี้ต้องยังมีคีย์ base ครบเหมือนการปฏิเสธอื่นทุกครั้ง
    assert analyzed["speaker_count"] == 1
    assert analyzed["speaking_seconds"] == 11.0
    assert analyzed["suggested_name"] == "สมชาย"
    assert "embedding" not in analyzed


def test_analyze_reports_unusable_embedding_when_extraction_omits_the_counted_label(
    tmp_path, monkeypatch
):
    """finding 2: บรานช์ "extraction สำเร็จแต่ไม่มี voiceprint ของป้ายที่นับเป็นคน" (เดิม
    ไม่มีเทสต์ตรวจเลย) -- ป้ายเดียวพูดยาวพอผ่านทั้งด่านนับคนและด่าน speaking_seconds แต่
    extract_voiceprints (จำลองว่าทำงานจบแล้ว) ไม่มี key ของป้ายนั้นใน dict ที่คืนมา (เช่น
    ทุกช่วงของป้ายนั้นคำนวณเวกเตอร์ไม่ได้จริง) ไฟล์ 45 วินาทีที่ extract ไม่สำเร็จต้องไม่
    ถูกรายงานเป็น 0.0 วิแล้วปฏิเสธว่า "สั้นเกินไป" (ดู docstring ของ analyze()) --
    speaking_seconds/speaker_count ต้องยังเป็นค่าจริง
    """
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    result = DiarizationResult(
        turns=[{"start": 0.0, "end": 45.0, "speaker": "SPEAKER_00"}],
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))
    _stub_extract_voiceprints(monkeypatch, {})  # extraction จบแล้ว แต่ไม่มี voiceprint ให้ป้ายนี้

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "unusable_embedding"
    assert analyzed["speaker_count"] == 1
    assert analyzed["speaking_seconds"] == 45.0
    assert "embedding" not in analyzed


def test_analyze_reports_unusable_embedding_when_extraction_raises(tmp_path, monkeypatch):
    """finding 2: บรานช์ "extraction raise" (เดิมไม่มีเทสต์ตรวจเลย) -- ไฟล์ 45 วินาทีที่
    extract_voiceprints พังทั้งฟังก์ชัน (เช่น torch/pyannote โยน exception ที่ไม่ได้ถูก
    กลืนไว้เอง) ต้องยังรายงาน speaking_seconds/speaker_count เป็นค่าจริง ไม่ใช่ 0.0 แล้ว
    ปฏิเสธผิดเหตุผลเป็น "สั้นเกินไป"
    """
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    result = DiarizationResult(
        turns=[{"start": 0.0, "end": 45.0, "speaker": "SPEAKER_00"}],
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))

    def _raise(*args, **kwargs):
        raise RuntimeError("cuda oom")

    monkeypatch.setattr("src.enroll.extract_voiceprints", _raise)

    analyzed = enroll.analyze(
        audio_path, pipeline=object(), model=MODEL, embedder=_StubEmbedder()
    )

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "unusable_embedding"
    assert analyzed["speaker_count"] == 1
    assert analyzed["speaking_seconds"] == 45.0
    assert "embedding" not in analyzed


def test_analyze_does_not_raise_when_the_embedder_has_no_checkpoint_attribute(
    tmp_path, monkeypatch
):
    """finding 3: embedder.checkpoint เดิมถูกอ่านนอก try ทั้งหมด -- embedder ที่ทำงานได้ปกติ
    ทุกอย่างแต่ไม่มี attribute นี้ (คนละกรณีกับ extract_voiceprints พัง) ทำให้ analyze()
    raise AttributeError หลุดออกไปตรง ๆ analyze() สัญญาไว้ในชื่อฟังก์ชันของมันเองว่าไม่ raise
    ไม่ว่าเกิดอะไรขึ้น เพราะผู้เรียกคือ watcher ที่ไม่มีมนุษย์นั่งดูอยู่ -- ไฟล์ผลที่ไม่ถูกเขียน
    เลยคือหน้าเว็บค้างที่ "กำลังวิเคราะห์" ตลอดกาล
    """
    audio_path = tmp_path / "สมชาย.ogg"
    audio_path.write_bytes(b"x")
    result = DiarizationResult(
        turns=[{"start": 0.0, "end": 40.0, "speaker": "SPEAKER_00"}],
    )
    monkeypatch.setattr("src.enroll.diarize_audio", fake_diarize(result))
    _stub_extract_voiceprints(
        monkeypatch,
        {"SPEAKER_00": Voiceprint(embedding=[0.1, 0.2], seconds=40.0, segment_count=1)},
    )

    class _EmbedderWithoutCheckpoint:
        """embedder ที่ทำงานได้ปกติทุกอย่าง ยกเว้นไม่มี .checkpoint -- จงใจไม่สืบทอดจาก
        _StubEmbedder เพื่อไม่ให้บังเอิญมี attribute นี้ติดมา"""

    analyzed = enroll.analyze(
        audio_path,
        pipeline=object(),
        model=MODEL,
        embedder=_EmbedderWithoutCheckpoint(),
    )

    assert analyzed["status"] == "rejected"
    assert analyzed["reason"] == "unusable_embedding"
    assert analyzed["speaker_count"] == 1
    assert analyzed["speaking_seconds"] == 40.0
    assert "embedding" not in analyzed


# --- Task 10: เกณฑ์ใหม่ที่นับ "กี่คน" จาก select_intervals(clean_intervals(turns)) ---
# เทสต์กลุ่มนี้ (จาก task-10-brief.md) ใช้ไฟล์เสียงจริง (_write_enroll_wav) กับ pipeline
# ปลอมแทน diarize_audio ปลอม -- ปล่อยให้ diarize_audio/extract_voiceprints ตัวจริงทำงาน
# เต็มเส้นทาง มีแค่ตัวโมเดล embedding ที่ถูกปลอม (embedder ไม่ต้องมี GPU/HF_TOKEN)


class _FakeEmbedder:
    checkpoint = "pyannote/wespeaker-voxceleb-resnet34-LM"

    def __call__(self, waveform, intervals):
        return [[1.0, 0.0] for _ in intervals]


def _pipeline_returning(turns):
    """pipeline ปลอมที่ให้ turn ตามที่กำหนด (diarize_audio อ่าน .speaker_diarization)"""
    tracks = [(SimpleNamespace(start=t["start"], end=t["end"]), None, t["speaker"]) for t in turns]
    diarization = MagicMock()
    diarization.itertracks.return_value = tracks
    return MagicMock(return_value=MagicMock(speaker_diarization=diarization))


def _write_enroll_wav(tmp_path, seconds=40.0, sample_rate=16000):
    directory = tmp_path / "enroll"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "satit.wav"
    frames = int(seconds * sample_rate)
    sf.write(path, np.sin(np.linspace(0.0, 4000.0, frames, dtype="float32")), sample_rate)
    return path


def test_analyze_accepts_a_clip_whose_only_extra_label_is_a_short_fragment(tmp_path):
    # pyannote แตกป้ายเศษออกจากคลิปพูดคนเดียวจริงราว 40% ของครั้ง (วัด 2026-07-29: เศษ
    # 0.5 วิซ้อนกลางช่วงที่ป้ายจริงพูดอยู่) เศษแบบนั้นไม่มีเสียงสะอาดถึง 1.5 วิ จึงไม่ใช่คน
    audio = _write_enroll_wav(tmp_path, seconds=40.0)
    pipeline = _pipeline_returning(
        [
            {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"},
            {"start": 12.8, "end": 13.3, "speaker": "SPEAKER_01"},
        ]
    )

    result = analyze(audio, pipeline=pipeline, model="m", embedder=_FakeEmbedder())

    assert result["status"] == "ok"
    assert result["speaker_count"] == 1
    assert result["ignored_short_labels"] == 1


def test_analyze_still_rejects_a_clip_with_a_real_second_speaker(tmp_path):
    # คนที่สองที่พูดสะอาด 5 วินาทีต้องยังถูกจับได้ -- นี่คือสิ่งที่ด่านนี้มีไว้กันจริง ๆ
    audio = _write_enroll_wav(tmp_path, seconds=40.0)
    pipeline = _pipeline_returning(
        [
            {"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00"},
            {"start": 25.0, "end": 30.0, "speaker": "SPEAKER_01"},
        ]
    )

    result = analyze(audio, pipeline=pipeline, model="m", embedder=_FakeEmbedder())

    assert result == {**result, "status": "rejected", "reason": "multiple_speakers"}
    assert result["speaker_count"] == 2


def test_analyze_stamps_the_embedding_model_it_actually_used(tmp_path):
    # ต้องเป็นชื่อจาก embedder ที่คำนวณจริง ไม่ใช่ค่าใน config ตอนกดยืนยันทีหลัง
    audio = _write_enroll_wav(tmp_path, seconds=40.0)
    pipeline = _pipeline_returning([{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}])

    result = analyze(audio, pipeline=pipeline, model="diar-model", embedder=_FakeEmbedder())

    assert result["embedding_model"] == "pyannote/wespeaker-voxceleb-resnet34-LM"
    assert result["model"] == "diar-model"
    assert result["segment_count"] == 1
    assert result["embedding_seconds"] == 30.0


def test_analyze_rejects_a_clip_too_short_before_looking_at_vectors(tmp_path):
    # ไฟล์ที่ไม่มีเสียงพูดเลยต้องได้เหตุผลที่ผู้ใช้ทำตามได้ ไม่ใช่ภาษาของระบบ
    audio = _write_enroll_wav(tmp_path, seconds=40.0)
    pipeline = _pipeline_returning([{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00"}])

    result = analyze(audio, pipeline=pipeline, model="m", embedder=_FakeEmbedder())

    assert result["reason"] == "too_short"


def test_analyze_reports_unusable_embedding_when_extraction_produces_nothing(tmp_path):
    audio = _write_enroll_wav(tmp_path, seconds=40.0)
    pipeline = _pipeline_returning([{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}])

    class _DeadEmbedder:
        checkpoint = "e"

        def __call__(self, waveform, intervals):
            return [None for _ in intervals]

    result = analyze(audio, pipeline=pipeline, model="m", embedder=_DeadEmbedder())

    assert result["reason"] == "unusable_embedding"


def make_audio(tmp_path, name="สมชาย.ogg"):
    directory = tmp_path / "enroll"
    directory.mkdir(exist_ok=True)
    path = directory / name
    path.write_bytes(b"fake audio")
    return path


def write_ok_result(base_dir, audio_file, analyzed, **kwargs):
    """เขียนผล status ok พร้อมผูกกับ stat ปัจจุบันของไฟล์เสียงเสมอ

    หลัง finding 1 (defense in depth): write_result ที่ได้รับ status "ok" แต่ไม่มี
    pre_analysis_stat มาเทียบ (None) จะถือว่ายืนยันไม่ได้แล้วล้างทั้งสอง sidecar ทิ้งทันที
    (finding 4) -- เทสต์จำนวนมากในไฟล์นี้แค่ต้องการ fixture "วิเคราะห์เสร็จแล้วสถานะ ok"
    ไม่ได้ตั้งใจทดสอบกลไกผูกไฟล์เอง จึงรวม stat ที่ต้องผูกไว้ที่นี่เรียกครั้งเดียว แทนที่
    จะคำนวณ pre_analysis_stat ซ้ำทุกจุดเรียก
    """
    stat = (base_dir / "enroll" / audio_file).stat()
    return enroll.write_result(
        base_dir,
        audio_file,
        analyzed,
        pre_analysis_stat=(stat.st_size, stat.st_mtime),
        **kwargs,
    )


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


def test_write_request_does_not_queue_a_file_thats_locked_instead_of_raising(
    tmp_path, monkeypatch
):
    """finding 2: (enroll_dir/audio_file).is_file() หลุด PermissionError ได้ (Minor C) --
    ไม่รู้แน่ชัดว่าไฟล์มีอยู่จริงไหม อย่าฟันธงว่า "ไม่มี" แต่ก็อย่าสั่งงานรอบนี้ ผู้ใช้กด
    วิเคราะห์ใหม่ได้เองเมื่อล็อกคลาย -- สำคัญคือไม่ raise ออกไปเป็น 500
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    real_is_file = Path.is_file

    def flaky_is_file(self, *args, **kwargs):
        if self == audio_path:
            raise PermissionError("locked by antivirus")
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", flaky_is_file)

    assert enroll.write_request(tmp_path, "สมชาย.ogg") is None
    assert not (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()


def test_write_request_refuses_a_file_whose_extension_is_not_audio(tmp_path):
    """finding 6: /api/enroll/analyze เดิมเช็คแค่ is_safe_filename ก่อนเรียก write_request --
    ชื่อที่ปลอดภัยแต่ไม่ใช่ไฟล์เสียง (เช่น notes.txt ที่มีอยู่จริงใน enroll/) ผ่านเข้ามาเขียน
    notes.txt.request.json ได้ ทั้งที่ scan_audio ไม่มีวันเห็นมัน (ไม่ใช่ AUDIO_EXTENSIONS)
    และ orphan sweep ก็ไม่ลบให้ (ไฟล์ต้นทางยังอยู่จริง) -- กลายเป็นขยะค้างตลอดกาล
    """
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "notes.txt").write_bytes(b"not audio")

    assert enroll.write_request(tmp_path, "notes.txt") is None
    assert not (directory / "notes.txt.request.json").exists()


def test_pending_requests_lists_requests_that_have_no_result_yet(tmp_path):
    make_audio(tmp_path, "a.ogg")
    make_audio(tmp_path, "b.ogg")
    make_audio(tmp_path, "c.ogg")
    enroll.write_request(tmp_path, "a.ogg")
    enroll.write_request(tmp_path, "b.ogg")
    write_ok_result(tmp_path, "b.ogg", {"status": "ok"})

    # a สั่งแล้วยังไม่มีผล, b มีผลแล้ว, c ยังไม่ได้สั่ง
    assert enroll.pending_requests(tmp_path) == ["a.ogg"]


def test_pending_requests_is_empty_when_the_folder_does_not_exist(tmp_path):
    assert enroll.pending_requests(tmp_path) == []


def test_pending_requests_skips_a_locked_folder_instead_of_raising(tmp_path, monkeypatch, caplog):
    """Minor C: directory.is_dir() ที่นี่เดิมเป็นแบบดิบ ไม่มี try/except เลย -- ต้องคืน []
    ให้รอบ poll นี้แทนการปล่อยให้ PermissionError หลุดขึ้นไปฆ่า watcher ทั้งตัว
    """
    (tmp_path / "enroll").mkdir()
    real_is_dir = Path.is_dir

    def flaky_is_dir(self, *args, **kwargs):
        if self == enroll.enroll_dir(tmp_path):
            raise PermissionError("locked by antivirus")
        return real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", flaky_is_dir)

    with caplog.at_level("WARNING"):
        assert enroll.pending_requests(tmp_path) == []
    assert "enroll" in caplog.text


def test_pending_requests_does_not_lose_other_files_when_one_result_sidecar_is_locked(
    tmp_path, monkeypatch
):
    """Minor C (finding 1 ของรีวิวรอบนี้): ก่อนแก้ result.is_file() เป็นการเรียกดิบ --
    PermissionError ชั่วคราวจาก sidecar ผลของไฟล์เดียวหลุดออกไปเป็น exception จริง แล้ว
    ฟังก์ชันทั้งก้อน raise ทำให้ watcher.process_enroll_requests (ซึ่งครอบด้วย try/except
    Exception เดียว) ข้ามการวิเคราะห์ *ทุกไฟล์ที่ค้างอยู่* ในรอบ poll นั้น ไม่ใช่แค่ไฟล์ที่ล็อก
    ต้องไม่ raise เลย และ 'a' ที่ไม่เกี่ยวข้องต้องยังอยู่ในผลลัพธ์
    """
    make_audio(tmp_path, "a.ogg")
    make_audio(tmp_path, "locked.ogg")
    enroll.write_request(tmp_path, "a.ogg")
    enroll.write_request(tmp_path, "locked.ogg")
    locked_result_path = tmp_path / "enroll" / "locked.ogg.result.json"
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self == locked_result_path:
            raise PermissionError("locked by antivirus")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    waiting = enroll.pending_requests(tmp_path)

    assert "a.ogg" in waiting
    # ambiguous ต้องถูกนับเป็น "มีผลแล้ว" (finding 5 ของรีวิวรอบนี้) -- ไม่งั้นไฟล์นี้จะถูก
    # ส่งวิเคราะห์ซ้ำทั้งที่ไม่รู้แน่ชัดว่ามีผลอยู่แล้วหรือไม่
    assert "locked.ogg" not in waiting


def test_pending_requests_treats_a_locked_request_sidecar_as_still_queued(tmp_path, monkeypatch):
    """ล็อกชั่วคราวที่ request.json (ไม่ใช่ result.json) ต้องไม่ถูกตีความว่า "ยังไม่ได้สั่งงาน"
    -- PermissionError แปลว่าไฟล์มีอยู่จริง (แค่เข้าไม่ได้ชั่วครู่) ไม่ใช่ไฟล์หายไป
    """
    make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")
    locked_request_path = tmp_path / "enroll" / "สมชาย.ogg.request.json"
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self == locked_request_path:
            raise PermissionError("locked by antivirus")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    assert enroll.pending_requests(tmp_path) == ["สมชาย.ogg"]


def test_write_result_stamps_the_file_and_keeps_the_embedding(tmp_path):
    make_audio(tmp_path)

    path = write_ok_result(
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

    path = write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["audio_size"] == stat.st_size
    assert payload["audio_mtime"] == pytest.approx(stat.st_mtime)


def test_write_result_verifies_binding_for_a_payload_carrying_an_embedding_without_status_ok(
    tmp_path,
):
    """finding 2 ของรีวิวรอบนี้: การ์ดกันไม่ให้ embedding หลุดโดยไม่ผ่านการยืนยัน binding
    เดิมเช็คจาก analyzed.get("status") == "ok" ซึ่งเป็นแค่ตัวแทนของสิ่งที่ต้องกันจริง --
    ถ้า analyzed มี "embedding" ติดมาโดยไม่มี status "ok" (เช่นถูกแก้ไขระหว่างทาง หรือผู้เรียก
    ในอนาคตที่ไม่ผ่าน analyze()) ต้องยังถูกบังคับให้ยืนยัน binding เหมือนกันทุกประการ ไม่ใช่
    หลุดผ่านไปได้เงียบ ๆ เพราะ status ไม่ตรงกับ "ok" พอดี
    """
    make_audio(tmp_path, "สมชาย.ogg")

    result_path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "weird", "embedding": [0.9, 0.9]},
        # ไม่มี pre_analysis_stat มาเทียบ -- ต้องยืนยันไม่ได้เหมือนกรณี status "ok"
    )

    assert result_path is None
    assert not (tmp_path / "enroll" / "สมชาย.ogg.result.json").exists()


def test_write_result_computed_binding_wins_over_a_spread_analyzed_dict(tmp_path):
    """finding 3 ของรีวิวรอบนี้: payload เดิม spread **analyzed หลัง audio_size/audio_mtime
    -- ถ้า analyzed มีคีย์ audio_file/analyzed/audio_size/audio_mtime ปนมาด้วย มันจะทับค่าที่
    คำนวณไว้แบบเงียบ ๆ ทำให้ผลที่เขียนลงดิสก์ผูกกับ binding ปลอมแทนที่จะเป็น binding จริงที่
    เพิ่งยืนยันผ่านไป ต้อง spread **analyzed ก่อนแล้วให้ bookkeeping keys ตามหลังเสมอ
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    stat = audio_path.stat()
    pre_analysis_stat = (stat.st_size, stat.st_mtime)

    path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {
            "status": "ok",
            "embedding": [0.1],
            "audio_file": "ปลอม.ogg",
            "analyzed": "ปลอม",
            "audio_size": -999,
            "audio_mtime": -999.0,
        },
        pre_analysis_stat=pre_analysis_stat,
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["audio_file"] == "สมชาย.ogg"
    assert payload["audio_size"] == stat.st_size
    assert payload["audio_mtime"] == pytest.approx(stat.st_mtime)
    assert payload["analyzed"] != "ปลอม"


def test_read_result_returns_none_when_the_audio_file_was_replaced_after_analysis(tmp_path):
    """finding 1 (critical): ผู้ใช้ลบไฟล์เดิมทิ้งแล้ววางไฟล์ใหม่ชื่อเดียวกัน (เนื้อหา/ขนาด
    ต่างกัน) -- ผลวิเคราะห์เก่าต้องไม่ผูกเข้ากับไฟล์ใหม่ ไม่งั้นเวกเตอร์เสียงของคนเดิม
    จะถูกเสนอให้ยืนยันภายใต้ชื่อคนใหม่ และหลุดเข้าทะเบียนถาวรถ้ากดบันทึก
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1, 0.2]})

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
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
    stat = audio_path.stat()
    os.utime(audio_path, (stat.st_atime, stat.st_mtime + 1.0))

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is not None


def test_read_result_rejects_an_mtime_drift_beyond_the_tolerance(tmp_path):
    audio_path = make_audio(tmp_path)
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
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


def test_list_entries_skips_a_row_thats_locked_instead_of_500ing_the_whole_page(tmp_path):
    """finding 2: path.stat() เดิมดัก FileNotFoundError อย่างเดียว -- PermissionError
    ชั่วคราว (Minor C) ต้องข้ามแถวนั้นไปเหมือนกัน ไม่ใช่ raise ออกไปเป็น 500 ทั้งหน้า
    (ต่างจาก FileNotFoundError ตรงที่ไฟล์นี้ไม่ได้หายจริง แค่ล็อกชั่วคราว)
    """
    make_audio(tmp_path, "ok.ogg")
    locked_path = make_audio(tmp_path, "locked.ogg")
    real_stat = Path.stat
    calls = {"locked.ogg": 0}

    def flaky_stat(self, *args, **kwargs):
        if self == locked_path:
            calls["locked.ogg"] += 1
            if calls["locked.ogg"] > 1:
                raise PermissionError("locked by antivirus")
        return real_stat(self, *args, **kwargs)

    import unittest.mock as mock

    with mock.patch.object(Path, "stat", flaky_stat):
        entries = enroll.list_entries(tmp_path)

    names = [entry["audio_file"] for entry in entries]
    assert names == ["ok.ogg"]


def test_list_entries_treats_a_request_thats_locked_as_not_queued_yet(tmp_path, monkeypatch):
    """finding 2: request.is_file() หลุด PermissionError ได้เหมือนกัน -- ต้องไม่ raise ออก
    ไปเป็น 500 แค่ถือว่ายังไม่มีใบสั่งงานให้เห็นในรอบนี้ (state "idle" ไม่ใช่ "queued")
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")
    request_file = audio_path.with_name(audio_path.name + ".request.json")
    real_is_file = Path.is_file

    def flaky_is_file(self, *args, **kwargs):
        if self == request_file:
            raise PermissionError("locked by antivirus")
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", flaky_is_file)

    entries = enroll.list_entries(tmp_path)

    assert entries[0]["state"] == "idle"


def test_write_result_clears_both_sidecars_when_the_file_was_replaced_mid_analysis(
    tmp_path,
):
    """CRITICAL A + finding 4: analyze() อ่านไบต์ ณ เวลา T1 แต่ write_result เดิม stat()
    ไฟล์ตอน T3 (นาทีให้หลัง) เท่านั้น -- ระหว่างนั้นผู้ใช้แทนที่ enroll/สมชาย.ogg ด้วยการอัด
    คนละคน ผลที่ได้ต้องไม่ถูกผูกเข้ากับไฟล์ใหม่แล้วรับรองว่า "ok" เพราะ embedding เป็นของ
    ไบต์ชุดเดิมที่ไม่มีอยู่บนดิสก์แล้ว

    finding 4: เดิม (ก่อนแก้) ทางแก้คือเขียน "rejected" ค้างไว้แทน -- แต่นั่นสร้างทางตันใหม่:
    audio_size/audio_mtime ที่บันทึกจะ "ตรง" กับไฟล์ใหม่ (คนละคน แต่ถูกต้องสมบูรณ์) ที่ยังอยู่
    บนดิสก์จริง ทำให้ rejected นั้นยืนยันได้ตลอดกาลสำหรับไฟล์ใหม่ การ์ดค้างที่ "ใช้ไม่ได้"
    ตลอดไปเพราะปุ่มวิเคราะห์เก็บเฉพาะไฟล์ state "idle" ตอนนี้ต้องล้างทั้ง .result.json และ
    .request.json ทิ้งแทน ให้การ์ดกลับไป idle แล้วผู้ใช้กดวิเคราะห์ไฟล์ที่วางใหม่ได้เอง
    (ทดสอบว่าใบสั่งงานหายไปด้วย ไม่ใช่แค่ผล -- ไม่งั้นจะยังค้างสถานะ "queued" อยู่ดี)
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")
    stat_before_analysis = audio_path.stat()
    pre_analysis_stat = (stat_before_analysis.st_size, stat_before_analysis.st_mtime)

    # จำลองผู้ใช้แทนที่ไฟล์ระหว่าง diarization กำลังทำงานอยู่ (เนื้อหา/ขนาดต่างไปจากเดิม)
    audio_path.unlink()
    audio_path.write_bytes(b"a completely different recording, replaced mid-analysis")

    result_path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2], "suggested_name": "สมชาย"},
        pre_analysis_stat=pre_analysis_stat,
    )

    assert result_path is None
    assert not (tmp_path / "enroll" / "สมชาย.ogg.result.json").exists()
    assert not (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()
    # ไฟล์ใหม่ (คนละคน) ที่วางทับต้องยังอยู่ให้วิเคราะห์ใหม่ได้ ไม่ถูกแตะต้อง
    assert audio_path.is_file()


def test_write_result_leaves_a_changed_marker_when_it_discards_a_result(tmp_path):
    """finding 5 ของรีวิวรอบนี้: การล้าง sidecar เพราะผูกไม่ได้ (CRITICAL A) ทำให้การ์ดเด้ง
    จาก "กำลังวิเคราะห์" กลับไป "รอวิเคราะห์" แบบเงียบ ๆ โดยไม่มีร่องรอยอะไรบอกผู้ใช้เลยว่า
    ทำไม -- ต้องทิ้งเครื่องหมายไว้ให้ list_entries อ่านแล้วอธิบายเหตุผลได้
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")
    stat_before_analysis = audio_path.stat()
    pre_analysis_stat = (stat_before_analysis.st_size, stat_before_analysis.st_mtime)
    audio_path.unlink()
    audio_path.write_bytes(b"a completely different recording, replaced mid-analysis")

    enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2]},
        pre_analysis_stat=pre_analysis_stat,
    )

    assert (tmp_path / "enroll" / "สมชาย.ogg.changed.json").exists()


def test_list_entries_reports_the_changed_marker_once_then_clears_it(tmp_path):
    """finding 5 ของรีวิวรอบนี้: คำเตือน "ไฟล์เปลี่ยนระหว่างวิเคราะห์" ต้องโผล่ให้ผู้ใช้เห็น
    ครั้งเดียวแล้วหายไปเอง (ไม่ค้าง state ถาวร) -- และไฟล์ยังต้องเป็น "idle" วิเคราะห์ใหม่ได้
    ปกติ ไม่ใช่ทางตันแบบ rejected ค้างไว้
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")
    stat_before_analysis = audio_path.stat()
    pre_analysis_stat = (stat_before_analysis.st_size, stat_before_analysis.st_mtime)
    audio_path.unlink()
    audio_path.write_bytes(b"a completely different recording, replaced mid-analysis")
    enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2]},
        pre_analysis_stat=pre_analysis_stat,
    )

    first = enroll.list_entries(tmp_path)
    assert first[0]["state"] == "idle"
    assert first[0]["changed_during_analysis"] is True

    second = enroll.list_entries(tmp_path)
    assert "changed_during_analysis" not in second[0]
    assert not (tmp_path / "enroll" / "สมชาย.ogg.changed.json").exists()


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


def test_write_result_clears_both_sidecars_when_it_cannot_stat_the_file_at_write_time(
    tmp_path,
):
    """CRITICAL B + finding 4: เดิม write_result เขียน audio_size/audio_mtime เป็น None
    เมื่อ stat ไม่สำเร็จ แล้วปล่อย "ok" หลุดออกไปพร้อม embedding -- ไม่มีทางผูกผลนี้กับไฟล์
    อะไรได้เลย (เช่น ผู้ใช้กด "เอาออกจากรายการ" ระหว่างที่ watcher กำลังวิเคราะห์อยู่พอดี
    ไฟล์ถูกย้ายเข้า done/ ไปแล้วตอนเขียนผล) -- finding 4: ล้าง sidecar ทิ้งแทนการเขียน
    rejected ค้างไว้ (audio_size=None ก็จริง แต่การ์ดยัง "มีผล" ค้างอยู่จนกว่าจะมีคนอ่านผ่าน
    read_result อีกที ล้างทันทีที่นี่ตัดปัญหานั้นตั้งแต่ต้น สอดคล้องกับกรณี CRITICAL A)
    """
    (tmp_path / "enroll").mkdir()  # ไม่มีไฟล์เสียงอยู่จริงตอนเขียนผล

    result_path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2], "suggested_name": "สมชาย"},
    )

    assert result_path is None
    assert not (tmp_path / "enroll" / "สมชาย.ogg.result.json").exists()
    assert not (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()


def test_write_result_clears_both_sidecars_when_pre_analysis_stat_is_missing_entirely(
    tmp_path,
):
    """finding 1 (defense in depth): pre_analysis_stat=None (ค่าเริ่มต้นของพารามิเตอร์) ต้อง
    ถือว่า "ผูกไม่ได้" เหมือนกรณี CRITICAL B ไม่ใช่ "ข้ามการเช็คคู่นี้ไปเงียบ ๆ" อย่างเดิม --
    เดิมคอมเมนต์ในโค้ดให้เหตุผลว่าต้อง "คงพฤติกรรมเดิมไว้ให้ผู้เรียกอื่นที่ไม่ได้อยู่ในเส้นทาง
    watcher" แต่นั่นคือช่องโหว่ที่ finding 1 ชี้ไว้เอง: ไม่มี production caller ตัวไหนส่ง
    payload status "ok" เข้ามาโดยไม่มี pre_analysis_stat (watcher.process_enroll_requests
    stat ไฟล์ก่อนวิเคราะห์เสมอ ถ้า stat ไม่สำเร็จก็ continue ไม่วิเคราะห์เลย) ไฟล์นี้ยังอยู่
    ครบถ้วนไม่มีอะไรเปลี่ยนแปลง แต่การไม่มี pre_analysis_stat มาเทียบเลยแปลว่ายืนยันไม่ได้
    """
    make_audio(tmp_path, "สมชาย.ogg")

    result_path = enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2], "suggested_name": "สมชาย"},
        # pre_analysis_stat ไม่ถูกส่งมาเลย (None) -- นี่คือประเด็นของเทสต์นี้
    )

    assert result_path is None
    assert not (tmp_path / "enroll" / "สมชาย.ogg.result.json").exists()
    assert not (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()


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


def test_read_result_returns_none_without_raising_when_the_sidecar_itself_is_locked(
    tmp_path, monkeypatch
):
    """finding 2: path.is_file() บน .result.json เองก็หลุด PermissionError ได้ (Minor C)
    -- ไม่รู้แน่ชัดว่ามีผลอยู่จริงไหม ต้องคืน None ให้รอบนี้ ไม่ raise ออกไปเป็น 500 และไม่
    ลบอะไรทิ้งเพราะไม่รู้ว่ามีอะไรให้ลบ
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
    result_file = audio_path.with_name(audio_path.name + ".result.json")
    real_is_file = Path.is_file

    def flaky_is_file(self, *args, **kwargs):
        if self == result_file:
            raise PermissionError("locked by antivirus")
        return real_is_file(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_file", flaky_is_file)

    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None


def test_read_result_keeps_the_sidecar_on_a_transient_permission_error(tmp_path, monkeypatch):
    """Minor C: PermissionError ชั่วคราวจากโปรแกรมสแกนไวรัส/ตัวซิงก์ไฟล์ (ปัญหาที่โปรเจกต์
    นี้ยอมรับอยู่แล้ว ดู storage.replace_with_retry) ต้องไม่ถูกตีความว่าไฟล์เสียงหายไปแล้ว
    -- ไม่งั้นผลวิเคราะห์ GPU ที่เสร็จสมบูรณ์แล้วจะถูกลบทิ้งฟรี ๆ โดยไม่มีอะไรบอกผู้ใช้เลย
    """
    audio_path = make_audio(tmp_path, "สมชาย.ogg")
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
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
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
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
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.1]})
    result_file = audio_path.with_name(audio_path.name + ".result.json")
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self == audio_path:
            raise PermissionError("locked by antivirus")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)

    enroll.list_entries(tmp_path)  # เรียก _sweep_orphan_sidecars ภายใน

    assert result_file.exists()


def test_sweep_orphan_sidecars_does_not_raise_when_the_folder_itself_is_locked(
    tmp_path, monkeypatch
):
    """finding 2: directory.is_dir() ใน _sweep_orphan_sidecars ก็หลุด PermissionError ได้
    เหมือนกัน -- ต้องข้ามการกวาดรอบนี้ไปเฉย ๆ ไม่ raise ออกไปเป็น 500 ทั้งหน้า
    """
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "หาย.ogg.result.json").write_text("{}", encoding="utf-8")
    real_is_dir = Path.is_dir

    def flaky_is_dir(self, *args, **kwargs):
        if self == directory:
            raise PermissionError("locked by antivirus")
        return real_is_dir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "is_dir", flaky_is_dir)

    enroll.list_entries(tmp_path)  # ต้องไม่ raise

    # ไม่รู้แน่ชัดว่ามีอะไรอยู่ในโฟลเดอร์นี้จริง ๆ ไหม -- ต้องไม่ลบทิ้ง
    assert (directory / "หาย.ogg.result.json").exists()


def test_clear_removes_both_sidecars_and_leaves_the_audio(tmp_path):
    audio_path = make_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok"})

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
    write_ok_result(tmp_path, "สมชาย.ogg", {"status": "ok"})

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
    write_ok_result(
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
    write_ok_result(
        tmp_path,
        "call.wav",
        {"status": "ok", "speaker_count": 1, "suggested_name": "call-wav-person"},
    )
    write_ok_result(
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
