import subprocess
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import DEFAULT_EMBEDDING_MODEL, Config
from src.job import (
    JOB_SUFFIX,
    NO_SUMMARY_MODEL,
    read_model,
    read_transcript,
    record_transcript,
    write_job,
)
from src.llm import UnknownModelError
from src.pipeline import process_file
from src.segments import WAV_HEADER_ALLOWANCE, finish_session, part_filename, session_dir_for, write_manifest

# See tests/test_segments.py: finish_session decides which parts are "real" by
# size on disk, so a fixture part must be comfortably larger than the allowance.
_FAKE_WAV_BYTES = b"fake wav bytes " * (WAV_HEADER_ALLOWANCE // 16 + 2)

from src import activity
from src.diarize import DiarizationResult
from src.voiceprint import Voiceprint

MODEL = "pyannote/speaker-diarization-community-1"


@pytest.fixture(autouse=True)
def skip_the_reachability_probe():
    """process_file ยิงคำขอเล็ก ๆ ไปที่โมเดลก่อนเริ่มสรุป (ดู check_model_reachable)
    เทสต์ในไฟล์นี้ไม่ได้เกี่ยวกับเรื่องนั้นเลย และหลายตัวจงใจใช้ model id ปลอมที่
    resolve() ไม่รู้จัก -- ปิดไว้ที่เดียวแทนการเติม patch ให้ทั้ง 45 จุด

    เทสต์ที่ตรวจตัว probe เองอยู่ที่ท้ายไฟล์นี้ และปิด fixture นี้ด้วยการ patch ซ้อน
    """
    with patch("src.pipeline.check_model_reachable"):
        yield


def _diarization(turns=None) -> DiarizationResult:
    """ผลแยกผู้พูดปลอมในรูปที่ diarize_audio ของจริงคืนมา

    DiarizationResult ไม่มี field `embeddings` แล้ว (ดู src/diarize.py) -- centroid ของ
    diarization ถูกแทนที่ด้วย voiceprint จาก extract_voiceprints ทั้งหมด (ดู
    _stub_voiceprints ด้านล่างสำหรับเทสต์ที่ต้องปลอมเวกเตอร์การจับคู่)
    """
    return DiarizationResult(turns=turns or [])


class _FakeEmbedder:
    """embedder ปลอมสำหรับเทสต์ที่ mock extract_voiceprints ไปแล้ว -- ไม่เคยถูกเรียกจริง

    checkpoint คงที่เท่ากับ DEFAULT_EMBEDDING_MODEL (ค่าเริ่มต้นของ config.embedding_model
    ใน make_config) เพื่อให้ registry ที่สร้างด้วย _registry_with ผูกอยู่ในพื้นที่เวกเตอร์
    เดียวกัน -- speakers.match_known ข้ามตัวอย่างที่ป้าย embedding_model ไม่ตรงทั้งหมด
    """

    checkpoint = DEFAULT_EMBEDDING_MODEL

    def __call__(self, waveform, intervals):
        return [[1.0, 0.0] for _ in intervals]


def _mock_load_embedder():
    """แทน load_embedder ตัวจริง (โหลดโมเดลจาก HF จริง) ด้วย _FakeEmbedder

    เทสต์ในไฟล์นี้ไม่ได้ตั้งใจวัดพฤติกรรมของ pyannote/wespeaker ตัวจริง แค่ต้องการ
    embedder ที่มี .checkpoint ให้ _match_known_speakers ใช้ -- โหลดของจริงทุกเทสต์จะ
    ผูกชุดเทสต์นี้ไว้กับโมเดลที่ต้องอยู่ในแคช HF ของเครื่องที่รันอยู่โดยไม่จำเป็น
    """
    return patch("src.pipeline.load_embedder", return_value=_FakeEmbedder())


def _stub_voiceprints(mapping: dict[str, list[float]]):
    """แทน extract_voiceprints ด้วย Voiceprint ปลอมตาม mapping ที่ให้

    ทดสอบการจับคู่ผู้พูด (match_known) โดยไม่ต้องพึ่งเสียงจริงหรือโมเดล embedding จริง --
    แนวเดียวกับ tests/test_enroll.py: _stub_extract_voiceprints เวลาพูด (seconds) ตั้งเป็น
    30.0 ให้พอเกิน MIN_SPEAKING_SECONDS (10.0) เสมอ ไม่ให้ด่านนั้นมากวนเทสต์ที่ไม่ได้ตั้งใจ
    ทดสอบมัน
    """
    return patch(
        "src.pipeline.extract_voiceprints",
        return_value={
            label: Voiceprint(embedding=vector, seconds=30.0, segment_count=1)
            for label, vector in mapping.items()
        },
    )


def make_config(tmp_path: Path) -> Config:
    return Config(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
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
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
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
    # บรรทัดผู้เข้าร่วมถูกเติมจากการนับ transcript ไม่ใช่จากที่โมเดลเขียน
    assert summary == (
        "ผู้เข้าร่วม: 1 เสียง (ระบุตัวได้จากลายเสียง 0 คน) — ผู้พูด 1 100.0%\n\n"
        "## ประเด็นสำคัญ\n- ทดสอบ\n"
    )
    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert meta.startswith(
        f"สรุปด้วย {config.claude_model}\nประเภทประชุม: {config.meeting_profile}\n"
    )
    assert "## ผู้พูดใน transcript" in meta
    assert (meeting_dir / "weekly-standup.mp3").exists()
    assert not audio_path.exists()


def test_glossary_corrects_the_transcript_before_it_reaches_the_summarizer(tmp_path):
    """apply_exact ต้องทำงานที่ pipeline ก่อนส่งเข้า summarize -- และไฟล์ transcript.md
    ที่เก็บไว้ต้องเป็นของดิบ ไม่ถูกแก้ เพราะถ้า glossary ผิด คนอ่านต้องย้อนดูได้ว่า
    เดิมพูดว่าอะไร"""
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    (tmp_path / "glossary.md").write_text(
        "## exact\nPostgreSQL: โพสเกรส\n\n## fuzzy\nElectron: อิเล็กตรอน\n",
        encoding="utf-8",
    )
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[
                {"start": 0.0, "end": 2.0, "text": "โพสเกรสกับอิเล็กตรอนพร้อมแล้ว"}
            ],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch(
            "src.pipeline.summarize_transcript", return_value="## สรุป"
        ) as summarize_mock,
    ):
        meeting_dir = process_file(audio_path, config)

    sent_to_model = summarize_mock.call_args.args[0]
    assert "PostgreSQL" in sent_to_model
    assert "โพสเกรส" not in sent_to_model
    # fuzzy ไม่ถูกแทนที่ในโค้ด โมเดลเป็นคนตีความ จึงต้องยังอยู่ในข้อความเดิม
    assert "อิเล็กตรอน" in sent_to_model

    raw = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "โพสเกรส" in raw, "transcript ดิบต้องไม่ถูกแก้"

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert "แก้คำตาม glossary: PostgreSQL 1 จุด" in meta
    assert "คำ fuzzy ที่เจอในห้อง: Electron 1 ครั้ง" in meta


def _run_with_profile(tmp_path, job_profile=None, env_profile=None):
    """รัน process_file หนึ่งครั้งแล้วคืน kwargs ที่ summarize_transcript ได้รับ"""
    config = make_config(tmp_path)
    if env_profile is not None:
        config.meeting_profile = env_profile
    config.inbox_dir.mkdir(parents=True)
    (tmp_path / "glossary.md").write_text(
        "## fuzzy\nElectron: อิเล็กตรอน\n\n"
        "## ambiguous\nเสร็จ | Business = demo ได้ | dev = merge แล้ว\n",
        encoding="utf-8",
    )
    (tmp_path / "teams.md").write_text("Business: สมชาย\ndev: บอล\n", encoding="utf-8")
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    if job_profile is not None:
        write_job(config.inbox_dir, "weekly-standup", config.claude_model, profile=job_profile)

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "เสร็จแล้วครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch(
            "src.pipeline.summarize_transcript", return_value="## สรุป"
        ) as summarize_mock,
    ):
        meeting_dir = process_file(audio_path, config)

    return summarize_mock.call_args.kwargs, meeting_dir


def test_the_cross_profile_reaches_both_the_prompt_and_the_glossary(tmp_path):
    """สองเส้นต้องต่อพร้อมกัน ถ้าต่อแค่เส้นแรก ประชุมข้ามฝ่ายจะได้กฎ cross
    แต่ไม่ได้ตาราง ambiguous/teams ที่กฎนั้นสั่งให้ไปดู"""
    kwargs, _ = _run_with_profile(tmp_path, job_profile="cross")

    assert kwargs["profile"] == "cross"
    glossary_text = kwargs["glossary_text"]
    assert "เสร็จ" in glossary_text, "ตาราง ambiguous ต้องเข้า prompt"
    assert "สมชาย" in glossary_text, "ตาราง teams ต้องเข้า prompt"


def test_the_dev_profile_keeps_the_cross_team_tables_out(tmp_path):
    kwargs, _ = _run_with_profile(tmp_path, job_profile="dev")

    assert kwargs["profile"] == "dev"
    glossary_text = kwargs["glossary_text"]
    assert "Electron" in glossary_text, "ตาราง fuzzy ยังต้องเข้าตามปกติ"
    assert "เสร็จ" not in glossary_text
    assert "สมชาย" not in glossary_text


def test_a_job_file_without_a_profile_falls_back_to_the_env_value(tmp_path):
    """ไฟล์ที่ลากใส่ inbox/ เอง และ .job.json เก่าที่ไม่มี profile"""
    kwargs, _ = _run_with_profile(tmp_path, job_profile=None, env_profile="cross")

    assert kwargs["profile"] == "cross"
    assert "สมชาย" in kwargs["glossary_text"]


def test_the_job_file_profile_beats_the_env_value(tmp_path):
    """คิวข้ามวันได้ ผู้ใช้แก้ .env ระหว่างนั้นแล้วประชุมที่ค้างอยู่ต้องยังใช้ค่าที่
    เลือกไว้ตอนอัด -- เหมือนที่ model ทำอยู่"""
    kwargs, _ = _run_with_profile(tmp_path, job_profile="dev", env_profile="cross")

    assert kwargs["profile"] == "dev"
    assert "สมชาย" not in kwargs["glossary_text"]


def test_the_meeting_profile_is_recorded_in_the_summary_meta(tmp_path):
    _, meeting_dir = _run_with_profile(tmp_path, job_profile="cross")

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert "ประเภทประชุม: cross" in meta


def _previous_summary(config, name, profile, open_items):
    d = config.meetings_dir / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.md").write_text(  # noqa: E501 - โครงสรุปย่อที่มีแค่ส่วนที่ carryover อ่าน
        f"## ตกลงแล้ว\n- x\n\n## ต้องคุยต่อครั้งหน้า\n{open_items}\n"
        f"\n---\nสรุปด้วย GLM-5.2\nประเภทประชุม: {profile}\n",
        encoding="utf-8",
    )
    return d


def test_open_items_from_the_previous_meeting_reach_the_summarizer(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    _previous_summary(config, "2026-07-20_09-00-standup", "dev", "- เรื่องค้างเมื่อวาน")
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดี"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป") as summarize,
    ):
        process_file(audio_path, config)

    carryover = summarize.call_args.kwargs["carryover_text"]
    assert "- เรื่องค้างเมื่อวาน" in carryover
    assert "## คืบหน้าจากครั้งก่อน" in carryover


def test_carryover_only_comes_from_the_same_profile(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    _previous_summary(config, "2026-07-26_09-00-crossteam", "cross", "- ของข้ามฝ่าย")
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดี"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป") as summarize,
    ):
        process_file(audio_path, config)

    assert summarize.call_args.kwargs["carryover_text"] == ""


def test_carryover_can_be_switched_off(tmp_path):
    config = make_config(tmp_path)
    config.carryover_enabled = False
    config.inbox_dir.mkdir(parents=True)
    _previous_summary(config, "2026-07-20_09-00-standup", "dev", "- เรื่องค้างเมื่อวาน")
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดี"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป") as summarize,
    ):
        process_file(audio_path, config)

    assert summarize.call_args.kwargs["carryover_text"] == ""


def test_a_retry_does_not_carry_over_its_own_previous_summary(tmp_path):
    """ลาก .job.json กลับ inbox แล้วรันซ้ำ -- รอบนี้ใช้ transcript เดิมและโฟลเดอร์เดิม
    ที่มี summary.md จากรอบก่อนอยู่แล้ว ถ้าไม่กันไว้ ประชุมจะยกเรื่องค้างของตัวเองมา
    เป็น carryover แล้วเขียน "คืบหน้าจากครั้งก่อน" ทับเรื่องเดียวกันวนไปเรื่อยๆ

    ต้องตั้ง .job.json ให้ชี้ transcript เดิมด้วย ไม่ใช่แค่สร้างโฟลเดอร์ชื่อซ้ำ --
    ชื่อซ้ำเฉยๆ process_file จะสร้างโฟลเดอร์ใหม่ลงท้าย -2 ซึ่งเป็นคนละเคสกัน
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    transcript_path = _saved_transcript(
        config, "2026-07-25_09-00-weekly-standup", "# Transcript\n\nของเดิม"
    )
    record_transcript(audio_path, transcript_path)
    own_dir = _previous_summary(
        config, "2026-07-25_09-00-weekly-standup", "dev", "- เรื่องค้างของตัวเอง"
    )
    assert (own_dir / "summary.md").exists(), "sanity: สรุปรอบก่อนต้องอยู่ในโฟลเดอร์เดิม"

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดี"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป") as summarize,
    ):
        process_file(audio_path, config)

    assert "เรื่องค้างของตัวเอง" not in summarize.call_args.kwargs["carryover_text"]


def test_glossary_reaches_the_summarizer_as_prompt_text(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    (tmp_path / "glossary.md").write_text(
        "## fuzzy\nElectron: อิเล็กตรอน\n", encoding="utf-8"
    )
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "อิเล็กตรอนพร้อม"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch(
            "src.pipeline.summarize_transcript", return_value="## สรุป"
        ) as summarize_mock,
    ):
        process_file(audio_path, config)

    glossary_text = summarize_mock.call_args.kwargs["glossary_text"]
    assert "Electron" in glossary_text
    assert "อิเล็กตรอน" in glossary_text


def test_a_missing_glossary_file_adds_no_glossary_lines(tmp_path):
    """ไม่มี glossary.md = ห้าม crash และห้ามมีบรรทัดเรื่องคำศัพท์โผล่มาใน summary.meta.md
    (บรรทัด "ประเภทประชุม" ไม่เกี่ยวกับ glossary มันมีเสมอเพื่อให้ย้อนดูได้ว่า
    สรุปนี้ใช้กฎชุดไหน)"""
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)

    summary = (meeting_dir / "summary.md").read_text(encoding="utf-8")
    assert summary.endswith("## สรุป\n")
    assert "glossary" not in summary
    assert "fuzzy" not in summary

    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    # ตารางผู้พูดก็ไม่เกี่ยวกับ glossary เหมือนบรรทัด "ประเภทประชุม" -- มันนับจาก
    # transcript จึงมีเสมอ ไม่ว่าจะมีไฟล์ glossary.md หรือไม่
    assert meta.startswith(
        f"สรุปด้วย {config.claude_model}\n"
        f"ประเภทประชุม: {config.meeting_profile}\n"
    )
    assert "glossary" not in meta
    assert "fuzzy" not in meta


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
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
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


def test_process_file_moves_to_failed_when_the_configured_model_is_not_registered(tmp_path):
    """ล็อกพฤติกรรมของ regression ที่พบจาก review: make_config ตั้ง claude_model เป็น
    "claude-opus-4-8" ซึ่งไม่อยู่ใน src.llm.PROVIDERS -- เดิม CLAUDE_MODEL ถูกส่งตรงเข้า
    Anthropic API โดยไม่ผ่าน registry นี้เลย ตอนนี้ summarize_transcript เรียกผ่าน
    resolve() ก่อนเสมอ id ที่ resolve ไม่ได้จึงโยน UnknownModelError กลางท่อ -- หลังถอด
    เสียงเสร็จไปแล้ว (ขั้นที่แพงที่สุด) ไม่ได้ mock summarize_transcript ในเทสต์นี้โดย
    ตั้งใจ เพื่อให้ resolve() ตัวจริงทำงาน แต่ resolve() ล้มก่อนมี network call ใด ๆ
    เกิดขึ้น จึงยังไม่แตะเน็ตเวิร์กจริง"""
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        pytest.raises(UnknownModelError),
    ):
        process_file(audio_path, config)

    assert not audio_path.exists()
    assert (config.failed_dir / "weekly-standup.mp3").exists()
    error_log = (config.failed_dir / "weekly-standup.error.log").read_text(encoding="utf-8")
    assert "Summarization failed" in error_log
    assert config.claude_model in error_log


def test_process_file_keeps_the_transcript_when_summarization_fails(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "broken.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
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
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
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
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
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
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
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
        return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}])
    )

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
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
        _mock_load_embedder(),
        patch("src.pipeline.transcribe_audio", mock_transcribe),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config, whisper_model=sentinel_model)

    assert mock_transcribe.call_args.kwargs["model"] is sentinel_model
    assert mock_transcribe.call_args.kwargs["model_size"] == config.whisper_model


def test_process_file_matches_with_the_embedder_checkpoint_not_the_config_value(
    tmp_path, monkeypatch
):
    # ป้ายที่ใช้เทียบต้องมาจากตัวที่คำนวณจริง ไม่ใช่ค่าใน .env ตอนนั้น -- สองอย่างนี้ต่างกัน
    # ได้เมื่อผู้ใช้แก้ .env ระหว่างที่ watcher ถือโมเดลเก่าค้างอยู่ในหน่วยความจำ
    seen = {}

    def fake_match_known(embeddings, speakers, high, low, *, embedding_model):
        seen["embedding_model"] = embedding_model
        return {}

    monkeypatch.setattr("src.pipeline.match_known", fake_match_known)

    config = make_config(tmp_path)
    # ต้องไม่ใช่ค่าที่ไปถึง match_known -- ถ้า pipeline สลับไปอ่าน config.embedding_model
    # แทน embedder.checkpoint จริง เทสต์นี้ต้องจับได้ทันที
    config.embedding_model = "config-value-that-must-not-be-used"
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config, embedder=_FakeEmbedder())

    assert seen["embedding_model"] == "pyannote/wespeaker-voxceleb-resnet34-LM"
    assert seen["embedding_model"] != config.embedding_model


def test_process_file_keeps_the_speaker_turns_when_voiceprints_fail(tmp_path, monkeypatch):
    # กฎเดิมของ repo: ความล้มเหลวของ "การจำเสียง" ต้องไม่ทำลาย "การแยกผู้พูด" ของประชุมที่
    # อัดซ้ำไม่ได้ การป้องกันนี้เคยอยู่ใน diarize._speaker_embeddings ซึ่งถูกลบไปแล้ว --
    # ต้องพิสูจน์ว่ามันย้ายมาอยู่กับตัวใหม่จริง ไม่ใช่หายไปพร้อมของเก่า
    monkeypatch.setattr(
        "src.pipeline.extract_voiceprints",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[
                {"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"},
                {"start": 2.0, "end": 4.0, "text": "สวัสดีค่ะ"},
            ],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization(
                [
                    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                    {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
                ]
            ),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config, embedder=_FakeEmbedder())

    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "ผู้พูด 2" in transcript  # ยังแยกผู้พูดได้ แค่จำเสียงไม่ได้


def test_process_file_completes_when_embedder_has_no_checkpoint_attribute(tmp_path):
    """embedder ที่สนองสัญญา embed(waveform, intervals) -> list ของ extract_voiceprints แต่
    ไม่มี .checkpoint ต้องไม่ทำให้ process_file ล้ม -- docstring ของ process_file ประกาศ
    พารามิเตอร์ embedder เป็น Any และผูกไว้แค่สัญญาของ extract_voiceprints เท่านั้น ไม่มี
    อะไรบังคับว่าต้องมี .checkpoint ผู้เรียกที่ทำตามสัญญานั้นแต่ไม่มี attribute นี้ต้องไม่เจอ
    AttributeError ก่อนถึง create_meeting_folder/save_transcript -- ไม่งั้นการถอดเสียงที่
    เสร็จไปแล้วทั้งรอบถูกทิ้งและไฟล์เสียงค้างใน inbox/ ทุกรอบ poll ไปเรื่อย ๆ
    """

    def bare_embedder(waveform, intervals):
        return [[1.0, 0.0] for _ in intervals]

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config, embedder=bare_embedder)

    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert transcript  # completed instead of raising AttributeError on .checkpoint


def test_process_file_treats_a_blank_checkpoint_as_a_voiceprint_failure(tmp_path):
    """embedder ที่มี .checkpoint แต่เป็นค่าว่างเปล่าต้องถูกปฏิบัติเหมือน voiceprint ล้มเหลว

    Minor 1 ของรีวิวรอบสอง: getattr(embedder, "checkpoint", "") กันแค่กรณีไม่มี
    attribute เลย (เทสต์ข้างบน) ไม่กันกรณีที่ attribute มีอยู่จริงแต่ว่างเปล่า --
    embedding_model="" หลุดผ่าน _match_known_speakers ไปได้เพราะ voiceprints ที่ได้มา
    ไม่ว่าง ด่าน `if not voiceprints` จึงไม่จับ แล้ว match_known validate ค่าว่างแล้ว
    raise ValueError ซึ่งถูก try ของ _match_known_speakers กลืนเงียบ ๆ -- แต่
    embedding_model ว่างเปล่ายังหลุดรอดต่อไปถึง _record_pending_speakers และถูกเขียน
    ลง speakers/pending/<meeting>.json ทำให้คนที่มายืนยันชื่อทีหลังเจอ
    speakers.add_sample raise ValueError ถาวร แก้ไม่ได้เลย
    """
    from src.pending import load_all_pending

    class _BlankCheckpointEmbedder:
        checkpoint = ""

        def __call__(self, waveform, intervals):
            return [[1.0, 0.0] for _ in intervals]

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[
                {"start": 0.0, "end": 30.0, "text": "สวัสดีครับ ผมขอเริ่มเลยนะครับ"}
            ],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization(
                [{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}]
            ),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(
            audio_path, config, embedder=_BlankCheckpointEmbedder()
        )

    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "ผู้พูด 1" in transcript  # ลดขั้นเป็นป้ายทั่วไป ไม่มีชื่อที่ยืนยันแล้ว
    # ไม่มีไฟล์คิวรอตั้งชื่อสำหรับการประชุมนี้เลย: voiceprints ที่มีต้องถูกล้างทิ้งไปพร้อม
    # กับ embedding_model ว่างเปล่า ไม่งั้น pending.json จะติด embedding_model="" ซึ่ง
    # speakers.add_sample ปฏิเสธถาวร
    assert load_all_pending(tmp_path) == []


def test_process_file_treats_any_non_string_checkpoint_as_a_voiceprint_failure(
    tmp_path, monkeypatch
):
    """embedder.checkpoint ที่ไม่ใช่สตริงเลยต้องถูกปฏิบัติเหมือน voiceprint ล้มเหลว เช่นเดียวกับ
    ค่าว่างเปล่า -- ไม่ใช่แค่กรณีเดียวที่รีวิวเจอ

    รีวิวรอบสาม: เช็คของรอบสอง (`if not embedding_model or not embedding_model.strip()`)
    เรียก .strip() โดยไม่เช็ค type ก่อน -- checkpoint ที่ truthy แต่ไม่ใช่สตริง (int, list, ...)
    ทำให้ .strip() ยิง AttributeError หลุดไปถึง except Exception ตัวนอกของทั้งบล็อก ซึ่ง
    เขียน log voiceprint_failed เหมือนกันแต่ "ไม่" ล้าง voiceprints -- ผลคือ pending.json
    ติด embedding_model ที่พังแบบเดียวกับ Minor 1 รีวิวรอบสอง เพียงแต่มาจากชนิดข้อมูลผิด
    แทนที่จะเป็นสตริงว่างเปล่า

    ครอบคลุมทั้งสี่แบบในเทสต์เดียว (ไม่มีแบบ parametrize ในไฟล์นี้มาก่อน จึงใช้ลูปแทน):
    int, None, list ที่ truthy แต่ไม่ใช่สตริงเลย และ ""/"   " ที่ว่างเปล่าซึ่งรอบสองแก้ไว้
    แล้ว -- เอามาซ้ำในนี้เพื่อยืนยันว่าเช็คใหม่ (isinstance ก่อน) ยังกันกรณีเดิมได้เหมือนเดิม
    """
    from src.pending import load_all_pending

    class _FixedCheckpointEmbedder:
        def __init__(self, checkpoint):
            self.checkpoint = checkpoint

        def __call__(self, waveform, intervals):
            return [[1.0, 0.0] for _ in intervals]

    bad_checkpoints = [7, None, ["x"], "", "   "]

    for index, checkpoint in enumerate(bad_checkpoints):
        case_dir = tmp_path / f"case-{index}"
        config = make_config(case_dir)
        config.inbox_dir.mkdir(parents=True)
        audio_path = config.inbox_dir / "weekly-standup.mp3"
        audio_path.write_bytes(b"fake audio")

        match_known_calls = []

        def fake_match_known(embeddings, speakers, high, low, *, embedding_model):
            match_known_calls.append(embedding_model)
            return {}

        monkeypatch.setattr("src.pipeline.match_known", fake_match_known)

        with (
            _mock_convert_to_wav(),
            _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
            patch(
                "src.pipeline.transcribe_audio",
                return_value=[
                    {"start": 0.0, "end": 30.0, "text": "สวัสดีครับ ผมขอเริ่มเลยนะครับ"}
                ],
            ),
            patch(
                "src.pipeline.diarize_audio",
                return_value=_diarization(
                    [{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}]
                ),
            ),
            patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
        ):
            meeting_dir = process_file(
                audio_path, config, embedder=_FixedCheckpointEmbedder(checkpoint)
            )

        transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
        assert "ผู้พูด 1" in transcript, (
            f"checkpoint={checkpoint!r} ต้องลดขั้นเป็นป้ายทั่วไป ไม่ใช่ทำให้พังทั้งงาน"
        )
        assert load_all_pending(case_dir) == [], (
            f"checkpoint={checkpoint!r} ต้องไม่ทิ้งไฟล์คิวรอตั้งชื่อที่มี embedding_model พัง"
        )
        assert match_known_calls == [], (
            f"checkpoint={checkpoint!r} ต้องไม่หลุดไปถึง match_known เลย"
        )


def test_process_file_clears_voiceprints_before_logging_an_unusable_checkpoint(
    tmp_path,
):
    """แม้ activity.append ระหว่าง log checkpoint ที่ใช้ไม่ได้จะพังเอง (ไม่ใช่ OSError)
    voiceprints ก็ต้องถูกล้างไปแล้วก่อนหน้านั้น ไม่ใช่หลัง

    รีวิวรอบสาม: บล็อกป้องกัน checkpoint ที่ใช้ไม่ได้เรียง log ก่อนล้าง voiceprints --
    activity.append กลืนเฉพาะ OSError ไว้ข้างใน (ดู src/activity.py) exception ชนิดอื่นที่
    หลุดออกมาจากมันจะทำให้ voiceprints ที่มี checkpoint เสียยังไม่ถูกล้าง แล้วเดินทางต่อไป
    ถึง _record_pending_speakers เหมือนเดิม -- ต้องล้างก่อนจึงจะปลอดภัยไม่ว่า logging จะพัง
    ด้วยอะไรก็ตาม

    จำลอง "dependency ที่พัง" แบบเดียวกับที่ไฟล์นี้ใช้ที่อื่น (เช่น
    `patch("src.pipeline.load_registry", side_effect=OSError(...))`) แต่ตัวที่ต้องพังคือ
    activity.append เอง และต้องไม่ใช่ OSError -- patch เฉพาะ event "voiceprint_failed" และ
    แค่ครั้งแรกที่มันถูกเรียก แล้วปล่อยให้ของจริงทำงานหลังจากนั้น ไม่งั้น activity.append
    ตัวอื่น ๆ ที่ process_file เรียกตลอดทั้งรอบ (queued, diarize_started, ...) จะพังไปด้วย
    ทั้งที่ไม่เกี่ยวกับสิ่งที่กำลังทดสอบ
    """
    from src.pending import load_all_pending

    real_append = activity.append
    raised = {"done": False}

    def flaky_append(base_dir, job, code, level="info", params=None):
        if code == "voiceprint_failed" and not raised["done"]:
            raised["done"] = True
            raise RuntimeError("บันทึก activity ล้มเหลว แบบไม่ใช่ OSError โดยตั้งใจ")
        return real_append(base_dir, job, code, level, params)

    class _BlankCheckpointEmbedder:
        checkpoint = ""

        def __call__(self, waveform, intervals):
            return [[1.0, 0.0] for _ in intervals]

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        patch("src.pipeline.activity.append", side_effect=flaky_append),
        _mock_convert_to_wav(),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[
                {"start": 0.0, "end": 30.0, "text": "สวัสดีครับ ผมขอเริ่มเลยนะครับ"}
            ],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization(
                [{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}]
            ),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(
            audio_path, config, embedder=_BlankCheckpointEmbedder()
        )

    assert raised["done"]  # ยืนยันว่า fault injection ทำงานจริง ไม่ใช่แค่ผ่านเงียบ ๆ
    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "ผู้พูด 1" in transcript
    assert load_all_pending(tmp_path) == [], (
        "voiceprints ต้องถูกล้างก่อน log เสมอ ไม่ว่า activity.append จะพังด้วย exception "
        "ชนิดไหนก็ตาม"
    )


def test_process_file_clears_state_when_reading_checkpoint_raises(tmp_path, monkeypatch):
    """embedder.checkpoint ที่เป็น property ซึ่ง raise เวลาอ่าน (ไม่ใช่แค่ค่าผิดชนิด/ว่างเปล่า)
    ต้องถูกปฏิบัติเหมือน voiceprint ล้มเหลวเช่นกัน -- ล้าง voiceprints/embedding_model ทิ้ง

    รีวิวรอบสี่: `getattr(embedder, "checkpoint", "")` ที่ pipeline.py:271 กันได้แค่
    AttributeError เท่านั้น embedder ที่ .checkpoint เป็น property แล้ว raise ด้วย
    exception ชนิดอื่น (เทสต์นี้ใช้ RuntimeError) หลุดผ่าน getattr ไปถึง except Exception
    ตัวนอกสุดของบล็อกที่คำนวณ voiceprints (เดิมอยู่ที่บรรทัด ~302) ซึ่ง log
    voiceprint_failed เหมือนทุกเทสต์ข้างบน แต่เดิม "ไม่" ล้าง voiceprints/embedding_model
    เลย -- ผลคือ match_known ถูกเรียกด้วย embedding_model="" และ/หรือ pending.json ติด
    embedding_model="" แบบเดียวกับ Minor 1 รีวิวรอบสอง เพียงแต่ทางเข้าคือ exception จาก
    การอ่าน .checkpoint เอง ไม่ใช่ค่าที่อ่านได้แต่ผิดชนิด/ว่างเปล่า
    """
    from src.pending import load_all_pending

    class _RaisingCheckpointEmbedder:
        def __call__(self, waveform, intervals):
            return [[1.0, 0.0] for _ in intervals]

        @property
        def checkpoint(self):
            raise RuntimeError("checkpoint พังตอนอ่าน แบบไม่ใช่ AttributeError โดยตั้งใจ")

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    match_known_calls = []

    def fake_match_known(embeddings, speakers, high, low, *, embedding_model):
        match_known_calls.append(embedding_model)
        return {}

    monkeypatch.setattr("src.pipeline.match_known", fake_match_known)

    with (
        _mock_convert_to_wav(),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[
                {"start": 0.0, "end": 30.0, "text": "สวัสดีครับ ผมขอเริ่มเลยนะครับ"}
            ],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization(
                [{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}]
            ),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(
            audio_path, config, embedder=_RaisingCheckpointEmbedder()
        )

    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "ผู้พูด 1" in transcript  # การประชุมยังจบสมบูรณ์ ไม่ใช่ทั้งงานพัง
    assert match_known_calls == [], (
        "checkpoint ที่ raise ตอนอ่านต้องไม่หลุดไปถึง match_known เลย"
    )
    assert load_all_pending(tmp_path) == [], (
        "checkpoint ที่ raise ตอนอ่านต้องไม่ทิ้งไฟล์คิวรอตั้งชื่อที่มี embedding_model พัง"
    )


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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        process_file(audio_path, config)

    assert summarize.call_args.kwargs["model"] == config.claude_model


@pytest.mark.parametrize("enabled", [True, False])
def test_the_merge_switch_from_config_reaches_the_summarizer(tmp_path, enabled):
    """สวิตช์ใน .env ต้องเดินทางถึงที่ที่มันทำงานจริง ไม่ใช่แค่ถูกอ่านเข้ามาใน Config

    การรวมบล็อกทำอยู่ใน summarize_transcript ไม่ใช่ที่นี่ เพราะมันต้องเกิดก่อนการตัดสินใจ
    ว่าจะยิงรอบเดียวหรือหั่น chunk -- pipeline จึงมีหน้าที่แค่ส่งค่าต่อ
    """
    config = make_config(tmp_path)
    config.merge_speaker_turns = enabled
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    summarize = MagicMock(return_value="## สรุป")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        process_file(audio_path, config)

    assert summarize.call_args.kwargs["merge_turns"] is enabled


def test_summarize_is_called_without_an_api_key(tmp_path):
    """key เป็นเรื่องของ provider -- pipeline ต้องไม่ส่ง key ของ Anthropic เข้าไปใน
    เส้นทางที่อาจไปจบที่ provider อื่น"""
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    summarize = MagicMock(return_value="## สรุป")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        process_file(audio_path, config)

    assert "api_key" not in summarize.call_args.kwargs


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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", side_effect=RuntimeError("boom")),
        pytest.raises(RuntimeError),
    ):
        process_file(audio_path, config)

    assert not (config.inbox_dir / f"weekly-standup{JOB_SUFFIX}").exists()
    assert read_model(config.failed_dir / "weekly-standup.mp3") == "claude-sonnet-5"


def test_the_recorded_model_choice_survives_from_manifest_to_summary(tmp_path):
    # The feature's actual premise: the model the user picked at record time
    # (written into the session manifest) must survive session -> job sidecar ->
    # pipeline -> summarize_transcript call -> summary.meta.md, with the sidecar
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
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        meeting_dir = process_file(audio_path, config)

    assert summarize.call_args.kwargs["model"] == "claude-sonnet-5"
    meta = (meeting_dir / "summary.meta.md").read_text(encoding="utf-8")
    assert "สรุปด้วย claude-sonnet-5\n" in meta
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
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


def test_conditioning_on_previous_text_follows_config(tmp_path):
    """ค่าจาก .env ต้องไปถึง decoder จริง ไม่ใช่ค้างอยู่ใน Config

    ปุ่มนี้เป็นตัวตัดวงวนซ้ำคำที่ทำให้ transcript ทั้งไฟล์กลายเป็นประโยคเดียววนซ้ำ
    ถ้ามันไม่ถูกส่งต่อ ค่าที่ตั้งใน .env จะดูเหมือนมีผลแต่ไม่มีอะไรเกิดขึ้นเลย

    ตั้งเป็น True โดยเจตนา ทั้งที่ค่าที่ระบบส่งจริงคือ False -- ทดสอบด้วยค่าเริ่มต้น
    จะผ่านแม้ pipeline จะ hardcode False ทิ้งไว้ ซึ่งไม่ได้พิสูจน์อะไรเลย
    """
    config = make_config(tmp_path)
    config.whisper_condition_on_previous_text = True
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ) as transcribe_mock,
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    kwargs = transcribe_mock.call_args.kwargs
    assert kwargs["condition_on_previous_text"] is True


def test_hotwords_are_not_sent_while_the_switch_is_off(tmp_path):
    """glossary.md ที่มีคำอยู่ต้องไม่ทำให้ decoder ถูก bias เองโดยไม่มีใครสั่ง

    ตารางนี้มีไว้ป้อนขั้นสรุปมาตั้งแต่แรก เครื่องที่ปิดสวิตช์ต้องได้พฤติกรรมเดิมเป๊ะ
    แม้จะมี glossary.md อยู่ก็ตาม
    """
    config = make_config(tmp_path)
    config.whisper_hotwords = False
    config.inbox_dir.mkdir(parents=True)
    (tmp_path / "glossary.md").write_text(
        "## fuzzy\nElectron: อิเล็กตรอน\n", encoding="utf-8"
    )
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "Electron พร้อม"}],
        ) as transcribe_mock,
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    assert transcribe_mock.call_args.kwargs["hotwords"] is None


def test_glossary_reaches_the_transcriber_as_hotwords(tmp_path):
    """คำถูกต้องไปถึง decoder ตั้งแต่ตอนถอดเสียง ไม่ใช่รอ glossary กู้ทีหลัง

    apply_exact กู้ได้แค่ตัวสะกด คำที่ถอดผิดทำให้ Whisper ตัด segment ผิดตำแหน่ง
    ไปแล้ว และตัวเลขที่เพี้ยนตามคำผิดกู้ไม่ได้เลย
    """
    config = make_config(tmp_path)
    config.whisper_hotwords = True
    config.inbox_dir.mkdir(parents=True)
    (tmp_path / "glossary.md").write_text(
        "## fuzzy\nElectron: อิเล็กตรอน\n", encoding="utf-8"
    )
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "Electron พร้อม"}],
        ) as transcribe_mock,
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    assert transcribe_mock.call_args.kwargs["hotwords"] == "Electron"


def test_hotwords_still_reach_the_transcriber_in_transcript_only_mode(tmp_path):
    """โหมดนี้ไม่มีขั้นสรุป transcript จึงเป็นผลลัพธ์สุดท้ายที่ผู้ใช้ได้จริง

    glossary ตัวเดิมถูกโหลดในบล็อกที่ข้ามไปทั้งก้อนเมื่อ claude_model เป็น
    transcript-only เพราะมันมีไว้ป้อนขั้นสรุป การเอา hotwords ไปฝากไว้ในบล็อกนั้น
    จะทำให้โหมดที่ต้องการคุณภาพ transcript มากที่สุดเป็นโหมดเดียวที่ไม่ได้มัน
    """
    config = make_config(tmp_path)
    config.whisper_hotwords = True
    config.inbox_dir.mkdir(parents=True)
    (tmp_path / "glossary.md").write_text(
        "## fuzzy\nElectron: อิเล็กตรอน\n", encoding="utf-8"
    )
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(config.inbox_dir, "weekly-standup", NO_SUMMARY_MODEL)

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "Electron พร้อม"}],
        ) as transcribe_mock,
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    assert transcribe_mock.call_args.kwargs["hotwords"] == "Electron"


def test_a_missing_glossary_file_sends_no_hotwords(tmp_path):
    """ไม่มี glossary.md = ห้าม crash และห้ามส่งสตริงว่างเปล่าให้ decoder

    faster-whisper เปิดช่อง sot_prev ให้ทันทีที่ hotwords truthy สตริงว่างจึงเป็น
    การจ่าย token ทิ้งเปล่า ๆ ทุกหน้าต่าง 30 วินาที
    """
    config = make_config(tmp_path)
    config.whisper_hotwords = True
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ) as transcribe_mock,
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    assert transcribe_mock.call_args.kwargs["hotwords"] is None


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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        meeting_dir = process_file(audio_path, config)

    summarize.assert_not_called()
    assert (meeting_dir / "transcript.md").exists()
    assert not (meeting_dir / "summary.md").exists()
    assert not (inbox / f"weekly-standup{JOB_SUFFIX}").exists()


def test_transcript_only_can_come_from_the_config_default(tmp_path):
    # README บอกว่าตั้ง CLAUDE_MODEL=transcript-only ใน .env แล้วไฟล์ที่ลากใส่
    # inbox/ เองจะไม่ถูกสรุป -- ได้มาฟรีเพราะเช็คที่ค่าหลัง resolve ไม่ใช่ที่ job file
    config = make_config(tmp_path)
    config.claude_model = NO_SUMMARY_MODEL
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", summarize),
    ):
        meeting_dir = process_file(audio_path, config)

    summarize.assert_not_called()
    assert not (meeting_dir / "summary.md").exists()


# --- ความคืบหน้าที่ส่งให้หน้าจอ ------------------------------------------


def _stage_codes(config: Config) -> list[str]:
    from src.activity import tail

    return [e["code"] for e in tail(config.base_dir)]


def _activity_events(config: Config, code: str) -> list[dict]:
    """เหตุการณ์ทั้งหมดในบันทึกที่มี code ตรงกับที่ขอ พร้อม params/level ของมัน"""
    from src.activity import tail

    return [e for e in tail(config.base_dir) if e["code"] == code]


def test_process_file_records_each_stage(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    assert _stage_codes(config) == [
        "queued",
        "transcribe_started",
        "diarize_started",
        "summarize_started",
        "meeting_done",
    ]


def test_process_file_skips_the_summarize_event_in_transcript_only_mode(tmp_path):
    """หน้าจอชี้ขั้นจากเหตุการณ์ล่าสุด ถ้ามี summarize_started ในโหมดนี้มันจะค้าง
    ที่ "กำลังสรุป" ตลอดกาลสำหรับประชุมที่ไม่ได้สั่งสรุป"""
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(config.inbox_dir, "weekly-standup", NO_SUMMARY_MODEL)

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
    ):
        process_file(audio_path, config)

    codes = _stage_codes(config)
    assert "summarize_started" not in codes
    assert codes[-1] == "meeting_done"


def test_process_file_records_a_failure(tmp_path):
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch(
            "src.pipeline.summarize_transcript",
            side_effect=RuntimeError("Claude ล่ม"),
        ),
        pytest.raises(RuntimeError),
    ):
        process_file(audio_path, config)

    assert _stage_codes(config)[-1] == "job_failed"


def test_process_file_survives_an_activity_log_that_cannot_be_written(tmp_path):
    """การเขียนสถานะที่ล้มต้องไม่ทำให้ประชุมที่อัดซ้ำไม่ได้พังตามไปด้วย

    ยึดที่ state/ ด้วยไฟล์ธรรมดาเพื่อให้ mkdir ล้มจริง แทนการ patch activity.append
    -- ทดสอบการรับประกันของจริงทั้งเส้น ไม่ใช่สมมติฐานเกี่ยวกับมัน
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    (tmp_path / "state").write_text("ไม่ใช่โฟลเดอร์", encoding="utf-8")
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)

    assert (meeting_dir / "summary.md").exists()


def _registry_with(tmp_path, name, embedding, embedding_model=DEFAULT_EMBEDDING_MODEL):
    """คนหนึ่งคนในทะเบียนพร้อมตัวอย่างเสียงเดียว ติดป้าย embedding_model ให้ตรงกับ
    _FakeEmbedder.checkpoint โดยค่าเริ่มต้น -- ป้ายไม่ตรง = match_known ข้ามตัวอย่างนี้ไป
    เงียบ ๆ (ดู speakers.match_known)"""
    from src.speakers import add_sample, save_registry

    save_registry(
        tmp_path,
        add_sample(
            [],
            name,
            {"embedding": embedding, "embedding_model": embedding_model},
            source="ก่อนหน้า",
        ),
    )


def test_process_file_writes_a_known_speakers_real_name_into_the_transcript(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    _registry_with(tmp_path, "สมหญิง็ม", [1.0, 0.0])

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)

    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "**สมหญิง็ม** [00:00]: สวัสดีครับ" in transcript
    assert "ผู้พูด 1" not in transcript

    matched = _activity_events(config, "speakers_matched")
    assert len(matched) == 1
    assert matched[0]["params"] == {"count": 1}


def test_process_file_only_counts_confident_matches_in_speakers_matched(tmp_path):
    """recognized ใน speakers_matched ต้องนับเฉพาะคนที่ confident=True เท่านั้น

    match_known คืนคนที่คะแนนอยู่แค่ระหว่าง low กับ high มาด้วย (แค่ข้อเสนอให้เลือก
    ไม่ใช่คนที่ระบุตัวได้แน่นอน) ถ้า pipeline สลับไปนับ len(matches) แทน เทสต์นี้
    ต้องพังเพราะจะรายงาน 2 คนทั้งที่มั่นใจแค่คนเดียว
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    _registry_with(tmp_path, "สมหญิง็ม", [1.0, 0.0])

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        _stub_voiceprints(
            {
                "SPEAKER_00": [1.0, 0.0],  # cos = 1.0 -- confident
                "SPEAKER_01": [0.6, 0.8],  # cos = 0.6 -- แค่ข้อเสนอ ไม่ confident
            }
        ),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[
                {"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"},
                {"start": 2.0, "end": 4.0, "text": "สวัสดีค่ะ"},
            ],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization(
                [
                    {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
                    {"start": 2.0, "end": 4.0, "speaker": "SPEAKER_01"},
                ]
            ),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    matched = _activity_events(config, "speakers_matched")
    assert len(matched) == 1
    assert matched[0]["params"] == {"count": 1}


def test_process_file_keeps_the_anonymous_label_below_the_high_threshold(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    _registry_with(tmp_path, "สมหญิง็ม", [1.0, 0.0])

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        # cos = 0.6 -> ระหว่างเกณฑ์: เสนอได้ แต่ห้ามใส่ชื่อให้เอง
        _stub_voiceprints({"SPEAKER_00": [0.6, 0.8]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)

    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "**ผู้พูด 1** [00:00]: สวัสดีครับ" in transcript
    assert "สมหญิง็ม" not in transcript


def test_process_file_finishes_the_meeting_when_the_registry_cannot_be_read(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.load_registry", side_effect=OSError("ดิสก์พัง")),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)

    # การจำเสียงพังต้องไม่ทำให้ประชุมที่อัดซ้ำไม่ได้พังตาม
    assert (meeting_dir / "transcript.md").exists()
    assert (meeting_dir / "summary.md").exists()

    failed = _activity_events(config, "speakers_failed")
    assert len(failed) == 1
    assert failed[0]["level"] == "warn"


@contextmanager
def _two_speaker_diarization_patched():
    """สอง speaker ที่พูดยาวกว่า MIN_SPEAKING_SECONDS มาก พร้อมเวกเตอร์ที่ใช้ได้

    แยกออกมาจาก _run_with_two_speakers เพื่อให้เทสต์ที่ *ไม่* อยาก mock
    guess_speaker_names (เช่นเทสต์โหมด transcript-only ที่ต้องให้ตัวจริงทำงาน)
    ยืมชุดพากย์เสียง/ถอดเสียงนี้ได้โดยไม่ต้องคัดลอกทั้งก้อน
    """
    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[
                {"start": 0.0, "end": 30.0, "text": "สวัสดีครับ ผมขอเริ่มเลยนะครับ"},
                {"start": 30.0, "end": 70.0, "text": "ครับผม ผมเห็นด้วยกับที่ว่ามา"},
            ],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization(
                [
                    {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"},
                    {"start": 30.0, "end": 70.0, "speaker": "SPEAKER_01"},
                ]
            ),
        ),
    ):
        yield


def _run_with_two_speakers(config, guess=None, guess_error=None):
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    guess_mock = MagicMock(
        return_value=guess or {}, side_effect=guess_error
    )
    with (
        _two_speaker_diarization_patched(),
        patch("src.pipeline.guess_speaker_names", guess_mock),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)
    return meeting_dir, guess_mock


def test_process_file_queues_unknown_speakers_for_naming(tmp_path):
    from src.pending import load_all_pending

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)

    meeting_dir, _ = _run_with_two_speakers(config)

    pending = load_all_pending(tmp_path)
    assert len(pending) == 1
    assert pending[0]["meeting_dir"] == meeting_dir.name
    assert pending[0]["audio_file"] == "weekly-standup.mp3"
    assert [entry["label"] for entry in pending[0]["speakers"]] == ["ผู้พูด 1", "ผู้พูด 2"]


def test_process_file_attaches_the_model_guess_to_the_queue(tmp_path):
    from src.pending import load_all_pending

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)

    _, guess_mock = _run_with_two_speakers(
        config,
        guess={"ผู้พูด 2": {"name": "สมหญิง็ม", "evidence": "มีคนเรียกชื่อ"}},
    )

    speakers = load_all_pending(tmp_path)[0]["speakers"]
    assert speakers[0]["guess"] is None
    assert speakers[1]["guess"] == {"name": "สมหญิง็ม", "evidence": "มีคนเรียกชื่อ"}
    # ต้องส่งป้ายที่เขียนลงไฟล์จริงไปให้โมเดล ไม่ใช่ label ดิบของ pyannote
    assert guess_mock.call_args.args[1] == ["ผู้พูด 1", "ผู้พูด 2"]


def test_process_file_still_queues_speakers_when_the_guess_call_fails(tmp_path):
    from src.pending import load_all_pending

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)

    _run_with_two_speakers(config, guess_error=RuntimeError("โมเดลล่ม"))

    speakers = load_all_pending(tmp_path)[0]["speakers"]
    assert len(speakers) == 2
    assert all(entry["guess"] is None for entry in speakers)


def test_process_file_guesses_with_the_model_the_meeting_chose(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    write_job(config.inbox_dir, "weekly-standup", "claude-sonnet-5")

    _, guess_mock = _run_with_two_speakers(config)

    assert guess_mock.call_args.kwargs["model"] == "claude-sonnet-5"


def test_process_file_queues_nobody_when_diarization_gave_no_embeddings(tmp_path):
    from src.pending import load_all_pending

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 30.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization([{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}]),
        ),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        process_file(audio_path, config)

    assert load_all_pending(tmp_path) == []


def test_process_file_finishes_the_meeting_when_writing_the_queue_fails(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    with (
        _mock_convert_to_wav(),
        _mock_load_embedder(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 30.0, "text": "สวัสดีครับ"}],
        ),
        patch(
            "src.pipeline.diarize_audio",
            return_value=_diarization(
                [{"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00"}]
            ),
        ),
        _stub_voiceprints({"SPEAKER_00": [1.0, 0.0]}),
        patch(
            "src.pipeline.write_pending", side_effect=OSError("ดิสก์เต็ม")
        ) as mock_write_pending,
        patch("src.pipeline.guess_speaker_names", return_value={}),
        patch("src.pipeline.summarize_transcript", return_value="## สรุป"),
    ):
        meeting_dir = process_file(audio_path, config)

    # ถ้าไม่มีใครเข้าคิว write_pending จะไม่ถูกเรียกเลย เทสต์จะผ่านโดยไม่ได้วัดอะไร --
    # ยืนยันว่าความล้มเหลวที่เราตั้งใจให้เกิด เกิดขึ้นจริง
    mock_write_pending.assert_called_once()
    assert (meeting_dir / "transcript.md").exists()
    assert (meeting_dir / "summary.md").exists()


def test_process_file_records_the_queue_only_after_the_meeting_is_done(tmp_path):
    from src import activity

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)

    _run_with_two_speakers(config)

    codes = [entry["code"] for entry in activity.tail(tmp_path)]
    # ผู้ใช้ต้องเห็น "เสร็จแล้ว" ก่อนงานหลังบ้าน ไม่ใช่รองานหลังบ้านก่อนถึงจะเสร็จ
    assert codes.index("meeting_done") < codes.index("speakers_pending")


def test_process_file_queues_speakers_in_transcript_only_mode_without_a_model_call(
    tmp_path,
):
    """transcript-only แปลว่าผู้ใช้เลือกเองว่าจะไม่ให้คำพูดของเขาหลุดออกจากเครื่อง

    ต่างจากเทสต์อื่นใน task นี้ที่ mock src.pipeline.guess_speaker_names ตรง ๆ --
    ที่นี่ปล่อยให้ตัวจริงทำงาน แล้ว mock แค่ src.speaker_guess.resolve เพื่อพิสูจน์ว่า
    ด่านกันในตัวมันเอง (เช็ค model == NO_SUMMARY_MODEL ก่อนเรียก resolve) เป็นสิ่งที่
    ทำให้ไม่มี network call เกิดขึ้นจริง ไม่ใช่แค่เชื่อว่ามันน่าจะกันไว้แล้ว
    """
    from src.pending import load_all_pending

    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")
    write_job(config.inbox_dir, "weekly-standup", NO_SUMMARY_MODEL)

    with (
        _two_speaker_diarization_patched(),
        patch("src.speaker_guess.resolve") as resolve_mock,
        patch("src.pipeline.summarize_transcript") as summarize_mock,
    ):
        meeting_dir = process_file(audio_path, config)

    resolve_mock.assert_not_called()
    summarize_mock.assert_not_called()
    speakers = load_all_pending(tmp_path)[0]["speakers"]
    assert [entry["label"] for entry in speakers] == ["ผู้พูด 1", "ผู้พูด 2"]


def test_an_unreachable_model_fails_before_the_map_stage_is_paid_for(tmp_path):
    """หัวใจของเรื่องนี้: 2026-07-31 เสียไปหนึ่งชั่วโมงเต็ม (5 chunks x 3 retries x 900
    วินาที) กว่าจะรู้ว่าปลายทางไปไม่ถึง -- probe ต้องยิงก่อน และ summarize ต้องไม่ถูก
    เรียกเลยเมื่อมันไม่ผ่าน

    fixture skip_the_reachability_probe ถูกทับด้วย patch ซ้อนในนี้โดยตั้งใจ
    """
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch(
            "src.pipeline.check_model_reachable",
            side_effect=TimeoutError("the read operation timed out"),
        ) as probe,
        patch("src.pipeline.summarize_transcript") as summarize,
        pytest.raises(TimeoutError),
    ):
        process_file(audio_path, config)

    probe.assert_called_once()
    summarize.assert_not_called()
    assert _stage_codes(config)[-1] == "job_failed"


def test_an_unreachable_model_still_leaves_the_transcript_behind(tmp_path):
    """transcript ถูกเซฟก่อนขั้นสรุปอยู่แล้ว การล้มที่ probe จึงต้องไม่ทำให้งานถอดเสียง
    หาย -- ไม่งั้นการ "ล้มเร็ว" กลายเป็นการทิ้งงาน 40 นาทีเร็วขึ้นเฉย ๆ"""
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
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch(
            "src.pipeline.check_model_reachable",
            side_effect=TimeoutError("the read operation timed out"),
        ),
        pytest.raises(TimeoutError),
    ):
        process_file(audio_path, config)

    transcripts = list(config.meetings_dir.glob("*/transcript.md"))
    assert len(transcripts) == 1, "transcript ต้องรอด"
    assert "สวัสดีครับ" in transcripts[0].read_text(encoding="utf-8")
    # และ .job.json ต้องชี้กลับมาที่ transcript นั้น เพื่อให้ลากกลับ inbox แล้วสรุปใหม่
    # ได้โดยไม่ต้องถอดเสียงซ้ำ (README: ลองใหม่หลังจากพัง)
    assert read_transcript(config.failed_dir / audio_path.name) == transcripts[0]


def test_the_activity_feed_moves_while_the_summary_is_being_written(tmp_path):
    """แถบใน UI เคยค้างที่ขั้น "กำลังสรุป" ตั้งแต่นาทีแรกจนจบ ระหว่าง summarize_started
    กับ job_failed/meeting_done ไม่มีเหตุการณ์อะไรคั่นเลย -- payload ของ /api/state
    จึงเหมือนกันทุกไบต์ทุกรอบที่ poll ซึ่งแยกไม่ออกว่ายังทำงานอยู่หรือแขวนไปแล้ว
    (2026-07-31: เงียบแบบนั้นไปหนึ่งชั่วโมง)"""
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "weekly-standup.mp3"
    audio_path.write_bytes(b"fake audio")

    def fake_summarize(*args, on_progress=None, **kwargs):
        assert on_progress is not None, "pipeline ต้องส่ง callback เข้าไปเสมอ"
        for done in (1, 2, 3):
            on_progress(done, 3)
        return "## สรุป"

    with (
        _mock_convert_to_wav(),
        patch(
            "src.pipeline.transcribe_audio",
            return_value=[{"start": 0.0, "end": 2.0, "text": "สวัสดีครับ"}],
        ),
        patch("src.pipeline.diarize_audio", return_value=_diarization([])),
        patch("src.pipeline.summarize_transcript", side_effect=fake_summarize),
    ):
        process_file(audio_path, config)

    events = _activity_events(config, "summarize_progress")
    assert [e["params"] for e in events] == [
        {"done": 1, "total": 3},
        {"done": 2, "total": 3},
        {"done": 3, "total": 3},
    ]
    codes = _stage_codes(config)
    assert codes.index("summarize_started") < codes.index("summarize_progress")
    assert codes.index("summarize_progress") < codes.index("meeting_done")
