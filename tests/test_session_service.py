import inspect
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src import enroll, session_service, speakers
from src.activity import append, tail
from src.config import Config
from src.pending import build_pending_speakers, pending_dir, write_pending
from src.session_service import create_app, probe_worker, gpu_is_busy
from src.speakers import add_sample, load_registry, save_registry
from src.voiceprint import Voiceprint

MODEL = "pyannote/speaker-diarization-community-1"
OTHER_MODEL = "pyannote/speaker-diarization-3.1"
# ป้ายพื้นที่เวกเตอร์ (ตัวที่ match_known ใช้กรองจริง) -- คนละตัวกับ MODEL/OTHER_MODEL
# ข้างบนซึ่งเป็นโมเดลแยกผู้พูด (diarization) ที่ตอนนี้เป็นแค่ provenance เสริม ค่าเดียวกับ
# ที่ tests/test_speakers.py และ tests/test_pending.py ใช้ (ตรงกับ DEFAULT_EMBEDDING_MODEL)
EMBED = "pyannote/wespeaker-voxceleb-resnet34-LM"
OTHER_EMBED = "speechbrain/spkrec-ecapa-voxceleb"


def _sample(embedding, embedding_model=EMBED, **extra):
    """payload dict สำหรับ add_sample -- ก่อน Task 12 add_sample รับเวกเตอร์เป็น list
    เดี่ยว ๆ (บวก keyword model=) แต่ตอนนี้ต้องเป็น dict ที่มี embedding_model บังคับเสมอ
    (ดู speakers.add_sample) ทุกเทสต์ในไฟล์นี้ที่ไม่ได้ตั้งใจทดสอบ "ไม่มีป้าย" โดยเฉพาะ
    เรียกผ่าน helper นี้แทนการประกอบ dict เองซ้ำทุกจุด
    """
    payload = {"embedding": list(embedding), "embedding_model": embedding_model}
    payload.update(extra)
    return payload


def _assert_no_embedding_vector_leaks(body):
    """เวกเตอร์เสียงต้องไม่รั่วออกไปไม่ว่าจะอยู่ใต้คีย์ไหน

    Task 12 เปลี่ยนการ์ดนี้จาก substring "embedding" ธรรมดา ไปเป็นเช็คคีย์ JSON
    ตรง ๆ ว่า '"embedding":' (เพราะตอนนั้น endpoint เริ่มมี embedding_model ที่การ์ด
    แบบเดิมปฏิเสธผิด ๆ) แต่การเช็คคีย์ตายตัวคือ denylist ตัวเดียว -- โปรเจกชันที่มันกัน
    (_public_speaker / list_entries) ก็เป็น denylist เหมือนกัน (ตัดแค่คีย์ชื่อ "embedding"
    ทิ้ง) ไฟล์ result.json และไฟล์คิวแก้มือได้ตามดีไซน์ของโปรเจกต์นี้เอง -- ใครใส่เวกเตอร์
    ไว้ใต้ชื่ออื่น (เช่น "embedding_backup" หรือ "raw_embedding") จะหลุดผ่าน denylist ทั้งสอง
    ชั้นไปเงียบ ๆ จับที่ "รูปทรง" ของค่าแทน (array ต่อท้ายคีย์ที่มีคำว่า embedding) ไม่ใช่
    ชื่อคีย์ตายตัว -- ยังยอม embedding_model ผ่านได้ตามปกติเพราะค่าของมันเป็นสตริง ไม่ใช่ array

    ยังจับได้แค่คีย์ที่ *ชื่อ* มีคำว่า "embedding" อยู่ดี -- ดู
    _assert_no_numeric_vector_leaks ด้านล่างสำหรับการ์ดที่ไม่สนใจชื่อคีย์เลย (ตัวที่ควร
    ใช้กับ endpoint ใหม่ ๆ จากนี้ไป)
    """
    dumped = json.dumps(body)
    leak = re.search(r'"(\w*embedding\w*)":\s*\[', dumped)
    if leak:
        pytest.fail(f'พบเวกเตอร์รั่วใต้คีย์ "{leak.group(1)}": {dumped}')


def _assert_no_numeric_vector_leaks(body):
    """เวกเตอร์เสียงต้องไม่รั่วไม่ว่าจะซ่อนอยู่ใต้คีย์ชื่ออะไรหรือลึกแค่ไหน (finding 3 ของ
    รีวิวรอบที่สี่)

    _assert_no_embedding_vector_leaks ข้างบนจับคีย์ที่ *ชื่อ* มีคำว่า "embedding" -- รอบ
    รีวิวที่ 3 และ 4 ของบั๊กนี้ทั้งคู่หลุดผ่านการ์ดแบบนั้นเพราะเวกเตอร์ถูกวางไว้ใต้คีย์ที่ไม่มี
    คำว่า embedding เลย (guess.voiceprint, samples[].voiceprint, suggested.voiceprint,
    ระดับบนสุดของ record ก็เคยเป็น "voiceprint" มาก่อน) -- การกันด้วยชื่อคีย์แพ้ได้เสมอแค่
    เปลี่ยนชื่อคีย์ ตัวนี้จึงเดินทั้งโครงสร้าง response แทน แล้วเช็คที่ "รูปทรง" ของค่าแทน
    ชื่อคีย์: list ที่ไม่ว่างและทุกสมาชิกเป็นตัวเลข (int/float, ไม่นับ bool ซึ่งเป็น subclass
    ของ int ใน Python) คือเวกเตอร์เสมอไม่ว่าจะอยู่ใต้คีย์ไหนหรือซ้อนลึกแค่ไหน

    ตรวจแล้วว่าไม่มี endpoint ไหนในไฟล์นี้ที่ควรส่ง numeric array ออกไปจริง ๆ:
    samples[].start/end, speaking_seconds, elapsed_seconds, sample_count, score ฯลฯ ล้วน
    เป็นตัวเลขเดี่ยว ไม่ใช่ array เลยสักตัว -- ถ้าอนาคตมี numeric array ที่ชอบธรรมจริง ๆ
    ให้เพิ่ม path เจาะจงไว้เป็นข้อยกเว้นในฟังก์ชันนี้ ไม่ใช่ผ่อนกฎทั้งหมดให้หลวมลง
    """

    def walk(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            if node and all(
                isinstance(item, (int, float)) and not isinstance(item, bool)
                for item in node
            ):
                pytest.fail(
                    f'พบ numeric array ที่ {path} (รูปทรงของเวกเตอร์เสียงที่รั่ว): {node}'
                )
            for index, item in enumerate(node):
                walk(item, f"{path}[{index}]")

    walk(body, "$")


# ชนิดของ "ใบ" ทุกใบที่ endpoint ในไฟล์นี้ประกาศว่าจะส่งออก -- ตรงกับที่ web/app.js,
# web/enroll.js และ D:\COWORK\COWORK Desktop\meetingrun.js อ่านจริง
#
# รอบที่หกของบั๊กนี้: ห้ารอบก่อนหน้าไล่ปิด "แกน" ทีละแกน (ชื่อคีย์ -> ความลึก -> ค่าของ
# คีย์ใน allowlist -> รูปทรง) แล้วรอบถัดไปก็โดนแกนใหม่ทุกครั้ง รูปทรงก็เป็นแค่แกนที่ห้า
# ไม่ใช่ค่าคงที่ของข้อมูล (ดู _PLANTED_VECTOR_SHAPES ด้านล่าง -- หกรูปทรงที่ทำซ้ำได้จริง
# บน endpoint ที่รันอยู่ ทุกตัวเลี่ยง "list ตัวเลขล้วน" ได้หมด) การ์ดที่ปิดทุกแกนพร้อมกัน
# คือการประกาศชนิดของใบ แล้วบังคับว่าไม่ใช่ชนิดนั้น = ออกไม่ได้ ไม่ว่าคนแก้ไฟล์จะเลือก
# ชื่อคีย์ ความลึก หรือรูปทรงอะไรก็ตาม เพราะไม่มีรูปทรงเหลือให้เลือกอีกแล้ว
_DECLARED_LEAF_TYPES = {
    # สตริง
    "meeting_dir": str,
    "label": str,
    "name": str,
    "evidence": str,
    "text": str,
    "audio_file": str,
    "state": str,
    "status": str,
    "reason": str,
    "suggested_name": str,
    "ts": str,
    "code": str,
    "level": str,
    "job": str,
    "path": str,
    # ตัวเลข
    "speaking_seconds": "number",
    "start": "number",
    "end": "number",
    "speaker_count": "number",
    "size_bytes": "number",
    "min_speaking_seconds": "number",
    "sample_count": "number",
    "score": "number",
    # bool
    "changed_during_analysis": bool,
    "confident": bool,
}


def _assert_declared_leaf_types(body):
    """ทุกใบที่ประกาศชนิดไว้ ต้องเป็นชนิดนั้นหรือ None เท่านั้น -- ไม่มีทางที่สาม

    _assert_no_numeric_vector_leaks ด้านบนถูกต้องแต่ตาบอดกับทุกอย่างที่ไม่ใช่ "list ที่
    สมาชิกเป็นตัวเลขล้วน" -- [0.11, 0.12, "x"] / ["x", 0.11, 0.12] / {"0": 0.11} /
    ["0.11", "0.12"] / [0.11, None, 0.12] / [{"v": 0.11}] ผ่านมันไปได้ทุกตัวทั้งที่ขน
    เวกเตอร์เต็ม ๆ ออกไปแบบกู้คืนได้ ตัวนี้จึงไม่ถามว่า "หน้าตาเหมือนเวกเตอร์ไหม" แต่ถาม
    ว่า "เป็นชนิดที่ endpoint สัญญาไว้หรือเปล่า" ซึ่งเป็นคำถามที่คนวางเวกเตอร์ตอบเลี่ยง
    ไม่ได้ เพราะเวกเตอร์ไม่ใช่ str ไม่ใช่ตัวเลขเดี่ยว และไม่ใช่ bool ไม่ว่าจะห่อมาแบบไหน
    """

    def check(node, path):
        if isinstance(node, dict):
            for key, value in node.items():
                declared = _DECLARED_LEAF_TYPES.get(key)
                here = f"{path}.{key}"
                if declared is not None and value is not None:
                    if declared == "number":
                        ok = isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        )
                    else:
                        ok = isinstance(value, declared)
                    if not ok:
                        pytest.fail(
                            f"ใบ {here} ประกาศไว้เป็น {declared} แต่ส่งออก "
                            f"{type(value).__name__}: {value!r}"
                        )
                check(value, here)
        elif isinstance(node, list):
            for index, item in enumerate(node):
                check(item, f"{path}[{index}]")

    check(body, "$")


# หกรูปทรงที่ทำซ้ำได้จริงบน endpoint ที่รันอยู่ ทุกตัวขนเวกเตอร์เดียวกันออกไปแบบกู้คืนได้
# และทุกตัวเลี่ยง _is_numeric_vector (ตาข่ายชั้นล่างของรอบที่ห้า) ได้หมด -- นี่คือหลักฐาน
# ว่า "รูปทรง" ไม่ใช่สิ่งที่คนวางเวกเตอร์เลือกไม่ได้ แต่เป็นแกนที่ห้าเหมือนชื่อคีย์และความลึก
_PLANTED_VECTOR = [0.1111111, 0.2222222, 0.3333333, 0.4444444]
_PLANT_MARK = "1111111"
_PLANTED_VECTOR_SHAPES = {
    "trailing_string": _PLANTED_VECTOR + ["x"],
    "leading_string": ["x"] + _PLANTED_VECTOR,
    "index_keyed_dict": {str(i): v for i, v in enumerate(_PLANTED_VECTOR)},
    "string_floats": [str(v) for v in _PLANTED_VECTOR],
    "null_padded": [_PLANTED_VECTOR[0], None, _PLANTED_VECTOR[1], None],
    "wrapped_objects": [{"v": v} for v in _PLANTED_VECTOR],
}


def make_config(tmp_path):
    return Config(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
        hf_token="hf-test-token",
    )


def blocking_recorder(name, model, config, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
    """ตัวอัดปลอมที่รอ stop_event เหมือนของจริง"""
    if on_event:
        on_event("room_opened", {"room": name or "", "model": model or ""})
    stop_event.wait(timeout=5)
    if on_event:
        on_event("encode_done", {"path": "fake.ogg"})
    return config.inbox_dir / "fake.ogg"


def _wait_until(client, predicate, tries=60):
    for _ in range(tries):
        body = client.get("/api/state").get_json()
        if predicate(body):
            return body
        time.sleep(0.05)
    return client.get("/api/state").get_json()


@pytest.fixture
def config(tmp_path):
    return make_config(tmp_path)


@pytest.fixture
def client(config):
    app = create_app(config, recorder=blocking_recorder, worker_probe=lambda: True)
    return app.test_client()


def test_state_starts_idle(client):
    body = client.get("/api/state").get_json()

    assert body["recorder"] == "idle"
    assert body["room"] is None
    assert body["worker_ready"] is True


def test_opening_a_room_moves_the_state_to_recording(client):
    response = client.post(
        "/api/session", json={"model": "claude-opus-5", "name": "standup"}
    )

    assert response.status_code == 201
    body = client.get("/api/state").get_json()
    assert body["recorder"] == "recording"
    assert body["room"] == "standup"
    assert body["model"] == "claude-opus-5"

    client.post("/api/session/stop")


def test_the_chosen_profile_reaches_the_recorder(config):
    """หน้าเว็บส่ง profile มาแล้วต้องไปถึงตัวอัดจริง ไม่ใช่หายที่ endpoint"""
    seen = {}

    def recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        seen["profile"] = profile
        seen["model"] = model
        stop_event.wait(timeout=5)
        return None

    app = create_app(config, recorder=recorder, worker_probe=lambda: True)
    client = app.test_client()

    client.post("/api/session", json={"model": "GLM-5.2", "profile": "cross"})
    deadline = time.monotonic() + 5
    while "profile" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    client.post("/api/session/stop")

    assert seen["profile"] == "cross"
    assert seen["model"] == "GLM-5.2"


def test_a_request_without_a_profile_leaves_it_to_the_config(config):
    """ผู้เรียกที่ไม่ส่ง profile (client เก่า) ต้องไม่ทำให้พัง -- ให้ pipeline
    ตกไปใช้ค่าจาก .env ตามปกติ"""
    seen = {}

    def recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        seen["profile"] = profile
        stop_event.wait(timeout=5)
        return None

    app = create_app(config, recorder=recorder, worker_probe=lambda: True)
    client = app.test_client()

    client.post("/api/session", json={"model": "GLM-5.2"})
    deadline = time.monotonic() + 5
    while "profile" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    client.post("/api/session/stop")

    assert seen["profile"] is None


def test_the_chosen_asr_engine_reaches_the_recorder(config):
    """หน้าเว็บส่ง asr_engine มาแล้วต้องไปถึงตัวอัดจริง แบบเดียวกับ profile"""
    seen = {}

    def recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        seen["asr_engine"] = asr_engine
        stop_event.wait(timeout=5)
        return None

    app = create_app(config, recorder=recorder, worker_probe=lambda: True)
    client = app.test_client()

    client.post("/api/session", json={"model": "GLM-5.2", "asr_engine": "typhoon"})
    deadline = time.monotonic() + 5
    while "asr_engine" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    client.post("/api/session/stop")

    assert seen["asr_engine"] == "typhoon"


def test_a_request_without_an_asr_engine_leaves_it_to_the_config(config):
    """ผู้เรียกที่ไม่ส่ง asr_engine (client เก่า) ต้องไม่ทำให้พัง -- ให้ pipeline
    ตกไปใช้ค่าจาก .env ตามปกติ"""
    seen = {}

    def recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        seen["asr_engine"] = asr_engine
        stop_event.wait(timeout=5)
        return None

    app = create_app(config, recorder=recorder, worker_probe=lambda: True)
    client = app.test_client()

    client.post("/api/session", json={"model": "GLM-5.2"})
    deadline = time.monotonic() + 5
    while "asr_engine" not in seen and time.monotonic() < deadline:
        time.sleep(0.01)
    client.post("/api/session/stop")

    assert seen["asr_engine"] is None


def test_a_blank_room_name_becomes_no_name(client):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "   "})

    assert client.get("/api/state").get_json()["room"] is None

    client.post("/api/session/stop")


def test_opening_a_second_room_is_refused(client):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "a"})

    second = client.post("/api/session", json={"model": "claude-opus-5", "name": "b"})

    assert second.status_code == 409
    # สถานะจริงต้องชนะเสมอ การกดซ้อนห้ามเปลี่ยนห้องที่กำลังอัดอยู่
    assert client.get("/api/state").get_json()["room"] == "a"

    client.post("/api/session/stop")


def test_stopping_when_idle_is_refused(client):
    assert client.post("/api/session/stop").status_code == 409


def test_stop_ends_the_recording_and_returns_to_idle(client):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "standup"})

    assert client.post("/api/session/stop").status_code == 202

    body = _wait_until(client, lambda b: b["recorder"] == "idle")
    assert body["recorder"] == "idle"


def test_stopping_twice_is_refused_the_second_time(client):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "standup"})

    assert client.post("/api/session/stop").status_code == 202
    assert client.post("/api/session/stop").status_code == 409


def test_state_reports_mic_muted_as_false_by_default(client):
    assert client.get("/api/state").get_json()["mic_muted"] is False


def test_muting_the_mic_while_recording_updates_the_state(client):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "x"})

    response = client.post("/api/session/mic", json={"muted": True})

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "muted": True}
    assert client.get("/api/state").get_json()["mic_muted"] is True

    client.post("/api/session/mic", json={"muted": False})
    assert client.get("/api/state").get_json()["mic_muted"] is False

    client.post("/api/session/stop")


def test_muting_the_mic_while_idle_is_refused(client):
    response = client.post("/api/session/mic", json={"muted": True})

    assert response.status_code == 409
    assert client.get("/api/state").get_json()["mic_muted"] is False


def test_mic_muted_resets_when_a_new_room_opens(client):
    # ค่าที่ค้างจากห้องก่อนต้องไม่รั่วเข้าห้องถัดไป -- ไม่งั้นประชุมใหม่จะเริ่มโดย
    # ไมค์ปิดอยู่แล้วโดยไม่มีใครกดปิดเองในรอบนี้
    client.post("/api/session", json={"model": "claude-opus-5", "name": "x"})
    client.post("/api/session/mic", json={"muted": True})
    client.post("/api/session/stop")
    _wait_until(client, lambda b: b["recorder"] == "idle")

    client.post("/api/session", json={"model": "claude-opus-5", "name": "y"})

    assert client.get("/api/state").get_json()["mic_muted"] is False

    client.post("/api/session/stop")


def test_mic_mute_events_land_in_the_activity_log(client, config):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "x"})
    client.post("/api/session/mic", json={"muted": True})
    client.post("/api/session/mic", json={"muted": False})

    codes = [e["code"] for e in tail(config.base_dir)]
    assert "mic_muted" in codes
    assert "mic_unmuted" in codes

    client.post("/api/session/stop")


def test_the_mic_muted_flag_passed_to_the_recorder_reflects_the_endpoint(config):
    """เทสต์นี้ตรวจว่า Event ที่ endpoint แก้กับที่ recorder เห็นเป็นอันเดียวกันจริง
    ไม่ใช่แค่ state.mic_muted ในหน้าเว็บที่ขยับแต่ recorder ไม่รู้เรื่อง
    """
    captured = {}

    def capturing_recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        captured["event"] = mic_muted
        stop_event.wait(timeout=5)
        return None

    app = create_app(config, recorder=capturing_recorder, worker_probe=lambda: True)
    client = app.test_client()
    client.post("/api/session", json={"model": "claude-opus-5", "name": "x"})
    for _ in range(60):
        if "event" in captured:
            break
        time.sleep(0.05)
    assert "event" in captured

    client.post("/api/session/mic", json={"muted": True})
    assert captured["event"].is_set() is True

    client.post("/api/session/mic", json={"muted": False})
    assert captured["event"].is_set() is False

    client.post("/api/session/stop")


def test_elapsed_seconds_counts_up_while_recording(client):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "standup"})

    time.sleep(1.1)

    assert client.get("/api/state").get_json()["elapsed_seconds"] >= 1

    client.post("/api/session/stop")


def test_warnings_from_the_recorder_reach_the_state(config):
    def warning_recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        on_event("device_changed", {"old": "A", "new": "B"}, "warn")
        stop_event.wait(timeout=5)
        return None

    app = create_app(config, recorder=warning_recorder, worker_probe=lambda: True)
    client = app.test_client()
    client.post("/api/session", json={"model": "claude-opus-5", "name": "x"})

    body = _wait_until(client, lambda b: b["warnings"])

    assert body["warnings"][0]["code"] == "device_changed"
    assert body["warnings"][0]["params"] == {"old": "A", "new": "B"}

    client.post("/api/session/stop")


def test_a_recorder_that_crashes_returns_the_state_to_idle(config):
    """ตัวอัดที่ระเบิดต้องไม่ทิ้งหน้าจอค้างที่ 'กำลังอัด' ตลอดไป"""

    def crashing_recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        raise RuntimeError("พัง")

    app = create_app(config, recorder=crashing_recorder, worker_probe=lambda: True)
    client = app.test_client()
    client.post("/api/session", json={"model": "claude-opus-5", "name": "x"})

    body = _wait_until(client, lambda b: b["recorder"] == "idle")

    assert body["recorder"] == "idle"


def test_state_includes_the_activity_log(client, config):
    append(config.base_dir, "meet-1", "queued")

    body = client.get("/api/state").get_json()

    assert body["activity"][-1]["code"] == "queued"


def test_recorder_events_land_in_the_activity_log(client, config):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "standup"})
    client.post("/api/session/stop")
    _wait_until(client, lambda b: b["recorder"] == "idle")

    assert "room_opened" in [e["code"] for e in tail(config.base_dir)]


def test_a_worker_that_is_not_running_is_reported(config):
    app = create_app(config, recorder=blocking_recorder, worker_probe=lambda: False)

    assert app.test_client().get("/api/state").get_json()["worker_ready"] is False


def test_the_index_page_is_served(client):
    assert client.get("/").status_code == 200


def test_activity_text_is_rendered_in_the_requested_language(client, config):
    append(config.base_dir, "meet-1", "transcribe_started")

    thai = client.get("/api/state").get_json()["activity"][-1]["text"]
    english = client.get("/api/state?lang=en").get_json()["activity"][-1]["text"]

    assert thai == "กำลังถอดเสียง"
    assert english == "Transcribing"


def test_warning_text_is_rendered_too(config):
    def warning_recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        on_event("device_changed", {"old": "A", "new": "B"}, "warn")
        stop_event.wait(timeout=5)
        return None

    app = create_app(config, recorder=warning_recorder, worker_probe=lambda: True)
    client = app.test_client()
    client.post("/api/session", json={"model": "claude-opus-5", "name": "x"})

    body = _wait_until(client, lambda b: b["warnings"])

    assert "A" in body["warnings"][0]["text"]
    assert "B" in body["warnings"][0]["text"]

    client.post("/api/session/stop")


_PENDING_MERGED = [
    {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_00", "text": "สวัสดีครับ ผมขอเริ่มเลย"},
    {"start": 30.0, "end": 70.0, "speaker": "SPEAKER_01", "text": "ครับผม ผมเห็นด้วย"},
]
_PENDING_LABELS = {"SPEAKER_00": "ผู้พูด 1", "SPEAKER_01": "ผู้พูด 2"}
_PENDING_EMBEDDINGS = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}


def _voiceprints(embeddings, seconds=21.4, segments=7):
    """dict[label -> Voiceprint] จากเวกเตอร์ดิบ -- build_pending_speakers ต้องการ
    Voiceprint จริง (มี .embedding/.seconds/.segment_count) ไม่ใช่ list เปล่า ๆ อีกต่อไป
    (ดู src/voiceprint.py และ tests/test_pending.py ซึ่งใช้ pattern เดียวกันนี้)
    """
    return {
        label: Voiceprint(embedding=list(vector), seconds=seconds, segment_count=segments)
        for label, vector in embeddings.items()
    }


def _queue_two_speakers(config, meeting="2026-07-28_10-30-standup"):
    write_pending(
        config.base_dir,
        meeting,
        "standup.ogg",
        build_pending_speakers(
            _PENDING_MERGED,
            _PENDING_LABELS,
            _voiceprints(_PENDING_EMBEDDINGS),
            MODEL,
            EMBED,
        ),
    )
    return meeting


def _write_queue(base_dir, entry, meeting="m", audio_file="a.ogg"):
    """เขียนไฟล์คิวหนึ่งผู้พูดตรง ๆ โดยไม่ผ่าน build_pending_speakers -- ใช้จำลองไฟล์คิว
    ที่ถูกแก้มือหรือมาจากเวอร์ชันเก่ากว่านี้ (เช่นไม่มีคีย์ embedding_model เลย) ซึ่งเป็น
    รูปทรงที่ build_pending_speakers ของวันนี้ไม่มีทางสร้างออกมาได้เองแล้ว (มันติดป้าย
    embedding_model ให้ทุกคนในคิวเสมอ) `entry` ทับค่า default ด้านล่างได้ทุกคีย์
    """
    base_entry = {"label": "SPEAKER_00", "diarization_id": "SPEAKER_00"}
    base_entry.update(entry)
    write_pending(base_dir, meeting, audio_file, [base_entry])
    return meeting


def test_pending_speakers_endpoint_is_empty_by_default(client):
    body = client.get("/api/speakers/pending").get_json()

    assert body == {"meetings": []}


def test_pending_speakers_endpoint_lists_the_queue(client, config):
    meeting = _queue_two_speakers(config)

    body = client.get("/api/speakers/pending").get_json()

    assert len(body["meetings"]) == 1
    assert body["meetings"][0]["meeting_dir"] == meeting
    # audio_file ไม่อยู่ใน allowlist ระดับการประชุม (_public_pending_meeting) อีกต่อไป --
    # web/app.js ไม่เคยอ่าน meeting.audio_file เลย (ดู pendingHtml/speakerAt ใน app.js ซึ่ง
    # ใช้แค่ meeting.meeting_dir กับ meeting.speakers) จึงไม่มีเหตุผลให้มันหลุดออก endpoint
    assert "audio_file" not in body["meetings"][0]
    labels = [entry["label"] for entry in body["meetings"][0]["speakers"]]
    assert labels == ["ผู้พูด 1", "ผู้พูด 2"]


def test_pending_speakers_endpoint_never_ships_the_voice_vectors(client, config):
    # เบราว์เซอร์ไม่ต้องใช้เวกเตอร์เลย และมันคือข้อมูล biometric -- ส่งออกไปเปล่า ๆ
    # คือเพิ่มที่ที่มันอาจรั่วโดยไม่ได้อะไรกลับมา
    meeting = _queue_two_speakers(config)
    # ไฟล์คิวแก้มือได้ตามดีไซน์ของโปรเจกต์นี้ (ดู docstring ของ _write_queue ด้านบน) --
    # จำลองไฟล์ที่ถูกแก้มือให้มีเวกเตอร์แอบอยู่ใต้ชื่อคีย์อื่นที่ไม่ใช่ "embedding" ตรง ๆ
    # _public_speaker เป็น denylist (ตัดแค่คีย์ชื่อ "embedding" ทิ้ง) คีย์แปลกใหม่แบบนี้จึง
    # หลุดผ่านไปได้เงียบ ๆ ถ้าการ์ดข้างล่างเช็คแค่ '"embedding":' ตรง ๆ (รูปแบบที่ task 12
    # เปลี่ยนมาใช้เพื่อยอม embedding_model แต่ดันแคบไปจนพลาดกรณีนี้)
    path = pending_dir(config.base_dir) / f"{meeting}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["speakers"][0]["raw_embedding"] = [1.0, 0.0]
    # finding ที่สามของรีวิวรอบนี้: ชั้นบนสุดของ record (นอก speakers[]) รั่วได้เหมือนกัน --
    # list_pending_speakers เดิมประกอบด้วย {**meeting, "speakers": ...} ซึ่งสเปรดคีย์ระดับ
    # การประชุมทุกตัวตรง ๆ โดยไม่กรองเลย (ต่างจาก _public_speaker ที่กรอง speaker แต่ละคน
    # แล้ว) วางทั้งคีย์ที่มีคำว่า "embedding" (regex ของ _assert_no_embedding_vector_leaks
    # จับได้) และคีย์ที่ไม่มีคำนั้นเลย เช่น "voiceprint" (regex จับไม่ได้ -- ต้องเช็คตรง ๆ
    # ว่าหลุดออกมาไหม เพื่อพิสูจน์ว่า allowlist เป็นตัวกันจริง ไม่ใช่แค่ regex ของการ์ดข้างล่าง)
    record["raw_embedding"] = [1.0, 0.0]
    record["voiceprint"] = [0.0, 1.0]
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    body = client.get("/api/speakers/pending").get_json()

    meeting_body = body["meetings"][0]
    for speaker in meeting_body["speakers"]:
        assert "embedding" not in speaker
    # allowlist ระดับการประชุมต้องคุมคีย์ที่ออกไปทั้งหมด ไม่ใช่แค่ตัดคีย์ต้องสงสัยทีละชื่อ --
    # เช็คว่า "voiceprint" หายไปตรง ๆ เพราะ regex ของ _assert_no_embedding_vector_leaks
    # ด้านล่างจับไม่ได้ (ไม่มีคำว่า embedding อยู่ในชื่อคีย์เลย)
    assert "voiceprint" not in meeting_body
    assert "raw_embedding" not in meeting_body
    # ตรวจทั้งก้อนด้วย เผื่อเวกเตอร์ไปโผล่ใต้คีย์อื่นที่ยังไม่มีในวันนี้ (เช่น raw_embedding
    # ข้างบน) -- จับที่รูปทรงของค่า (array ต่อท้ายคีย์ที่มีคำว่า embedding) ไม่ใช่คีย์ตายตัว
    # เดียว เพราะทุกคนในคิววันนี้มีคีย์ embedding_model ติดมาด้วยโดยตั้งใจ (ป้ายพื้นที่เวกเตอร์
    # เป็นสตริงชื่อโมเดล ไม่ใช่ข้อมูล biometric เหมือนตัวเวกเตอร์เอง จึงไม่ใช่สิ่งที่การ์ดนี้
    # มีไว้กัน) เช็คแบบ substring ธรรมดาจะชนกับ "embedding_model" เข้าเองอย่างผิด ๆ
    _assert_no_embedding_vector_leaks(body)
    _assert_no_numeric_vector_leaks(body)


def test_pending_speakers_endpoint_never_ships_vectors_nested_in_guess_samples_or_suggested(
    client, config
):
    """finding 1 ของรีวิวรอบที่สี่: _public_speaker allowlist *ชื่อคีย์* ของ guess/
    samples/suggested ถูกแล้ว แต่คืนค่าที่อยู่ใต้คีย์เหล่านั้นตรง ๆ โดยไม่กรองอะไรเลย --
    ไฟล์คิวแก้มือได้ตามดีไซน์ของโปรเจกต์นี้ (ดู docstring ของ _write_queue) ใครใส่เวกเตอร์
    ไว้ใต้คีย์ที่ชื่อไม่มีคำว่า "embedding" เลย (เช่น "voiceprint") ในสามจุดนี้จะหลุดออก
    endpoint ไปเงียบ ๆ เหมือนที่รอบ 3 เคยปล่อยให้ "voiceprint" หลุดออกจากระดับการประชุมมา
    แล้วครั้งหนึ่ง -- ทดสอบทั้งสามจุดพร้อมกันเพราะเป็นบั๊กเดียวกันซ้ำสามที่ในฟังก์ชันเดียว
    """
    meeting = _queue_two_speakers(config)
    path = pending_dir(config.base_dir) / f"{meeting}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["speakers"][0]["guess"] = {"name": "สมชาย", "voiceprint": [1.0, 0.0]}
    record["speakers"][0]["samples"][0]["voiceprint"] = [0.0, 1.0]
    record["speakers"][0]["suggested"] = {
        "name": "สมหญิง",
        "speaker_id": "sp-1",
        "score": 0.9,
        "voiceprint": [0.5, 0.5],
    }
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    body = client.get("/api/speakers/pending").get_json()

    speaker = body["meetings"][0]["speakers"][0]
    assert "voiceprint" not in speaker["guess"]
    assert "voiceprint" not in speaker["samples"][0]
    assert "voiceprint" not in speaker["suggested"]
    # suggested ไม่ใช่แค่ต้องไม่มีเวกเตอร์ -- ต้องไม่มี speaker_id/score เลยด้วย เพราะ
    # web/app.js อ่านแค่ suggested.name (ดู pendingHtml ใน app.js) คีย์ทั้งสองไม่มีเหตุผล
    # ต้องออกไปเลย
    assert speaker["suggested"] == {"name": "สมหญิง"}
    assert speaker["guess"] == {"name": "สมชาย", "evidence": None}
    _assert_no_numeric_vector_leaks(body)


def test_pending_speakers_endpoint_drops_vectors_planted_as_values_of_allowlisted_keys(
    client, config
):
    """finding 1 ของรีวิวรอบที่ห้า: allowlist กรอง "ชื่อคีย์" แล้วคืน "ค่า" ของคีย์นั้นดิบ ๆ

    สี่รอบที่ผ่านมาแก้แบบเดียวกันหมด คือแจกแจงชื่อคีย์เพิ่มอีกหนึ่งชั้น แล้วรอบถัดไปก็เจอ
    รูรั่วที่ลึกลงไปอีกหนึ่งชั้นทุกครั้ง -- รอบนี้เวกเตอร์ไม่ได้อยู่ใต้ "คีย์ใหม่" ที่ allowlist
    ไม่รู้จักอีกแล้ว แต่เป็น *ค่า* ของคีย์ที่อยู่ใน allowlist เองทุกตัว (meeting_dir, label,
    speaking_seconds, guess.evidence, samples[].start, suggested.name) การไล่แจกแจงชื่อคีย์
    อีกชั้นจึงกันกรณีนี้ไม่ได้เลยไม่ว่าจะไล่ไปกี่ชั้น ต้องกรองด้วย "รูปทรง" ของค่าแทน
    (ดู speakers.drop_numeric_vectors) -- หกจุดนี้ทำซ้ำได้จริงบน endpoint ที่รันอยู่
    """
    meeting = _queue_two_speakers(config)
    path = pending_dir(config.base_dir) / f"{meeting}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["meeting_dir"] = [0.11, 0.12]
    speaker = record["speakers"][0]
    speaker["label"] = [0.21, 0.22]
    speaker["speaking_seconds"] = [0.31, 0.32]
    speaker["guess"] = {"name": "สมชาย", "evidence": [0.41, 0.42]}
    speaker["samples"][0]["start"] = [0.51, 0.52]
    speaker["suggested"] = {"name": [0.61, 0.62]}
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    body = client.get("/api/speakers/pending").get_json()

    _assert_no_embedding_vector_leaks(body)
    _assert_no_numeric_vector_leaks(body)


@pytest.mark.parametrize("shape_name", sorted(_PLANTED_VECTOR_SHAPES))
def test_pending_speakers_endpoint_drops_every_planted_vector_shape(
    client, config, shape_name
):
    """finding ของรีวิวรอบที่หก: "รูปทรง" ไม่ใช่สิ่งที่คนวางเวกเตอร์เลือกไม่ได้

    docstring ของ drop_numeric_vectors (รอบที่ห้า) อ้างว่า "สิ่งเดียวที่คนวางเวกเตอร์
    เลือกไม่ได้คือรูปทรงของเวกเตอร์เอง" ซึ่งไม่จริง และเป็นข้ออ้างที่ค้ำดีไซน์ทั้งรอบนั้นไว้
    -- หกรูปทรงใน _PLANTED_VECTOR_SHAPES ทำซ้ำได้จริงบน endpoint ที่รันอยู่ ทุกตัวไม่ใช่
    "list ตัวเลขล้วน" จึงผ่านตาข่ายรูปทรงไปได้หมด และทุกตัวขนเวกเตอร์เดียวกันออกไปแบบ
    กู้คืนได้ครบ

    ปลูกที่ทั้งหกตำแหน่งที่รอบที่ห้าเคยพิสูจน์ไว้พร้อมกัน (meeting_dir, label,
    speaking_seconds, guess.evidence, samples[].start, suggested.name) เพราะเป็นบั๊ก
    เดียวกันซ้ำหกที่ ไม่ใช่หกบั๊ก
    """
    shape = _PLANTED_VECTOR_SHAPES[shape_name]
    meeting = _queue_two_speakers(config)
    path = pending_dir(config.base_dir) / f"{meeting}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["meeting_dir"] = shape
    speaker = record["speakers"][0]
    speaker["label"] = shape
    speaker["speaking_seconds"] = shape
    speaker["guess"] = {"name": "สมชาย", "evidence": shape}
    speaker["samples"][0]["start"] = shape
    speaker["suggested"] = {"name": shape}
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    body = client.get("/api/speakers/pending").get_json()

    _assert_declared_leaf_types(body)
    _assert_no_embedding_vector_leaks(body)
    _assert_no_numeric_vector_leaks(body)
    # เข้มกว่าการ์ดข้างบน: ตัวเลขของเวกเตอร์ต้องไม่ปรากฏใน body เลยไม่ว่าในรูปใด
    assert _PLANT_MARK not in json.dumps(body)


def test_pending_speakers_endpoint_still_serves_every_field_the_ui_reads(client, config):
    """การ์ดชนิดของใบต้องไม่กินของจริงไปด้วย -- ทุกฟิลด์ที่ web/app.js อ่านต้องยังมาครบ

    รอบที่สี่เคยแก้บั๊กนี้ด้วยการตัดฟิลด์ทิ้ง แล้วทำให้คิวตั้งชื่อผู้พูดเข้าไม่ถึงทั้งฟีเจอร์
    -- เทสต์นี้ตรึงฝั่งตรงข้ามของการ์ดไว้ ไม่ให้รอบนี้ซ้ำรอยเดิม
    """
    meeting = _queue_two_speakers(config)

    body = client.get("/api/speakers/pending").get_json()

    found = next(m for m in body["meetings"] if m["meeting_dir"] == meeting)
    speaker = found["speakers"][0]
    assert isinstance(speaker["label"], str) and speaker["label"]
    assert isinstance(speaker["speaking_seconds"], (int, float))
    sample = speaker["samples"][0]
    assert isinstance(sample["text"], str) and sample["text"]
    assert isinstance(sample["start"], (int, float))
    assert isinstance(sample["end"], (int, float))
    _assert_declared_leaf_types(body)


def test_state_activity_never_ships_a_vector_hidden_in_params(client, config):
    """finding 2 ของรีวิวรอบที่สี่: get_state ประกอบ {**e, "text": ...} จาก entry ที่อ่าน
    ตรงจาก state/activity.jsonl (ดู activity.tail) -- ไฟล์นี้แก้มือได้ตามดีไซน์เดียวกับ
    ไฟล์คิว (ดู activity.append) วันนี้ยังไม่มี production call ไหนส่ง params ที่เป็นเวกเตอร์
    จริง (latent ไม่ใช่ live) แต่โครงสร้างเหมือนสามจุดที่แก้ไปแล้วทุกประการ

    รอบที่ห้าแก้วิธีกัน: รอบที่สี่ "ตัด params ทิ้งทั้งก้อน" ซึ่งพังของจริง (ดู
    test_state_activity_still_serves_job_and_params_path ด้านล่าง) -- params ต้องออกไป
    ตามปกติ แต่ถูกกรองด้วยรูปทรงจนไม่มีเวกเตอร์เหลือ
    """
    append(
        config.base_dir,
        "meet-1",
        "queued",
        params={"voiceprint": [1.0, 0.0]},
    )

    body = client.get("/api/state").get_json()

    entry = body["activity"][-1]
    assert "voiceprint" not in entry["params"]
    assert "voiceprint" not in json.dumps(entry)
    _assert_no_numeric_vector_leaks(body)


def test_state_activity_still_serves_job_and_params_path(client, config):
    """finding 2 ของรีวิวรอบที่ห้า -- regression จริง ไม่ใช่รูรั่วที่ยังไม่ถูกใช้

    รอบที่สี่ตัด job/params ออกจาก /api/state โดยอ้าง grep ของ renderLog ตัวเดียว ซึ่งเป็น
    grep ที่ไม่ครบ: e.job ถูกอ่านอีกสามที่ -- web/app.js jobProgress() (แถบความคืบหน้าหลัง
    ปิดห้อง) web/app.js poll() (สัญญาณ speakers_pending ที่ทำให้คิวตั้งชื่อโผล่มา) และ
    D:\\COWORK\\COWORK Desktop\\meetingrun.js progressOf()/finishedMeetingId() (ซึ่งอ่าน
    e.params.path ด้วย) ผลจริงคือหน้าจอค้างที่ "กำลังประมวลผล" ขั้นที่ 1 ตลอดกาล viewDone
    ไปไม่ถึง และเพราะ viewProcessing ไม่ได้ render pendingHtml() คิวตั้งชื่อ -- ซึ่งเป็น
    เหตุผลทั้งหมดที่ฟีเจอร์นี้มีอยู่ -- จึงเข้าไม่ถึงจากหน้าเว็บเลย

    job เป็นสตริงชื่องาน การตัดมันทิ้งจึงไม่เคยกันเวกเตอร์อะไรได้ตั้งแต่แรก
    """
    append(
        config.base_dir,
        "2026-07-30_10-00-standup",
        "meeting_done",
        params={"path": "meetings/2026-07-30_10-00-standup"},
    )

    body = client.get("/api/state").get_json()

    entry = body["activity"][-1]
    assert entry["job"] == "2026-07-30_10-00-standup"
    assert entry["params"]["path"] == "meetings/2026-07-30_10-00-standup"
    # ฟิลด์ที่ renderLog/logHtml อ่านต้องยังอยู่ครบเหมือนเดิมด้วย
    assert entry["code"] == "meeting_done"
    assert entry["level"] == "info"
    assert entry["ts"]
    assert entry["text"]


@pytest.mark.parametrize("shape_name", sorted(_PLANTED_VECTOR_SHAPES))
def test_state_activity_drops_every_planted_vector_shape(client, config, shape_name):
    """หกรูปทรงเดียวกัน ฝั่ง /api/state -- ปลูกทั้งใน params และในใบระดับบรรทัด

    ts/job/code/level เป็นสตริงทั้งหมดตามที่ renderLog (app.js), logHtml (meetingrun.js),
    jobProgress() และ progressOf() อ่านจริง -- เวกเตอร์ที่ถูกวางแทนที่ค่าเหล่านั้นต้องไม่
    ออกไปไม่ว่าจะห่อมาในรูปทรงไหน
    """
    shape = _PLANTED_VECTOR_SHAPES[shape_name]
    append(config.base_dir, "meet-1", "meeting_done", params={"path": shape})
    log = config.base_dir / "state" / "activity.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    planted = json.loads(lines[-1])
    planted.update({"ts": shape, "job": shape, "code": shape, "level": shape})
    lines[-1] = json.dumps(planted, ensure_ascii=False)
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    body = client.get("/api/state").get_json()

    _assert_declared_leaf_types(body)
    _assert_no_numeric_vector_leaks(body)
    assert _PLANT_MARK not in json.dumps(body)


def test_state_activity_text_cannot_carry_a_serialized_vector(client, config):
    """render() เอา params ไปเติมลงเทมเพลตข้อความ -- เวกเตอร์ที่วางไว้ใต้ {path} จะออกไป
    เป็น *สตริง* ในคีย์ text ซึ่งการ์ดรูปทรงและการ์ดชนิดของใบจับไม่ได้ทั้งคู่ เพราะปลายทาง
    เป็น str จริง ๆ ตามที่ประกาศไว้

    ปิดด้วยสองอย่างพร้อมกัน: (1) ค่าที่ไม่ใช่ scalar ไม่ถูกเอาไปเติมเลย (2) ค่าที่เป็นสตริง
    อยู่แล้วถูกตัดความยาว -- เวกเตอร์ 256 มิติที่ serialize แล้วยาวเกินสองพันตัวอักษร ส่วน
    param จริงในโปรเจกต์นี้เป็นชื่องาน พาธไฟล์ และตัวนับเล็ก ๆ เท่านั้น
    """
    vector = [1 / (index + 3) for index in range(256)]
    serialized = json.dumps(vector)
    # เวกเตอร์ 256 มิติที่ความละเอียดของ float จริง serialize แล้วยาวราวห้าพันตัวอักษร
    assert len(serialized) > 2000
    append(config.base_dir, "meet-1", "meeting_done", params={"path": serialized})

    body = client.get("/api/state").get_json()

    entry = body["activity"][-1]
    # ตัวสุดท้ายของเวกเตอร์คือหลักฐานว่ามันออกไปครบทั้งก้อน -- ต้องไม่มีทั้งใน text และ params
    tail = str(vector[-1])
    assert tail not in entry["text"]
    assert tail not in entry["params"]["path"]
    assert len(entry["params"]["path"]) <= session_service.PARAM_VALUE_MAX_CHARS
    assert len(entry["text"]) <= len("เสร็จแล้ว: ") + session_service.PARAM_VALUE_MAX_CHARS
    _assert_declared_leaf_types(body)


def test_state_activity_text_never_renders_a_non_scalar_param(client, config):
    """params ที่ไม่ใช่ scalar ต้องไม่ไปโผล่ในข้อความที่ render ออกมา"""
    append(
        config.base_dir,
        "meet-1",
        "meeting_done",
        params={"path": {"0": 0.1111111, "1": 0.2222222}},
    )

    body = client.get("/api/state").get_json()

    entry = body["activity"][-1]
    assert _PLANT_MARK not in entry["text"]
    assert _PLANT_MARK not in json.dumps(body)
    _assert_declared_leaf_types(body)


def test_state_activity_ships_only_the_one_param_the_consumers_read(client, config):
    """params เป็นฟิลด์ปลายเปิดฟิลด์เดียวที่เหลือ -- ตรวจแล้วทั้ง app.js, enroll.js และ
    meetingrun.js ว่าไม่มีใครอ่านอะไรนอกจาก params.path (ซึ่งเป็นสตริง) การส่ง params
    ทั้งก้อนออกไปจึงเป็นการเปิดแกนที่หกไว้เปล่า ๆ ให้รอบที่เจ็ด
    """
    append(
        config.base_dir,
        "meet-1",
        "meeting_done",
        params={"path": "meetings/m1", "voiceprint": [0.11, 0.12], "extra": "x"},
    )

    body = client.get("/api/state").get_json()

    entry = body["activity"][-1]
    assert entry["params"] == {"path": "meetings/m1"}
    # แต่ข้อความยังต้องเติม {path} ได้เหมือนเดิม
    assert entry["text"].endswith("meetings/m1")
    _assert_declared_leaf_types(body)


def test_speakers_endpoint_lists_names_and_sample_counts(client, config):
    registry = add_sample([], "สมหญิง็ม", _sample([1.0, 0.0]), source="m1")
    registry = add_sample(registry, "สมหญิง็ม", _sample([0.9, 0.1]), source="m2")
    save_registry(config.base_dir, registry)

    body = client.get("/api/speakers").get_json()

    assert body["speakers"] == [
        {"id": registry[0]["id"], "name": "สมหญิง็ม", "sample_count": 2}
    ]
    _assert_no_numeric_vector_leaks(body)


def test_renaming_a_speaker_keeps_their_samples(client, config):
    """ชื่อครั้งแรกมาจากชื่อไฟล์ใน enroll/ ซึ่งมักติดส่วนเกินมา (เช่น วงเล็บชื่อเล่น) -- ก่อนมี endpoint นี้
    ทางแก้เดียวคือลบทิ้งแล้วอัดใหม่ ซึ่งทำลายตัวอย่างเสียงที่สะสมไว้เพราะสะกดผิด
    """
    registry = add_sample([], "สมชาย ( ชาย )", _sample([1.0, 0.0]), source="m1")
    registry = add_sample(registry, "สมชาย ( ชาย )", _sample([0.9, 0.1]), source="m2")
    save_registry(config.base_dir, registry)

    response = client.patch(
        f"/api/speakers/{registry[0]['id']}", json={"name": "  ชาย  "}
    )

    assert response.status_code == 200
    assert response.get_json()["speaker"]["name"] == "ชาย"
    updated = load_registry(config.base_dir)
    assert len(updated) == 1
    assert updated[0]["name"] == "ชาย"
    assert updated[0]["id"] == registry[0]["id"]
    assert len(updated[0]["samples"]) == 2
    # เวกเตอร์เสียงต้องไม่ออกไปกับ response -- ข้อมูล biometric
    assert "embedding" not in json.dumps(response.get_json())
    _assert_no_numeric_vector_leaks(response.get_json())


def test_renaming_to_a_name_someone_else_already_has_is_refused(client, config):
    registry = add_sample([], "สมชาย", _sample([1.0, 0.0]), source="m1")
    registry = add_sample(registry, "สมหญิง", _sample([0.0, 1.0]), source="m2")
    save_registry(config.base_dir, registry)
    target = next(s for s in registry if s["name"] == "สมหญิง")

    response = client.patch(f"/api/speakers/{target['id']}", json={"name": "สมชาย"})

    assert response.status_code == 409
    assert response.get_json()["error"] == "duplicate_name"
    # ทะเบียนต้องไม่ถูกแตะเลยเมื่อถูกปฏิเสธ
    after = load_registry(config.base_dir)
    assert sorted(s["name"] for s in after) == ["สมชาย", "สมหญิง"]


def test_renaming_with_an_empty_or_non_string_name_is_a_400(client, config):
    registry = add_sample([], "สมชาย", _sample([1.0, 0.0]), source="m1")
    save_registry(config.base_dir, registry)

    for payload in ({"name": "  **  "}, {"name": ""}, {"name": 42}, {}):
        response = client.patch(f"/api/speakers/{registry[0]['id']}", json=payload)
        assert response.status_code == 400, payload
        assert response.get_json()["error"] == "bad_name"

    assert load_registry(config.base_dir)[0]["name"] == "สมชาย"


def test_renaming_an_unknown_speaker_is_a_404(client, config):
    save_registry(
        config.base_dir, add_sample([], "สมชาย", _sample([1.0, 0.0]), source="m1")
    )

    response = client.patch("/api/speakers/ไม่มีจริง", json={"name": "ชื่อใหม่"})

    assert response.status_code == 404
    assert load_registry(config.base_dir)[0]["name"] == "สมชาย"


def test_deleting_a_speaker_removes_them_from_the_registry(client, config):
    registry = add_sample([], "สมหญิง็ม", _sample([1.0, 0.0]), source="m1")
    save_registry(config.base_dir, registry)

    response = client.delete(f"/api/speakers/{registry[0]['id']}")

    assert response.status_code == 200
    assert load_registry(config.base_dir) == []


def test_deleting_an_unknown_speaker_is_a_404(client):
    assert client.delete("/api/speakers/ไม่มีจริง").status_code == 404


def _saved_transcript_for(config, meeting):
    meeting_dir = config.meetings_dir / meeting
    meeting_dir.mkdir(parents=True, exist_ok=True)
    (meeting_dir / "transcript.md").write_text(
        "# Transcript\n\n"
        "**ผู้พูด 1** [00:00]: สวัสดีครับ ผมขอเริ่มเลย\n\n"
        "**ผู้พูด 2** [00:30]: ครับผม ผมเห็นด้วย\n",
        encoding="utf-8",
    )
    return meeting_dir


def test_confirming_a_name_registers_the_voice_and_rewrites_the_transcript(client, config):
    meeting = _queue_two_speakers(config)
    meeting_dir = _saved_transcript_for(config, meeting)

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "สมหญิง็ม"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "renamed": True, "name": "สมหญิง็ม"}
    registry = load_registry(config.base_dir)
    assert registry[0]["name"] == "สมหญิง็ม"
    assert registry[0]["samples"][0]["embedding"] == [0.0, 1.0]
    assert registry[0]["samples"][0]["source"] == meeting
    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "**สมหญิง็ม** [00:30]: ครับผม ผมเห็นด้วย" in transcript
    assert "**ผู้พูด 1** [00:00]" in transcript


def test_confirming_a_name_takes_that_speaker_out_of_the_queue(client, config):
    meeting = _queue_two_speakers(config)
    _saved_transcript_for(config, meeting)

    client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "สมหญิง็ม"},
    )

    body = client.get("/api/speakers/pending").get_json()
    assert [s["label"] for s in body["meetings"][0]["speakers"]] == ["ผู้พูด 1"]


def test_confirming_records_the_model_the_queue_was_built_with_not_the_current_one(
    client, config
):
    """ผู้ใช้สลับ DIARIZATION_MODEL ระหว่างที่คิวค้างอยู่ได้ -- เวกเตอร์ในคิวยังเป็นของ
    พื้นที่เดิม การติดป้ายตามค่าใน config ตอนกดยืนยันจะทำให้มันถูกเอาไปเทียบข้ามพื้นที่
    ในการประชุมครั้งหน้า ซึ่งเป็นสิ่งเดียวที่ป้ายนี้มีไว้กัน
    """
    write_pending(
        config.base_dir,
        "2026-07-28_10-30-standup",
        "standup.ogg",
        build_pending_speakers(
            _PENDING_MERGED,
            _PENDING_LABELS,
            _voiceprints(_PENDING_EMBEDDINGS),
            OTHER_MODEL,
            EMBED,
        ),
    )
    _saved_transcript_for(config, "2026-07-28_10-30-standup")
    assert config.diarization_model != OTHER_MODEL

    client.post(
        "/api/speakers/confirm",
        json={
            "meeting": "2026-07-28_10-30-standup",
            "label": "ผู้พูด 2",
            "name": "สมหญิง็ม",
        },
    )

    registry = load_registry(config.base_dir)
    assert registry[0]["samples"][0]["model"] == OTHER_MODEL


def test_confirming_an_existing_person_by_id_adds_a_second_sample(client, config):
    registry = add_sample([], "สมหญิง็ม", _sample([0.9, 0.1]), source="เมื่อวาน")
    save_registry(config.base_dir, registry)
    meeting = _queue_two_speakers(config)
    _saved_transcript_for(config, meeting)

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "speaker_id": registry[0]["id"]},
    )

    assert response.status_code == 200
    updated = load_registry(config.base_dir)
    assert len(updated) == 1
    assert len(updated[0]["samples"]) == 2


def test_skipping_a_speaker_removes_them_without_touching_the_registry(client, config):
    meeting = _queue_two_speakers(config)
    _saved_transcript_for(config, meeting)

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "skip": True},
    )

    assert response.status_code == 200
    assert load_registry(config.base_dir) == []
    body = client.get("/api/speakers/pending").get_json()
    assert [s["label"] for s in body["meetings"][0]["speakers"]] == ["ผู้พูด 1"]


def test_confirming_still_registers_the_voice_when_the_meeting_folder_is_gone(client, config):
    # meetings/ เป็นของผู้ใช้ เขาย้ายโฟลเดอร์ไปแล้วได้ -- สิ่งที่มีค่าคือทะเบียน
    # เพราะมันไปออกดอกที่การประชุมครั้งหน้า ไม่ใช่การแก้ไฟล์เก่า
    meeting = _queue_two_speakers(config)

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "สมหญิง็ม"},
    )

    assert response.get_json() == {"ok": True, "renamed": False, "name": "สมหญิง็ม"}
    assert load_registry(config.base_dir)[0]["name"] == "สมหญิง็ม"


def test_confirming_an_empty_name_is_rejected_and_keeps_the_queue_intact(client, config):
    meeting = _queue_two_speakers(config)

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "  **  "},
    )

    assert response.status_code == 400
    assert load_registry(config.base_dir) == []
    body = client.get("/api/speakers/pending").get_json()
    assert len(body["meetings"][0]["speakers"]) == 2


def test_confirming_is_a_400_when_the_queued_embedding_is_missing(client, config):
    # find_pending คืนสิ่งที่อ่านจากไฟล์คิวตรง ๆ โดยไม่ตรวจอะไรเลย -- ไฟล์คิวที่ถูก
    # แก้มือหรือมาจากเวอร์ชันเก่ากว่านี้อาจไม่มีคีย์ embedding เลย ต้องได้ 400 ที่อธิบาย
    # ได้ ไม่ใช่ KeyError ที่กลายเป็น 500
    meeting = _queue_two_speakers(config)
    path = pending_dir(config.base_dir) / f"{meeting}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    del record["speakers"][0]["embedding"]
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 1", "name": "สมหญิง็ม"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "bad_embedding"
    assert load_registry(config.base_dir) == []


def test_confirming_is_a_400_when_the_queued_embedding_is_a_zero_vector(client, config):
    # pyannote pad ศูนย์เข้ามาเมื่อจำนวน label มากกว่าจำนวน centroid -- เวกเตอร์ศูนย์
    # ล้วนไม่มีทิศทาง cosine และ "เหมือน" กับเวกเตอร์ศูนย์อื่นทุกตัวถ้าปล่อยเข้าทะเบียน
    meeting = _queue_two_speakers(config)
    path = pending_dir(config.base_dir) / f"{meeting}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["speakers"][0]["embedding"] = [0.0, 0.0]
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 1", "name": "สมหญิง็ม"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "bad_embedding"
    assert load_registry(config.base_dir) == []


def test_confirm_speaker_refuses_a_queue_entry_with_no_embedding_model(client, tmp_path):
    # คิวที่สร้างไว้ก่อนอัปเกรด (หรือถูกแก้มือ) ต้องถูกปฏิเสธพร้อมเหตุผลของตัวเอง ไม่ใช่
    # 500 ที่ไม่มีใครอธิบาย และไม่ใช่ผ่านไปเก็บ sample ที่ match_known จะข้ามตลอดกาล --
    # เวกเตอร์ตัวนี้ใช้ได้ (is_usable_embedding ผ่าน) ปัญหาคือไม่มีคีย์ embedding_model เลย
    meeting = _write_queue(tmp_path, entry={"embedding": [1.0, 0.0]})

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "SPEAKER_00", "name": "satit"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "missing_embedding_model"
    assert load_registry(tmp_path) == []


def test_confirm_speaker_stores_the_stamp_from_the_queue_not_from_config(client, tmp_path):
    # คิวอยู่ข้ามวันได้ ผู้ใช้แก้ EMBEDDING_MODEL ระหว่างนั้นได้ -- ป้ายที่ถูกคือของคิว
    meeting = _write_queue(
        tmp_path,
        entry={
            "embedding": [1.0, 0.0],
            "embedding_model": EMBED,
            "embedding_seconds": 21.4,
            "segment_count": 7,
            "model": MODEL,
        },
    )

    client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "SPEAKER_00", "name": "satit"},
    )

    sample = load_registry(tmp_path)[0]["samples"][0]
    assert sample["embedding_model"] == EMBED
    assert sample["embedding_seconds"] == 21.4
    assert sample["segment_count"] == 7
    assert sample["model"] == MODEL


def test_confirming_an_unknown_meeting_or_label_is_a_404(client, config):
    _queue_two_speakers(config)

    assert client.post(
        "/api/speakers/confirm", json={"meeting": "ไม่มีจริง", "label": "ผู้พูด 1", "name": "ก"}
    ).status_code == 404
    assert client.post(
        "/api/speakers/confirm",
        json={"meeting": "2026-07-28_10-30-standup", "label": "ผู้พูด 9", "name": "ก"},
    ).status_code == 404


def test_confirming_without_the_required_fields_is_a_400(client):
    assert client.post("/api/speakers/confirm", json={}).status_code == 400
    assert client.post("/api/speakers/confirm", json={"meeting": "m"}).status_code == 400


def test_confirming_a_name_still_succeeds_when_the_dequeue_fails(client, config, monkeypatch):
    # ทะเบียนถูกเซฟไปแล้วก่อนตัดคิว -- เสียงถูกจำแล้วจริง ตัดคิวไม่สำเร็จจึงต้องไม่
    # กลายเป็น 500 ที่บอกผู้ใช้ว่าล้มเหลวทั้งที่งานหลักทำสำเร็จไปแล้ว
    meeting = _queue_two_speakers(config)
    _saved_transcript_for(config, meeting)
    monkeypatch.setattr("src.session_service.pending.resolve_pending", lambda *a, **k: False)

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "สมหญิง็ม"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "renamed": True, "name": "สมหญิง็ม"}
    assert load_registry(config.base_dir)[0]["name"] == "สมหญิง็ม"


def test_skipping_when_the_dequeue_fails_is_a_500_and_the_registry_stays_empty(
    client, config, monkeypatch
):
    # ทางข้ามไม่ได้ทำอะไรอย่างอื่นเลย ถ้าตัดคิวไม่สำเร็จก็แปลว่าไม่มีอะไรเกิดขึ้นจริง
    # การตอบ ok จึงเป็นการโกหก
    meeting = _queue_two_speakers(config)
    _saved_transcript_for(config, meeting)
    monkeypatch.setattr("src.session_service.pending.resolve_pending", lambda *a, **k: False)

    response = client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "skip": True},
    )

    assert response.status_code == 500
    assert load_registry(config.base_dir) == []


def test_speaker_audio_endpoint_serves_the_archived_recording(client, config):
    meeting = _queue_two_speakers(config)
    meeting_dir = config.meetings_dir / meeting
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "standup.ogg").write_bytes(b"OggS-fake-audio")

    response = client.get(f"/api/speakers/audio/{meeting}")

    assert response.status_code == 200
    assert response.get_data() == b"OggS-fake-audio"


def test_speaker_audio_endpoint_supports_seeking(client, config):
    # หน้าเว็บกระโดดไปยังช่วงที่คนนั้นพูด การเล่นตั้งแต่ต้นไฟล์ประชุมชั่วโมงหนึ่ง
    # เพื่อฟังหกวินาทีคือสิ่งที่ทำให้ไม่มีใครใช้ปุ่มนี้
    meeting = _queue_two_speakers(config)
    meeting_dir = config.meetings_dir / meeting
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "standup.ogg").write_bytes(b"0123456789")

    response = client.get(
        f"/api/speakers/audio/{meeting}", headers={"Range": "bytes=2-5"}
    )

    assert response.status_code == 206
    assert response.get_data() == b"2345"


def test_speaker_audio_endpoint_refuses_a_path_that_escapes_meetings(client, config):
    outside = config.base_dir / "ความลับ.ogg"
    outside.write_bytes("ห้ามอ่าน".encode("utf-8"))

    response = client.get("/api/speakers/audio/..%2fความลับ.ogg")

    assert response.status_code == 404
    assert "ห้ามอ่าน".encode("utf-8") not in response.get_data()


def test_speaker_audio_endpoint_refuses_a_tampered_meeting_dir(client, config):
    # เทสข้างบน 404 ตั้งแต่ guard แรก (ไม่มีการประชุมชื่อนั้นในคิว) จึงไม่เคยไปถึง
    # safe_meeting_dir เลย -- ลบตัวกันนั้นทิ้งก็ยังผ่าน ตัวนี้บังคับให้เดินไปถึงจริง
    # โดยแก้ field meeting_dir ในไฟล์คิวให้พาออกนอกโฟลเดอร์ ซึ่งเป็นสิ่งเดียวที่
    # safe_meeting_dir มีไว้กัน
    outside = config.base_dir / "ความลับ.ogg"
    outside.write_bytes("ห้ามอ่าน".encode("utf-8"))
    meeting = _queue_two_speakers(config)
    path = pending_dir(config.base_dir) / f"{meeting}.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["meeting_dir"] = "../ความลับ.ogg"
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")

    response = client.get(f"/api/speakers/audio/{record['meeting_dir']}")

    assert response.status_code == 404
    assert "ห้ามอ่าน".encode("utf-8") not in response.get_data()


def test_speaker_audio_endpoint_is_404_for_a_meeting_with_nothing_queued(client, config):
    meeting_dir = config.meetings_dir / "ไม่ได้อยู่ในคิว"
    meeting_dir.mkdir(parents=True)
    (meeting_dir / "a.ogg").write_bytes(b"x")

    assert client.get("/api/speakers/audio/ไม่ได้อยู่ในคิว").status_code == 404


def test_speaker_audio_endpoint_is_404_when_the_file_was_moved(client, config):
    meeting = _queue_two_speakers(config)
    (config.meetings_dir / meeting).mkdir(parents=True)

    assert client.get(f"/api/speakers/audio/{meeting}").status_code == 404


def put_enroll_audio(base_dir, name="สมชาย.ogg"):
    directory = base_dir / "enroll"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_bytes(b"fake audio")


def write_ok_result(base_dir, audio_file, analyzed):
    """เขียนผล status ok พร้อมผูกกับ stat ปัจจุบันของไฟล์เสียงเสมอ

    หลัง finding 1 (defense in depth): write_result ที่ได้รับ status "ok" แต่ไม่มี
    pre_analysis_stat มาเทียบ (None) จะถือว่ายืนยันไม่ได้แล้วล้างทั้งสอง sidecar ทิ้งทันที
    (finding 4) -- เทสต์จำนวนมากในไฟล์นี้แค่ต้องการ fixture "วิเคราะห์เสร็จแล้วสถานะ ok"
    ไม่ได้ตั้งใจทดสอบกลไกผูกไฟล์เอง จึงรวม stat ที่ต้องผูกไว้ที่นี่เรียกครั้งเดียว
    """
    stat = (base_dir / "enroll" / audio_file).stat()
    return enroll.write_result(
        base_dir,
        audio_file,
        analyzed,
        pre_analysis_stat=(stat.st_size, stat.st_mtime),
    )


def test_get_enroll_lists_files_with_state_and_never_leaks_vectors(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2], "suggested_name": "สมชาย"},
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    assert body["files"][0]["audio_file"] == "สมชาย.ogg"
    assert body["files"][0]["state"] == "done"
    assert "embedding" not in json.dumps(body)
    _assert_no_numeric_vector_leaks(body)


def test_get_enroll_surfaces_the_changed_during_analysis_flag_once(tmp_path):
    """finding 5 ของรีวิวรอบนี้: เมื่อ write_result ล้างผลทิ้งเพราะไฟล์เปลี่ยนระหว่างวิเคราะห์
    /api/enroll ต้องส่งคำอธิบายให้หน้าเว็บเห็นสักครั้งหนึ่ง ไม่ใช่แค่เด้งกลับไป "idle" เงียบ ๆ
    โดยไม่มีร่องรอยอะไรให้ผู้ใช้รู้ว่าทำไม แล้วต้องหายไปเองในรอบ poll ถัดไป (ไม่ค้างถาวร)
    """
    config = make_config(tmp_path)
    audio_path = tmp_path / "enroll" / "สมชาย.ogg"
    put_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    stat_before = audio_path.stat()
    pre_analysis_stat = (stat_before.st_size, stat_before.st_mtime)
    audio_path.unlink()
    audio_path.write_bytes(b"a completely different recording, replaced mid-analysis")
    enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding": [0.1, 0.2]},
        pre_analysis_stat=pre_analysis_stat,
    )
    client = create_app(config).test_client()

    first = client.get("/api/enroll").get_json()
    assert first["files"][0]["state"] == "idle"
    assert first["files"][0]["changed_during_analysis"] is True

    second = client.get("/api/enroll").get_json()
    assert "changed_during_analysis" not in second["files"][0]


def test_get_enroll_reports_the_minimum_speaking_seconds_threshold(tmp_path):
    """finding 5: MIN_SPEAKING_SECONDS ต้องมาจากแหล่งเดียว (src/speakers.py) หน้าเว็บ
    ต้องอ่านค่ามาแสดง ไม่ใช่ฝังตัวเลข 10 ซ้ำไว้เป็นสตริงคงที่อีกชุดหนึ่ง
    """
    config = make_config(tmp_path)
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    assert body["min_speaking_seconds"] == speakers.MIN_SPEAKING_SECONDS


def test_get_enroll_reports_whether_the_worker_is_running(tmp_path):
    config = make_config(tmp_path)
    client = create_app(config, worker_probe=lambda: False).test_client()

    # ไม่มี watcher = ไฟล์จะค้างที่ "กำลังวิเคราะห์" ตลอดไป ผู้ใช้ต้องเห็นสาเหตุ
    assert client.get("/api/enroll").get_json()["worker"] is False


def test_get_enroll_also_returns_the_current_registry(tmp_path):
    config = make_config(tmp_path)
    speakers.save_registry(
        tmp_path, speakers.add_sample([], "สมหญิง", _sample([0.1, 0.2]), source="enroll:x.ogg")
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    assert body["speakers"][0]["name"] == "สมหญิง"
    assert body["speakers"][0]["sample_count"] == 1
    assert "embedding" not in json.dumps(body)
    _assert_no_numeric_vector_leaks(body)


def test_get_enroll_reports_the_best_registry_match_when_at_or_above_the_low_threshold(
    tmp_path,
):
    """finding B ของรีวิวรอบสุดท้าย: สเปกต้องการคะแนนความคล้ายกับคนที่มีอยู่แล้วในทะเบียน
    แต่ฟีเจอร์นี้ไม่เคยถูกสร้างจริงและไม่มีใครจดไว้ -- นี่คือด่านเดียวที่จับได้ทั้งกรณี
    ลงทะเบียนคนเดิมซ้ำด้วยการสะกดชื่อคนละแบบ และกรณีเสียงเป็นของคนอื่น
    """
    config = make_config(tmp_path)
    speakers.save_registry(
        tmp_path, speakers.add_sample([], "สมชาย", _sample([1.0, 0.0]), source="m1")
    )
    put_enroll_audio(tmp_path)
    # cosine([1,0], [0.6,0.8]) = 0.6 -- อยู่ระหว่าง LOW (0.45) กับ HIGH (0.70) เกณฑ์
    # เริ่มต้นของ Config พอดี ไม่ถึงขั้นเสนอให้รวมชื่อ แต่ต้องเตือนให้เห็น -- embedding_model
    # ต้องตรงกับป้ายในทะเบียนไม่งั้น match_known ข้ามตัวอย่างนี้ไปเงียบ ๆ (คนละพื้นที่เวกเตอร์)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "model": MODEL, "embedding_model": EMBED, "embedding": [0.6, 0.8]},
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    match = body["files"][0]["match"]
    assert match["name"] == "สมชาย"
    assert match["score"] == pytest.approx(0.6, abs=0.01)
    assert match["confident"] is False


def test_get_enroll_flags_a_match_at_or_above_the_high_threshold(tmp_path):
    """คะแนนถึงเกณฑ์ HIGH ต้องบอกตรง ๆ ว่าบันทึกชื่อเดิมจะรวมตัวอย่างเข้าคนเดิมทันที
    (พฤติกรรมจริงของ add_sample เมื่อชื่อซ้ำ) หน้าเว็บใช้ flag นี้ตัดสินว่าจะโชว์
    ประโยคเตือนเพิ่มหรือไม่
    """
    config = make_config(tmp_path)
    speakers.save_registry(
        tmp_path, speakers.add_sample([], "สมชาย", _sample([1.0, 0.0]), source="m1")
    )
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "model": MODEL, "embedding_model": EMBED, "embedding": [1.0, 0.0]},
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    match = body["files"][0]["match"]
    assert match["score"] == 1.0
    assert match["confident"] is True


def test_get_enroll_omits_match_below_the_low_threshold(tmp_path):
    """คะแนนต่ำกว่า LOW ถือว่าไม่รู้จัก -- ต้องไม่มีคีย์ match เลย ไม่ใช่ match ที่มี
    คะแนนต่ำเกลื่อนจนผู้ใช้เพิกเฉยข้อความเตือนที่มีความหมายจริงไปด้วย
    """
    config = make_config(tmp_path)
    speakers.save_registry(
        tmp_path, speakers.add_sample([], "สมชาย", _sample([1.0, 0.0]), source="m1")
    )
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding_model": EMBED, "embedding": [0.0, 1.0]},
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    assert "match" not in body["files"][0]


def test_get_enroll_never_leaks_the_embedding_even_when_a_match_is_found(tmp_path):
    """ตรึงไว้ว่า /api/enroll ต้องไม่ส่ง embedding ออกไปเลย แม้แต่ตอนที่คำนวณ match
    (ซึ่งต้องอ่าน result.json ดิบเพื่อเทียบ) -- ผู้เรียกใน session_service ต้องเลือก
    เอาแค่ name/score/confident ออกจาก Match เท่านั้น ไม่ใช่ทั้งก้อน
    """
    config = make_config(tmp_path)
    speakers.save_registry(
        tmp_path, speakers.add_sample([], "สมชาย", _sample([1.0, 0.0]), source="m1")
    )
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {
            "status": "ok",
            "model": MODEL,
            "embedding_model": EMBED,
            "embedding": [1.0, 0.0],
            # result.json แก้มือได้ตามดีไซน์ของโปรเจกต์นี้ (ดู docstring ของ write_ok_result
            # ด้านบน) -- จำลองไฟล์ที่มีเวกเตอร์แอบอยู่ใต้ชื่อคีย์อื่นที่ไม่ใช่ "embedding"
            # ตรง ๆ list_entries เป็น denylist (ตัดแค่คีย์ชื่อ "embedding" ทิ้ง) คีย์แปลกใหม่
            # แบบนี้จึงหลุดผ่านไปได้เงียบ ๆ ถ้าการ์ดข้างล่างเช็คแค่ '"embedding":' ตรง ๆ
            "embedding_backup": [1.0, 0.0],
        },
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    assert "match" in body["files"][0]
    # ตรวจทั้งก้อนด้วย เผื่อเวกเตอร์ไปโผล่ใต้คีย์อื่นที่ยังไม่มีในวันนี้ (เช่น embedding_backup
    # ข้างบน) -- จับที่รูปทรงของค่า (array ต่อท้ายคีย์ที่มีคำว่า embedding) ไม่ใช่คีย์ตายตัว
    # เดียว เพราะผลลัพธ์ทุกไฟล์วันนี้มีคีย์ embedding_model ติดมาด้วยโดยตั้งใจ (สตริงชื่อโมเดล
    # ไม่ใช่ข้อมูล biometric เหมือนตัวเวกเตอร์เอง) เช็คแบบ substring ธรรมดาจะชนกับ
    # "embedding_model" เข้าเองอย่างผิด ๆ
    _assert_no_embedding_vector_leaks(body)
    _assert_no_numeric_vector_leaks(body)


def test_get_enroll_drops_vectors_planted_as_values_of_allowlisted_keys(tmp_path):
    """finding 1 ของรีวิวรอบที่ห้า ฝั่ง /api/enroll -- คู่แฝดของเทสต์ชื่อเดียวกันฝั่งคิว

    allowlist กรอง "ชื่อคีย์" แล้ว entry.update() เอา "ค่า" ของคีย์เหล่านั้นมา
    ตรง ๆ -- result.json แก้มือได้ตามดีไซน์ของโปรเจกต์นี้ เวกเตอร์ที่วางเป็นค่าของ
    speaking_seconds/speaker_count/suggested_name จึงหลุดออกไปได้ทั้งที่ allowlist ทำงาน
    ถูกต้องทุกประการ และ reason ที่เป็น dict ทั้งก้อนพิสูจน์ว่าความลึกไม่มีขอบ: ซับทรี
    อะไรก็ได้ที่ห้อยอยู่ใต้คีย์ใน allowlist จะออกไปทั้งดุ้น
    """
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {
            "status": "rejected",
            "speaking_seconds": [0.11, 0.12],
            "speaker_count": [0.21, 0.22],
            "suggested_name": [0.31, 0.32],
            "reason": {"voiceprint": [0.41, 0.42]},
        },
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    _assert_no_embedding_vector_leaks(body)
    _assert_no_numeric_vector_leaks(body)
    # ค่าที่เซิร์ฟเวอร์คำนวณเองต้องไม่หายไปด้วย: suggested_name ที่ถูกต้องมาจากชื่อไฟล์
    # (suggested_name_from) และถูกทับด้วยของปลอมจาก result.json -- ตัดของปลอมทิ้งแล้ว
    # ของจริงต้องยังอยู่ ไม่ใช่กลายเป็นช่องว่าง
    assert body["files"][0]["suggested_name"] == "สมชาย"


@pytest.mark.parametrize("shape_name", sorted(_PLANTED_VECTOR_SHAPES))
def test_get_enroll_drops_every_planted_vector_shape(tmp_path, shape_name):
    """หกรูปทรงเดียวกัน ฝั่ง /api/enroll -- คู่แฝดของเทสต์ชื่อเดียวกันฝั่งคิว

    ปลูกที่ทุกใบที่มาจาก result.json (ซึ่งแก้มือได้ตามดีไซน์ของโปรเจกต์นี้) พร้อมกัน:
    status/reason เป็นสตริง speaking_seconds/speaker_count เป็นตัวเลข suggested_name
    เป็นสตริง -- ตรงกับที่ web/enroll.js อ่านจริง (chipFor/renderFile)
    """
    shape = _PLANTED_VECTOR_SHAPES[shape_name]
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {
            "status": shape,
            "reason": shape,
            "speaking_seconds": shape,
            "speaker_count": shape,
            "suggested_name": shape,
        },
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    _assert_declared_leaf_types(body)
    _assert_no_embedding_vector_leaks(body)
    _assert_no_numeric_vector_leaks(body)
    assert _PLANT_MARK not in json.dumps(body)
    # ค่าที่เซิร์ฟเวอร์คำนวณเองต้องรอด ไม่ถูกของปลอมทับ
    assert body["files"][0]["suggested_name"] == "สมชาย"
    assert body["files"][0]["audio_file"] == "สมชาย.ogg"
    assert body["files"][0]["state"] == "done"


def test_get_enroll_still_serves_every_field_the_ui_reads(tmp_path):
    """ฝั่งตรงข้ามของการ์ด: ทุกฟิลด์ที่ web/enroll.js อ่านต้องยังมาครบและเป็นชนิดที่ประกาศไว้"""
    config = make_config(tmp_path)
    speakers.save_registry(
        tmp_path, speakers.add_sample([], "สมชาย", _sample([1.0, 0.0]), source="m1")
    )
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {
            "status": "ok",
            "model": MODEL,
            "embedding_model": EMBED,
            "embedding": [1.0, 0.0],
            "speaking_seconds": 68.5,
            "speaker_count": 1,
        },
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    entry = body["files"][0]
    assert entry["audio_file"] == "สมชาย.ogg"
    assert entry["state"] == "done"
    assert entry["status"] == "ok"
    assert entry["speaking_seconds"] == 68.5
    assert entry["speaker_count"] == 1
    assert isinstance(entry["size_bytes"], int)
    assert entry["suggested_name"] == "สมชาย"
    assert isinstance(entry["match"]["score"], float)
    assert isinstance(entry["match"]["name"], str)
    assert isinstance(entry["match"]["confident"], bool)
    assert isinstance(body["min_speaking_seconds"], (int, float))
    assert isinstance(body["speakers"][0]["sample_count"], int)
    _assert_declared_leaf_types(body)


def test_enroll_similarity_uses_the_stamp_recorded_in_the_result_not_the_current_config(
    tmp_path,
):
    """เวกเตอร์ใน result.json วิเคราะห์ไว้ตอนไหนก็ได้ (ผู้ใช้สลับ EMBEDDING_MODEL ได้
    ระหว่างที่ยังไม่กดยืนยัน) ต้องเทียบกับทะเบียนในพื้นที่ของ *มัน* เอง ไม่ใช่ของโมเดลที่
    ตั้งอยู่ตอนนี้ -- ตัวเลขเหมือนกันเป๊ะ (cosine 1.0) แต่คนละพื้นที่เวกเตอร์ต้องไม่ match กัน
    """
    config = make_config(tmp_path)
    speakers.save_registry(
        tmp_path, speakers.add_sample([], "สมชาย", _sample([1.0, 0.0], embedding_model=EMBED), source="m1")
    )
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "model": MODEL, "embedding_model": OTHER_EMBED, "embedding": [1.0, 0.0]},
    )
    client = create_app(config).test_client()

    body = client.get("/api/enroll").get_json()

    assert "match" not in body["files"][0]


def test_list_speakers_and_list_enroll_share_the_projection_helper():
    """finding C ของรีวิวรอบสุดท้าย: list_enroll เคยคัดลอกการฉายภาพผู้พูด
    (id/name/sample_count) มาจาก list_speakers ตรง ๆ เป็นสำเนาที่สาม โดยไม่ใช้ helper
    ตัวไหนเลยแม้แต่ _public_speaker ที่มีอยู่แล้ว -- เทสต์นี้เป็น white-box โดยตั้งใจ
    เพราะเอาต์พุตของทั้งสอง endpoint เหมือนกันอยู่แล้วตั้งแต่ก่อนแก้ (นั่นคือตัวปัญหา
    เอง) จึงจับด้วยเทสต์ระดับ HTTP ไม่ได้ ต้องตรวจซอร์สโค้ดว่าฉายภาพซ้ำกี่ที่แทน
    """
    source = inspect.getsource(create_app)

    # ก่อนแก้: list_speakers กับ list_enroll ต่างประกอบ {"id":..,"name":..,
    # "sample_count":..} เองคนละที่ ทำให้ literal "sample_count" โผล่ในซอร์สของ
    # create_app สองครั้ง หลังแก้ต้องเหลือศูนย์ครั้งเพราะย้ายไปอยู่ใน helper ระดับ
    # โมดูลแทน (_speaker_summary) ซึ่งอยู่นอก create_app
    assert source.count('"sample_count"') == 0, (
        "พบการประกอบ sample_count อยู่ใน create_app โดยตรง -- "
        "ต้องย้ายไปที่ helper ระดับโมดูลแทน ไม่ใช่คัดลอกซ้ำในสอง endpoint"
    )


def test_speakers_and_enroll_endpoints_report_the_same_projection_for_a_speaker(
    client, config
):
    """พฤติกรรมที่สังเกตได้จากภายนอกต้องไม่เปลี่ยนหลังรวม helper -- shape ของทั้งสอง
    endpoint ต้องยังตรงกันทุกฟิลด์เป๊ะเหมือนก่อนแก้
    """
    registry = add_sample([], "สมหญิง็ม", _sample([1.0, 0.0]), source="m1")
    save_registry(config.base_dir, registry)

    speakers_body = client.get("/api/speakers").get_json()
    enroll_body = client.get("/api/enroll").get_json()

    assert speakers_body["speakers"] == enroll_body["speakers"]


def test_post_analyze_writes_a_request_for_each_named_file(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path, "a.ogg")
    put_enroll_audio(tmp_path, "b.ogg")
    client = create_app(config).test_client()

    response = client.post("/api/enroll/analyze", json={"files": ["a.ogg", "b.ogg"]})

    assert response.status_code == 202
    assert enroll.pending_requests(tmp_path) == ["a.ogg", "b.ogg"]


def test_post_analyze_rejects_a_filename_that_escapes_the_folder(tmp_path):
    config = make_config(tmp_path)
    (tmp_path / "enroll").mkdir()
    client = create_app(config).test_client()

    response = client.post("/api/enroll/analyze", json={"files": ["../../evil.ogg"]})

    assert response.status_code == 400
    assert not (tmp_path / "evil.request.json").exists()


def test_post_analyze_rejects_a_filename_thats_not_an_audio_extension(tmp_path):
    """finding 6: เดิม route นี้เช็คแค่ is_safe_filename -- ชื่อที่ปลอดภัยแต่ไม่ใช่ไฟล์เสียง
    (เช่น notes.txt ที่มีอยู่จริงใน enroll/) ผ่านเข้ามาเขียน notes.txt.request.json ได้ ซึ่ง
    ไม่มีอะไรกวาดทิ้งเลย (ไฟล์ต้นทางยังอยู่จริง) กลายเป็นขยะค้างตลอดกาล
    """
    config = make_config(tmp_path)
    directory = tmp_path / "enroll"
    directory.mkdir()
    (directory / "notes.txt").write_bytes(b"not audio")
    client = create_app(config).test_client()

    response = client.post("/api/enroll/analyze", json={"files": ["notes.txt"]})

    assert response.status_code == 404
    assert not (directory / "notes.txt.request.json").exists()


def test_post_confirm_saves_the_embedding_and_archives_the_file(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "ok", "embedding_model": EMBED, "embedding": [0.1, 0.2, 0.3]},
    )
    client = create_app(config).test_client()

    response = client.post(
        "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย (เล็ก)"}
    )

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "name": "สมชาย (เล็ก)"}
    registry = speakers.load_registry(tmp_path)
    assert registry[0]["name"] == "สมชาย (เล็ก)"
    assert registry[0]["samples"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert registry[0]["samples"][0]["source"] == "enroll:สมชาย.ogg"
    assert (tmp_path / "enroll" / "done" / "สมชาย.ogg").is_file()


def test_post_confirm_merges_into_an_existing_person_of_the_same_name(tmp_path):
    config = make_config(tmp_path)
    speakers.save_registry(
        tmp_path, speakers.add_sample([], "สมชาย", _sample([0.9]), source="meeting-1")
    )
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path, "สมชาย.ogg", {"status": "ok", "embedding_model": EMBED, "embedding": [0.1]}
    )
    client = create_app(config).test_client()

    client.post("/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย"})

    registry = speakers.load_registry(tmp_path)
    # ชื่อซ้ำ = คนเดิม การสร้าง entry ที่สองทำให้โปรไฟล์แตกเป็นสองก้อนที่ต่างก็อ่อนแอ
    assert len(registry) == 1
    assert len(registry[0]["samples"]) == 2


def test_post_confirm_refuses_a_rejected_result(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    enroll.write_result(
        tmp_path,
        "สมชาย.ogg",
        {"status": "rejected", "reason": "multiple_speakers", "speaker_count": 3},
    )
    client = create_app(config).test_client()

    response = client.post(
        "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย"}
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "not_enrollable"
    assert speakers.load_registry(tmp_path) == []
    assert (tmp_path / "enroll" / "สมชาย.ogg").is_file()


def test_post_confirm_refuses_a_zero_vector(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [0.0, 0.0]}
    )
    client = create_app(config).test_client()

    response = client.post(
        "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย"}
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "bad_embedding"
    assert speakers.load_registry(tmp_path) == []


def test_post_confirm_refuses_a_huge_vector_with_a_400_not_a_500(tmp_path):
    # ไฟล์ผลถูกแก้มือได้และมาจากเวอร์ชันเก่ากว่านี้ได้ (ดูคอมเมนต์ที่จุดเช็คใน
    # session_service) ค่าใหญ่ระดับนี้ทำให้การคิด norm ในการ์ดล้นเป็น OverflowError
    # ซึ่งไม่ใช่ ValueError ที่ endpoint นี้ดักไว้ ผู้ใช้จึงได้ 500 เปล่า ๆ แทนที่จะได้
    # bad_embedding ที่บอกได้ว่าไฟล์ผลใช้ไม่ได้
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path, "สมชาย.ogg", {"status": "ok", "embedding": [1e308, 1e308]}
    )
    client = create_app(config).test_client()

    response = client.post(
        "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย"}
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "bad_embedding"
    assert speakers.load_registry(tmp_path) == []


def test_post_confirm_refuses_a_name_that_is_empty_after_cleaning(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path, "สมชาย.ogg", {"status": "ok", "embedding_model": EMBED, "embedding": [0.1]}
    )
    client = create_app(config).test_client()

    response = client.post(
        "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "***"}
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "bad_name"
    # ไฟล์ต้องยังอยู่ให้ลองใหม่ได้
    assert (tmp_path / "enroll" / "สมชาย.ogg").is_file()


def test_post_confirm_leaves_the_file_alone_when_saving_the_registry_fails(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path, "สมชาย.ogg", {"status": "ok", "embedding_model": EMBED, "embedding": [0.1]}
    )
    client = create_app(config).test_client()

    with patch("src.speakers.save_registry", side_effect=OSError("disk full")):
        response = client.post(
            "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย"}
        )

    assert response.status_code == 500
    # บันทึกไม่สำเร็จแล้วย้ายไฟล์ = ผู้ใช้เสียทั้งชื่อที่พิมพ์และไฟล์ที่จะลองใหม่
    assert (tmp_path / "enroll" / "สมชาย.ogg").is_file()
    assert not (tmp_path / "enroll" / "done" / "สมชาย.ogg").exists()


def test_post_confirm_still_returns_200_when_archiving_the_file_fails(tmp_path):
    # ตรงข้ามกับเทสข้างบน: ทะเบียนบันทึกสำเร็จไปแล้วตอนที่ archive พัง เสียงถูกจำแล้วจริง
    # การตอบ error ตรงนี้จะทำให้ผู้ใช้กดยืนยันซ้ำและได้ตัวอย่างซ้ำเข้าทะเบียนคนเดิม
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path, "สมชาย.ogg", {"status": "ok", "embedding_model": EMBED, "embedding": [0.1]}
    )
    client = create_app(config).test_client()

    with patch("src.enroll.archive", side_effect=OSError("disk full")):
        response = client.post(
            "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย"}
        )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["name"] == "สมชาย"
    # finding 2: การ์ดต้องเลิกเสนอ Save ซ้ำ -- ไม่งั้นกดซ้ำได้ตัวอย่างซ้ำเข้าทะเบียนคนเดิม
    # เสียโควตา 2 ใน 10 ช่องต่อคน
    assert body["warning"] == "archive_failed"
    registry = speakers.load_registry(tmp_path)
    assert registry[0]["name"] == "สมชาย"
    assert enroll.read_result(tmp_path, "สมชาย.ogg") is None


def test_post_confirm_survives_a_non_os_error_from_archive_after_the_registry_is_saved(
    tmp_path,
):
    """Minor E: confirm_enroll ดักแค่ OSError รอบ archive() -- คอมเมนต์เดิมของโค้ดบอกว่า
    shutil.move raise shutil.Error ได้ (เช่นมีไฟล์ชื่อชนกันโผล่ใน done/ ระหว่างเช็คกับตอน
    ย้ายจริง) ซึ่งไม่ใช่ OSError แต่ python 3.12 ที่โปรเจกต์นี้ใช้ทำให้ shutil.Error เป็น
    subclass ของ OSError ไปแล้วจริง ๆ (ตรวจแล้ว) -- ใช้ RuntimeError แทนเพื่อพิสูจน์คุณค่า
    ของการดักกว้างเป็น Exception: อะไรก็ตามที่ archive() โยนออกมาหลังทะเบียนถูกบันทึกแล้ว
    (เสียงถูกจำแล้วจริง) ต้องไม่หลุดเป็น 500 -- ผู้ใช้กดซ้ำแล้วได้ตัวอย่างซ้ำเข้าทะเบียน
    คนเดิม ซึ่งเป็นสิ่งที่ guard นี้มีไว้กันอยู่แล้ว
    """
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    write_ok_result(
        tmp_path, "สมชาย.ogg", {"status": "ok", "embedding_model": EMBED, "embedding": [0.1]}
    )
    client = create_app(config).test_client()

    with patch("src.enroll.archive", side_effect=RuntimeError("unexpected")):
        response = client.post(
            "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย"}
        )

    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["warning"] == "archive_failed"
    registry = speakers.load_registry(tmp_path)
    assert registry[0]["name"] == "สมชาย"


def test_post_confirm_returns_404_when_there_is_no_result_yet(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    client = create_app(config).test_client()

    response = client.post(
        "/api/enroll/confirm", json={"audio_file": "สมชาย.ogg", "name": "สมชาย"}
    )

    assert response.status_code == 404


def test_delete_enroll_archives_the_file_without_touching_the_registry(tmp_path):
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    enroll.write_result(
        tmp_path, "สมชาย.ogg", {"status": "rejected", "reason": "too_short"}
    )
    client = create_app(config).test_client()

    response = client.delete("/api/enroll/สมชาย.ogg")

    assert response.status_code == 200
    assert (tmp_path / "enroll" / "done" / "สมชาย.ogg").is_file()
    assert speakers.load_registry(tmp_path) == []


def test_delete_enroll_returns_a_notice_instead_of_500_when_the_move_is_blocked(tmp_path):
    """Minor D: ปุ่ม "เอาออกจากรายการ" ตอนนี้โชว์ได้แม้การ์ดยังอยู่ในคิว (กำลังแปลงไฟล์
    ด้วย ffmpeg) -- shutil.move บน Windows raise PermissionError ถ้าไฟล์ยังถูกเปิดค้างอยู่
    dismiss_enroll เดิมไม่มี try ล้อม archive() เลย exception จึงหลุดเป็น 500 ที่ผู้ใช้ไม่
    เห็นคำอธิบายอะไรเลย ต้องได้ response ที่หน้าเว็บ (errAction) render เป็น notice ได้
    """
    config = make_config(tmp_path)
    put_enroll_audio(tmp_path)
    client = create_app(config).test_client()

    with patch("src.enroll.archive", side_effect=PermissionError("file in use")):
        response = client.delete("/api/enroll/สมชาย.ogg")

    # ต้องได้ JSON error ที่หน้าเว็บ render เป็น notice ได้ ไม่ใช่ 500 ที่ไม่มี body ให้อ่าน
    assert response.status_code == 500
    assert response.get_json()["error"]
    # ไฟล์ต้องยังอยู่ที่เดิมให้ผู้ใช้ลองใหม่ได้ -- ไม่มีทางรู้ว่า archive คืบหน้าไปแค่ไหน
    assert (tmp_path / "enroll" / "สมชาย.ogg").is_file()


def test_get_enroll_page_is_served(tmp_path):
    config = make_config(tmp_path)
    client = create_app(config).test_client()

    response = client.get("/enroll")

    assert response.status_code == 200
    assert b"enroll.js" in response.data


def _capture_probe_run(monkeypatch, returncode):
    """เรียก probe_worker โดยดัก subprocess.run ไว้ คืน (ผลลัพธ์, kwargs ที่ส่งเข้าไป)"""
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(kwargs)
        captured["command"] = command
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(session_service.subprocess, "run", fake_run)
    return probe_worker(), captured


def test_worker_probe_does_not_create_a_console_window(monkeypatch):
    """probe ต้องส่ง CREATE_NO_WINDOW เสมอ

    ถ้าธงนี้หายไป bug จะกลับมาแบบที่มองไม่เห็นในเทสต์อื่นเลย: service ที่วิดเจ็ตเปิด
    ด้วย pythonw.exe ไม่มี console ลูกที่เป็น console subsystem จึงได้ console ใหม่
    ทุกตัว แล้ว Windows 11 ส่งต่อให้ Windows Terminal -- หน้าต่างดำเด้งทุก 10 วินาที
    ตามรอบแคชของ probe เทสต์นี้จับที่ธง เพราะการเด้งหน้าต่างเป็นผลข้างเคียงระดับ OS
    ที่ assert จากในโปรเซสไม่ได้
    """
    _, captured = _capture_probe_run(monkeypatch, 0)

    assert captured["creationflags"] == session_service._NO_WINDOW
    if hasattr(subprocess, "CREATE_NO_WINDOW"):  # Windows เท่านั้น
        # กันเทสต์ผ่านแบบว่างเปล่า -- ถ้า _NO_WINDOW เป็น 0 การเทียบข้างบนก็ยังผ่าน
        assert captured["creationflags"] == subprocess.CREATE_NO_WINDOW != 0


@pytest.mark.parametrize("returncode,expected", [(0, True), (1, False)])
def test_worker_probe_still_reads_the_exit_code(monkeypatch, returncode, expected):
    """ธงใหม่ต้องไม่เปลี่ยนคำตอบของ probe -- 0 คือ watcher รันอยู่ นอกนั้นคือไม่"""
    result, captured = _capture_probe_run(monkeypatch, returncode)

    assert result is expected
    assert captured["command"] == session_service._WORKER_PROBE_COMMAND
    assert captured["capture_output"] is True
    assert captured["timeout"] == 10


def test_worker_probe_is_false_when_powershell_cannot_be_launched(monkeypatch):
    """ตัวกันเดิมต้องยังอยู่: launch ไม่ได้ = ตอบไม่รู้ ไม่ใช่โยน 500 ออกหน้าเว็บ"""

    def boom(command, **kwargs):
        raise OSError("powershell หายไป")

    monkeypatch.setattr(session_service.subprocess, "run", boom)

    assert probe_worker() is False


def test_importing_the_web_service_does_not_pull_in_torch():
    """หน้าเว็บต้องเปิดได้โดยไม่แตะ torch เลย -- ไม่ใช่แค่เรื่องความเร็ว

    session_service ไม่เคยเรียก pyannote สักบรรทัด แต่มัน import pending/enroll ซึ่งเคย
    ลากทาง src.voiceprint -> src.waveform -> torch เข้ามาที่หัวไฟล์ ผลคือ 500MB และ ~1.5
    วินาทีก่อน Flask จะ bind พอร์ต เพื่อของที่ไม่ได้ใช้

    ที่ทำให้มันเป็นเรื่องความถูกต้องไม่ใช่ความเร็ว: start-ui.bat เรียก
    `python -m src.session_service` เป็นกระบวนการของตัวเอง และทางเข้านั้น *ไม่มี* บล็อก
    os.add_dll_directory ที่ src/main.py มี (ตั้งแต่ python 3.8 PATH ไม่ถูกใช้ตอนแก้ชื่อ
    DLL ที่ torch พึ่งพา) -- บนเครื่องที่ชนเงื่อนไขนั้น หน้าเว็บทั้งหน้า (อัดเสียง ดู
    transcript ตั้งชื่อผู้พูด ลงทะเบียน) จะ traceback ตั้งแต่ import ทั้งที่เดิมพังแค่ watcher

    รันในกระบวนการใหม่เพราะ sys.modules ของกระบวนการที่รันเทสต์มี torch อยู่แล้วจาก
    เทสต์ไฟล์อื่น -- เช็คในนี้จะผ่านตลอดไม่ว่าโค้ดจะเป็นยังไง
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import src.session_service; "
            "print('torch' in sys.modules or 'pyannote' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parent.parent),
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False", (
        "session_service โหลด torch/pyannote ตอน import แล้ว -- หา import ที่หัวไฟล์ของ "
        "โมดูลที่มันดึงเข้ามา แล้วย้ายเข้าไปในฟังก์ชัน (หรือ TYPE_CHECKING ถ้าใช้เป็น "
        "annotation อย่างเดียว)"
    )


class RecordingCompanion:
    """companion ปลอมที่จำว่าถูกสั่งอะไรบ้างตามลำดับ"""

    def __init__(self, command, cwd=None):
        self.command = command
        self.cwd = cwd
        self.calls = []
        self.env_extra = None

    def start(self, env_extra=None):
        self.env_extra = env_extra
        self.calls.append("start")

    def stop(self):
        self.calls.append("stop")

    def is_running(self):
        return self.calls.count("start") > self.calls.count("stop")


def _client_with_companion(config, recorder=blocking_recorder):
    made = []

    def factory(command, cwd=None):
        companion = RecordingCompanion(command, cwd)
        made.append(companion)
        return companion

    app = create_app(
        config,
        recorder=recorder,
        worker_probe=lambda: True,
        companion_factory=factory,
    )
    return app.test_client(), made


def test_no_companion_is_started_when_none_is_configured(config):
    """ดีฟอลต์คือไม่มี -- เครื่องที่ไม่ตั้งค่าต้องได้พฤติกรรมเดิมเป๊ะ"""
    client, made = _client_with_companion(config)

    client.post("/api/session", json={"model": "GLM-5.2"})
    _wait_until(client, lambda b: b["recorder"] == "recording")
    client.post("/api/session/stop")

    assert made == []


def test_opening_a_room_starts_the_configured_companion(config):
    config.companion_command = ["prog", "--x"]
    client, made = _client_with_companion(config)

    client.post("/api/session", json={"model": "GLM-5.2", "name": "standup"})
    _wait_until(client, lambda b: b["recorder"] == "recording")

    assert len(made) == 1
    assert made[0].command == ["prog", "--x"]
    assert made[0].calls == ["start"]

    client.post("/api/session/stop")


def test_the_room_name_reaches_the_companion(config):
    """ให้ context ทาง env เพื่อไม่ให้มันต้องไปไล่อ่าน state ของ service"""
    config.companion_command = ["prog"]
    client, made = _client_with_companion(config)

    client.post("/api/session", json={"model": "GLM-5.2", "name": "standup"})
    _wait_until(client, lambda b: b["recorder"] == "recording")

    assert made[0].env_extra["MEETING_ROOM"] == "standup"

    client.post("/api/session/stop")


def test_the_companion_is_stopped_when_the_encode_stage_begins(config):
    """ต้องหยุดตอนเข้า encode ไม่ใช่หลัง encode จบ -- วัดจากประชุมจริงพบว่าช่วง
    encode กว้าง 62-69 วินาที ส่วนช่วงหลัง encode_done ถึงงานถัดไปมีแค่ ~2 วินาที"""
    config.companion_command = ["prog"]

    def recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        stop_event.wait(timeout=5)
        if on_event:
            on_event("encode_started", {})
        return None

    client, made = _client_with_companion(config, recorder=recorder)
    client.post("/api/session", json={"model": "GLM-5.2"})
    _wait_until(client, lambda b: b["recorder"] == "recording")

    client.post("/api/session/stop")
    _wait_until(client, lambda b: b["recorder"] == "idle")

    assert made[0].calls == ["start", "stop"]


def test_the_companion_is_stopped_even_when_the_encode_stage_never_begins(config):
    """ตัวอัดที่ระเบิดกลางทางต้องไม่ทิ้ง companion ค้างถือทรัพยากรไว้"""
    config.companion_command = ["prog"]

    def exploding_recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        raise RuntimeError("ตัวอัดระเบิด")

    client, made = _client_with_companion(config, recorder=exploding_recorder)
    client.post("/api/session", json={"model": "GLM-5.2"})
    _wait_until(client, lambda b: b["recorder"] == "idle")

    assert made[0].calls == ["start", "stop"]


def test_the_companion_is_stopped_exactly_once(config):
    """encode_started แล้ว finally อีกรอบ -- ต้องไม่กลายเป็นสองครั้ง"""
    config.companion_command = ["prog"]

    def recorder(name, model, cfg, stop_event, on_event=None, mic_muted=None, profile=None, asr_engine=None):
        stop_event.wait(timeout=5)
        if on_event:
            on_event("encode_started", {})
        return None

    client, made = _client_with_companion(config, recorder=recorder)
    client.post("/api/session", json={"model": "GLM-5.2"})
    _wait_until(client, lambda b: b["recorder"] == "recording")
    client.post("/api/session/stop")
    _wait_until(client, lambda b: b["recorder"] == "idle")

    assert made[0].calls.count("stop") == 1


def test_a_companion_that_cannot_start_still_lets_the_room_open(config):
    """กฎข้อเดียวที่สำคัญที่สุดของฟีเจอร์นี้"""
    config.companion_command = ["prog"]

    def factory(command, cwd=None):
        raise OSError("no such file")

    app = create_app(
        config,
        recorder=blocking_recorder,
        worker_probe=lambda: True,
        companion_factory=factory,
    )
    client = app.test_client()

    response = client.post("/api/session", json={"model": "GLM-5.2"})

    assert response.status_code == 201
    assert _wait_until(client, lambda b: b["recorder"] == "recording")["recorder"] == "recording"

    client.post("/api/session/stop")


def test_each_room_gets_its_own_companion(config):
    config.companion_command = ["prog"]
    client, made = _client_with_companion(config)

    for _ in range(2):
        client.post("/api/session", json={"model": "GLM-5.2"})
        _wait_until(client, lambda b: b["recorder"] == "recording")
        client.post("/api/session/stop")
        _wait_until(client, lambda b: b["recorder"] == "idle")

    assert len(made) == 2


def _ev(job, code):
    return {"job": job, "code": code, "ts": "2026-08-08T09:00:00", "level": "info"}


@pytest.mark.parametrize(
    "name,entries,worker_running,expected",
    [
        ("ไม่มีอะไรเลย", [], True, False),
        ("watcher ดับ งานค้างคิว", [_ev("a", "queued")], False, False),
        ("งานรอคิว", [_ev("a", "queued")], True, True),
        ("กำลังถอดเสียง", [_ev("a", "queued"), _ev("a", "transcribe_started")], True, True),
        ("กำลังแยกผู้พูด",
         [_ev("a", "transcribe_started"), _ev("a", "diarize_started")], True, True),
        ("ถึงขั้นสรุปแล้ว",
         [_ev("a", "diarize_started"), _ev("a", "summarize_started")], True, False),
        ("สรุปเสร็จแล้วเริ่มงานใหม่",
         [_ev("a", "summarize_started"), _ev("b", "transcribe_started")], True, True),
        ("จบแล้ว", [_ev("a", "transcribe_started"), _ev("a", "meeting_done")], True, False),
        ("พังแล้ว", [_ev("a", "transcribe_started"), _ev("a", "job_failed")], True, False),
        ("งานหนึ่งจบ อีกงานยังถอด",
         [_ev("a", "meeting_done"), _ev("b", "transcribe_started")], True, True),
        ("สองงานจบหมด", [_ev("a", "meeting_done"), _ev("b", "job_failed")], True, False),
        ("บรรทัดพัง ๆ ปนมา",
         [None, {"job": 1}, {"code": []}, _ev("a", "meeting_done")], True, False),
    ],
)
def test_gpu_is_busy(name, entries, worker_running, expected):
    assert gpu_is_busy(entries, worker_running) is expected
