import json
import time

import pytest

from src.config import Config
from src.pending import build_pending_speakers, pending_dir, write_pending
from src.session_service import create_app
from src.speakers import add_sample, load_registry, save_registry


def make_config(tmp_path):
    return Config(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
        hf_token="hf-test-token",
    )


def blocking_recorder(name, model, config, stop_event, on_event=None):
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


def test_elapsed_seconds_counts_up_while_recording(client):
    client.post("/api/session", json={"model": "claude-opus-5", "name": "standup"})

    time.sleep(1.1)

    assert client.get("/api/state").get_json()["elapsed_seconds"] >= 1

    client.post("/api/session/stop")


def test_warnings_from_the_recorder_reach_the_state(config):
    def warning_recorder(name, model, cfg, stop_event, on_event=None):
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

    def crashing_recorder(name, model, cfg, stop_event, on_event=None):
        raise RuntimeError("พัง")

    app = create_app(config, recorder=crashing_recorder, worker_probe=lambda: True)
    client = app.test_client()
    client.post("/api/session", json={"model": "claude-opus-5", "name": "x"})

    body = _wait_until(client, lambda b: b["recorder"] == "idle")

    assert body["recorder"] == "idle"


def test_state_includes_the_activity_log(client, config):
    from src.activity import append

    append(config.base_dir, "meet-1", "queued")

    body = client.get("/api/state").get_json()

    assert body["activity"][-1]["code"] == "queued"


def test_recorder_events_land_in_the_activity_log(client, config):
    from src.activity import tail

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
    from src.activity import append

    append(config.base_dir, "meet-1", "transcribe_started")

    thai = client.get("/api/state").get_json()["activity"][-1]["text"]
    english = client.get("/api/state?lang=en").get_json()["activity"][-1]["text"]

    assert thai == "กำลังถอดเสียง"
    assert english == "Transcribing"


def test_warning_text_is_rendered_too(config):
    def warning_recorder(name, model, cfg, stop_event, on_event=None):
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


def _queue_two_speakers(config, meeting="2026-07-28_10-30-standup"):
    write_pending(
        config.base_dir,
        meeting,
        "standup.ogg",
        build_pending_speakers(_PENDING_MERGED, _PENDING_LABELS, _PENDING_EMBEDDINGS),
    )
    return meeting


def test_pending_speakers_endpoint_is_empty_by_default(client):
    body = client.get("/api/speakers/pending").get_json()

    assert body == {"meetings": []}


def test_pending_speakers_endpoint_lists_the_queue(client, config):
    meeting = _queue_two_speakers(config)

    body = client.get("/api/speakers/pending").get_json()

    assert len(body["meetings"]) == 1
    assert body["meetings"][0]["meeting_dir"] == meeting
    assert body["meetings"][0]["audio_file"] == "standup.ogg"
    labels = [entry["label"] for entry in body["meetings"][0]["speakers"]]
    assert labels == ["ผู้พูด 1", "ผู้พูด 2"]


def test_pending_speakers_endpoint_never_ships_the_voice_vectors(client, config):
    # เบราว์เซอร์ไม่ต้องใช้เวกเตอร์เลย และมันคือข้อมูล biometric -- ส่งออกไปเปล่า ๆ
    # คือเพิ่มที่ที่มันอาจรั่วโดยไม่ได้อะไรกลับมา
    _queue_two_speakers(config)

    body = client.get("/api/speakers/pending").get_json()

    for speaker in body["meetings"][0]["speakers"]:
        assert "embedding" not in speaker
    # ตรวจทั้งก้อนด้วย เผื่อเวกเตอร์ไปโผล่ใต้คีย์อื่นที่ยังไม่มีในวันนี้
    assert "embedding" not in json.dumps(body)


def test_speakers_endpoint_lists_names_and_sample_counts(client, config):
    registry = add_sample([], "พี่เอ็ม", [1.0, 0.0], source="m1")
    registry = add_sample(registry, "พี่เอ็ม", [0.9, 0.1], source="m2")
    save_registry(config.base_dir, registry)

    body = client.get("/api/speakers").get_json()

    assert body["speakers"] == [
        {"id": registry[0]["id"], "name": "พี่เอ็ม", "sample_count": 2}
    ]


def test_deleting_a_speaker_removes_them_from_the_registry(client, config):
    registry = add_sample([], "พี่เอ็ม", [1.0, 0.0], source="m1")
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
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "พี่เอ็ม"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "renamed": True, "name": "พี่เอ็ม"}
    registry = load_registry(config.base_dir)
    assert registry[0]["name"] == "พี่เอ็ม"
    assert registry[0]["samples"][0]["embedding"] == [0.0, 1.0]
    assert registry[0]["samples"][0]["source"] == meeting
    transcript = (meeting_dir / "transcript.md").read_text(encoding="utf-8")
    assert "**พี่เอ็ม** [00:30]: ครับผม ผมเห็นด้วย" in transcript
    assert "**ผู้พูด 1** [00:00]" in transcript


def test_confirming_a_name_takes_that_speaker_out_of_the_queue(client, config):
    meeting = _queue_two_speakers(config)
    _saved_transcript_for(config, meeting)

    client.post(
        "/api/speakers/confirm",
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "พี่เอ็ม"},
    )

    body = client.get("/api/speakers/pending").get_json()
    assert [s["label"] for s in body["meetings"][0]["speakers"]] == ["ผู้พูด 1"]


def test_confirming_an_existing_person_by_id_adds_a_second_sample(client, config):
    registry = add_sample([], "พี่เอ็ม", [0.9, 0.1], source="เมื่อวาน")
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
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "พี่เอ็ม"},
    )

    assert response.get_json() == {"ok": True, "renamed": False, "name": "พี่เอ็ม"}
    assert load_registry(config.base_dir)[0]["name"] == "พี่เอ็ม"


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
        json={"meeting": meeting, "label": "ผู้พูด 1", "name": "พี่เอ็ม"},
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
        json={"meeting": meeting, "label": "ผู้พูด 1", "name": "พี่เอ็ม"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "bad_embedding"
    assert load_registry(config.base_dir) == []


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
        json={"meeting": meeting, "label": "ผู้พูด 2", "name": "พี่เอ็ม"},
    )

    assert response.status_code == 200
    assert response.get_json() == {"ok": True, "renamed": True, "name": "พี่เอ็ม"}
    assert load_registry(config.base_dir)[0]["name"] == "พี่เอ็ม"


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
