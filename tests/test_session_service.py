import time

import pytest

from src.config import Config
from src.session_service import create_app


def make_config(tmp_path):
    return Config(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
        anthropic_api_key="sk-ant-test",
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
