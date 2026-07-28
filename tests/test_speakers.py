import json
from unittest.mock import patch

import pytest

from src.speakers import (
    clean_name,
    cosine_similarity,
    is_usable_embedding,
    load_registry,
    registry_path,
    save_registry,
)


def test_load_registry_returns_empty_list_when_file_missing(tmp_path):
    assert load_registry(tmp_path) == []


def test_save_then_load_registry_round_trips(tmp_path):
    speakers = [
        {
            "id": "abc123",
            "name": "พี่เอ็ม",
            "samples": [
                {"embedding": [1.0, 0.0], "source": "2026-07-28_10-30-standup", "added": "2026-07-28"}
            ],
        }
    ]

    save_registry(tmp_path, speakers)

    assert load_registry(tmp_path) == speakers
    assert registry_path(tmp_path) == tmp_path / "speakers" / "registry.json"


def test_save_registry_writes_utf8_json_with_a_version(tmp_path):
    save_registry(tmp_path, [{"id": "a", "name": "พี่เอ็ม", "samples": []}])

    payload = json.loads(registry_path(tmp_path).read_text(encoding="utf-8"))

    assert payload["version"] == 1
    # ห้าม escape เป็น \uXXXX -- ชื่อไทยต้องอ่านออกเมื่อเปิดไฟล์ดูด้วยตา
    assert "พี่เอ็ม" in registry_path(tmp_path).read_text(encoding="utf-8")


def test_load_registry_returns_empty_list_when_file_is_not_json(tmp_path):
    path = registry_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("ไม่ใช่ json", encoding="utf-8")

    assert load_registry(tmp_path) == []


def test_load_registry_drops_entries_with_the_wrong_shape(tmp_path):
    path = registry_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": [
                    {"id": "a", "name": "ดี", "samples": []},
                    {"id": "b", "samples": []},
                    "ไม่ใช่ dict",
                    {"id": "c", "name": "ไม่มี samples"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_registry(tmp_path) == [{"id": "a", "name": "ดี", "samples": []}]


def test_save_registry_retries_when_windows_holds_the_old_file(tmp_path):
    # WinError 32: ตัวสแกนไวรัส/indexer จับไฟล์ที่เพิ่งปิดไปค้างได้ราวหนึ่งวินาที
    # การเขียนครั้งเดียวแล้วยอมแพ้คือวิธีทำให้ผู้ใช้เสียชื่อที่เพิ่งตั้งไปเฉย ๆ
    save_registry(tmp_path, [])
    attempts = {"count": 0}
    real_replace = type(registry_path(tmp_path)).replace

    def flaky_replace(self, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("WinError 32")
        return real_replace(self, target)

    with patch("pathlib.Path.replace", flaky_replace), patch("time.sleep"):
        save_registry(tmp_path, [{"id": "a", "name": "ดี", "samples": []}])

    assert attempts["count"] == 3
    assert load_registry(tmp_path) == [{"id": "a", "name": "ดี", "samples": []}]


def test_save_registry_raises_when_the_file_stays_locked(tmp_path):
    with (
        patch("pathlib.Path.replace", side_effect=PermissionError("WinError 32")),
        patch("time.sleep"),
        pytest.raises(PermissionError),
    ):
        save_registry(tmp_path, [])


def test_clean_name_strips_markdown_and_newlines():
    assert clean_name("  พี่เอ็ม  ") == "พี่เอ็ม"
    assert clean_name("พี่\nเอ็ม") == "พี่ เอ็ม"
    assert clean_name("**พี่เอ็ม**") == "พี่เอ็ม"
    assert clean_name("[พี่เอ็ม]") == "พี่เอ็ม"
    assert clean_name("") == ""


def test_clean_name_caps_the_length():
    assert len(clean_name("ก" * 200)) == 60


def test_is_usable_embedding_rejects_the_shapes_pyannote_can_hand_back():
    assert is_usable_embedding([1.0, 0.0]) is True
    # pyannote pad ศูนย์เมื่อจำนวน label มากกว่าจำนวน centroid -- เวกเตอร์ศูนย์ล้วน
    # ไม่มีทิศทาง เก็บเข้าทะเบียนแล้วจะ match กับทุกอย่างที่เป็นศูนย์เหมือนกัน
    assert is_usable_embedding([0.0, 0.0]) is False
    assert is_usable_embedding([]) is False
    assert is_usable_embedding(None) is False
    assert is_usable_embedding("ไม่ใช่เวกเตอร์") is False
    assert is_usable_embedding([1.0, "x"]) is False


def test_cosine_similarity_scores_identical_orthogonal_and_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    # ขนาดไม่มีผล ทิศทางเท่านั้นที่นับ
    assert cosine_similarity([1.0, 1.0], [5.0, 5.0]) == pytest.approx(1.0)


def test_cosine_similarity_returns_zero_for_unusable_input():
    assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0
