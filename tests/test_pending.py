import json

from src.pending import (
    build_pending_speakers,
    find_pending,
    load_all_pending,
    pending_dir,
    resolve_pending,
    write_pending,
)
from src.speakers import Match

MERGED = [
    {"start": 0.0, "end": 20.0, "speaker": "SPEAKER_00", "text": "ประโยคที่ยาวที่สุดของคนแรก"},
    {"start": 20.0, "end": 45.0, "speaker": "SPEAKER_01", "text": "คนที่สองพูดยาวมาก"},
    {"start": 45.0, "end": 47.0, "speaker": "SPEAKER_01", "text": "สั้น"},
]
LABELS = {"SPEAKER_00": "ผู้พูด 1", "SPEAKER_01": "ผู้พูด 2"}
EMBEDDINGS = {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}


def test_build_pending_speakers_reports_everyone_unknown():
    result = build_pending_speakers(MERGED, LABELS, EMBEDDINGS)

    assert [entry["label"] for entry in result] == ["ผู้พูด 1", "ผู้พูด 2"]
    assert result[0]["diarization_id"] == "SPEAKER_00"
    assert result[0]["embedding"] == [1.0, 0.0]
    assert result[0]["speaking_seconds"] == 20.0
    assert result[1]["speaking_seconds"] == 27.0
    assert result[0]["guess"] is None
    assert result[0]["suggested"] is None


def test_build_pending_speakers_skips_people_already_recognized():
    matches = {"SPEAKER_00": Match("id-1", "พี่เอ็ม", 0.91, confident=True)}

    result = build_pending_speakers(MERGED, LABELS, EMBEDDINGS, matches=matches)

    assert [entry["label"] for entry in result] == ["ผู้พูด 2"]


def test_build_pending_speakers_carries_a_mid_confidence_match_as_a_suggestion():
    matches = {"SPEAKER_00": Match("id-1", "พี่เอ็ม", 0.612345, confident=False)}

    result = build_pending_speakers(MERGED, LABELS, EMBEDDINGS, matches=matches)

    assert result[0]["suggested"] == {
        "speaker_id": "id-1",
        "name": "พี่เอ็ม",
        "score": 0.612,
    }


def test_build_pending_speakers_skips_speakers_who_barely_spoke():
    merged = [{"start": 0.0, "end": 4.0, "speaker": "SPEAKER_00", "text": "ครับ"}]

    result = build_pending_speakers(merged, {"SPEAKER_00": "ผู้พูด 1"}, EMBEDDINGS)

    # เวกเตอร์จากเสียงไม่กี่วินาทีเชื่อถือไม่ได้พอจะเก็บเข้าทะเบียนถาวร
    assert result == []


def test_build_pending_speakers_skips_speakers_without_a_usable_embedding():
    merged = [
        {"start": 0.0, "end": 30.0, "speaker": "SPEAKER_UNKNOWN", "text": "ไม่มีเวกเตอร์"},
        {"start": 30.0, "end": 60.0, "speaker": "SPEAKER_00", "text": "เวกเตอร์ศูนย์"},
    ]
    labels = {"SPEAKER_UNKNOWN": "ผู้พูด 1", "SPEAKER_00": "ผู้พูด 2"}

    result = build_pending_speakers(merged, labels, {"SPEAKER_00": [0.0, 0.0]})

    assert result == []


def test_build_pending_speakers_keeps_the_longest_lines_as_samples():
    merged = [
        {"start": 0.0, "end": 12.0, "speaker": "SPEAKER_00", "text": "สั้น"},
        {"start": 12.0, "end": 24.0, "speaker": "SPEAKER_00", "text": "ประโยคกลาง ๆ ยาวขึ้นมาหน่อย"},
        {"start": 24.0, "end": 36.0, "speaker": "SPEAKER_00", "text": "ประโยคนี้ยาวที่สุดในบรรดาสามประโยคนี้แน่นอน"},
        {"start": 36.0, "end": 48.0, "speaker": "SPEAKER_00", "text": "อีกอัน"},
    ]

    result = build_pending_speakers(merged, {"SPEAKER_00": "ผู้พูด 1"}, {"SPEAKER_00": [1.0, 0.0]})

    samples = result[0]["samples"]
    assert len(samples) == 3
    # เรียงตามเวลาเพื่อให้ผู้อ่านตามบทสนทนาได้ ไม่ใช่เรียงตามความยาว
    assert [sample["start"] for sample in samples] == sorted(sample["start"] for sample in samples)
    assert any("ยาวที่สุด" in sample["text"] for sample in samples)


def test_write_pending_then_load_all_pending_round_trips(tmp_path):
    speakers = build_pending_speakers(MERGED, LABELS, EMBEDDINGS)

    path = write_pending(tmp_path, "2026-07-28_10-30-standup", "standup.ogg", speakers)

    assert path == pending_dir(tmp_path) / "2026-07-28_10-30-standup.json"
    loaded = load_all_pending(tmp_path)
    assert len(loaded) == 1
    assert loaded[0]["meeting_dir"] == "2026-07-28_10-30-standup"
    assert loaded[0]["audio_file"] == "standup.ogg"
    assert [entry["label"] for entry in loaded[0]["speakers"]] == ["ผู้พูด 1", "ผู้พูด 2"]


def test_write_pending_writes_nothing_when_there_is_nobody_to_name(tmp_path):
    assert write_pending(tmp_path, "m", "a.ogg", []) is None
    assert load_all_pending(tmp_path) == []


def test_load_all_pending_skips_a_corrupt_file(tmp_path):
    write_pending(tmp_path, "ดี", "a.ogg", build_pending_speakers(MERGED, LABELS, EMBEDDINGS))
    (pending_dir(tmp_path) / "พัง.json").write_text("ไม่ใช่ json", encoding="utf-8")

    loaded = load_all_pending(tmp_path)

    assert [entry["meeting_dir"] for entry in loaded] == ["ดี"]


def test_find_pending_returns_the_entry_without_changing_the_file(tmp_path):
    write_pending(tmp_path, "m1", "a.ogg", build_pending_speakers(MERGED, LABELS, EMBEDDINGS))

    entry = find_pending(tmp_path, "m1", "ผู้พูด 2")

    assert entry["diarization_id"] == "SPEAKER_01"
    assert entry["embedding"] == [0.0, 1.0]
    assert len(load_all_pending(tmp_path)[0]["speakers"]) == 2


def test_find_pending_returns_none_for_an_unknown_meeting_or_label(tmp_path):
    write_pending(tmp_path, "m1", "a.ogg", build_pending_speakers(MERGED, LABELS, EMBEDDINGS))

    assert find_pending(tmp_path, "ไม่มีจริง", "ผู้พูด 1") is None
    assert find_pending(tmp_path, "m1", "ผู้พูด 9") is None


def test_find_pending_refuses_a_meeting_name_that_escapes_the_folder(tmp_path):
    assert find_pending(tmp_path, "../../etc/passwd", "ผู้พูด 1") is None
    assert find_pending(tmp_path, "sub/dir", "ผู้พูด 1") is None


def test_resolve_pending_removes_one_speaker_and_keeps_the_rest(tmp_path):
    write_pending(tmp_path, "m1", "a.ogg", build_pending_speakers(MERGED, LABELS, EMBEDDINGS))

    assert resolve_pending(tmp_path, "m1", "ผู้พูด 1") is True

    remaining = load_all_pending(tmp_path)
    assert [entry["label"] for entry in remaining[0]["speakers"]] == ["ผู้พูด 2"]


def test_resolve_pending_deletes_the_file_once_everyone_is_named(tmp_path):
    write_pending(tmp_path, "m1", "a.ogg", build_pending_speakers(MERGED, LABELS, EMBEDDINGS))

    resolve_pending(tmp_path, "m1", "ผู้พูด 1")
    resolve_pending(tmp_path, "m1", "ผู้พูด 2")

    assert load_all_pending(tmp_path) == []
    assert not (pending_dir(tmp_path) / "m1.json").exists()


def test_resolve_pending_returns_false_when_there_is_nothing_to_remove(tmp_path):
    assert resolve_pending(tmp_path, "ไม่มีจริง", "ผู้พูด 1") is False
