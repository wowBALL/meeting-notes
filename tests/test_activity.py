from src.activity import activity_path, append, tail, trim


def test_append_then_tail_reads_it_back(tmp_path):
    append(tmp_path, "meet-1", "queued")
    entries = tail(tmp_path)
    assert len(entries) == 1
    assert entries[0]["job"] == "meet-1"
    assert entries[0]["code"] == "queued"
    assert entries[0]["level"] == "info"
    assert entries[0]["params"] == {}
    assert entries[0]["ts"]


def test_append_keeps_the_order_it_was_written(tmp_path):
    for code in ["queued", "transcribe_started", "meeting_done"]:
        append(tmp_path, "meet-1", code)
    assert [e["code"] for e in tail(tmp_path)] == [
        "queued",
        "transcribe_started",
        "meeting_done",
    ]


def test_tail_returns_only_the_last_n(tmp_path):
    for i in range(10):
        append(tmp_path, "meet-1", f"code-{i}")
    entries = tail(tmp_path, limit=3)
    assert [e["code"] for e in entries] == ["code-7", "code-8", "code-9"]


def test_tail_skips_a_corrupt_line_instead_of_failing(tmp_path):
    append(tmp_path, "meet-1", "queued")
    with activity_path(tmp_path).open("a", encoding="utf-8") as handle:
        handle.write("นี่ไม่ใช่ json\n")
    append(tmp_path, "meet-1", "meeting_done")
    assert [e["code"] for e in tail(tmp_path)] == ["queued", "meeting_done"]


def test_tail_returns_empty_when_the_file_does_not_exist(tmp_path):
    assert tail(tmp_path) == []


def test_append_never_raises_when_the_path_cannot_be_written(tmp_path):
    # state/ ถูกยึดที่ด้วยไฟล์ธรรมดา สร้างโฟลเดอร์ไม่ได้
    (tmp_path / "state").write_text("ไม่ใช่โฟลเดอร์", encoding="utf-8")
    append(tmp_path, "meet-1", "queued")  # ต้องไม่ raise


def test_append_stores_the_parameters(tmp_path):
    append(tmp_path, "meet-1", "job_failed", level="error", params={"error": "boom"})
    entry = tail(tmp_path)[0]
    assert entry["level"] == "error"
    assert entry["params"] == {"error": "boom"}


def test_trim_keeps_only_the_last_n_lines(tmp_path):
    for i in range(10):
        append(tmp_path, "meet-1", f"code-{i}")
    trim(tmp_path, keep=4)
    assert [e["code"] for e in tail(tmp_path)] == [
        "code-6",
        "code-7",
        "code-8",
        "code-9",
    ]


def test_trim_does_nothing_when_the_file_is_missing(tmp_path):
    trim(tmp_path, keep=4)  # ต้องไม่ raise
    assert tail(tmp_path) == []
