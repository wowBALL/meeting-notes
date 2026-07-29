import json
from pathlib import Path
from unittest.mock import patch

from src import enroll
from src.config import Config
from src.watcher import is_file_stable, scan_inbox, watch_loop


def make_config(tmp_path) -> Config:
    return Config(
        base_dir=tmp_path,
        inbox_dir=tmp_path / "inbox",
        failed_dir=tmp_path / "failed",
        meetings_dir=tmp_path / "meetings",
        hf_token="hf-test-token",
    )


def test_scan_inbox_returns_only_audio_files_sorted(tmp_path):
    inbox_dir = tmp_path / "inbox"
    inbox_dir.mkdir()
    (inbox_dir / "b.mp3").write_bytes(b"x")
    (inbox_dir / "a.wav").write_bytes(b"x")
    (inbox_dir / "notes.txt").write_bytes(b"x")

    result = scan_inbox(inbox_dir)

    assert result == [inbox_dir / "a.wav", inbox_dir / "b.mp3"]


def test_scan_inbox_returns_empty_list_when_dir_missing(tmp_path):
    assert scan_inbox(tmp_path / "does-not-exist") == []


def test_scan_inbox_ignores_session_directories_and_accepts_ogg(tmp_path):
    session = tmp_path / ".session-meet1"
    session.mkdir()
    (session / "part0001.wav").write_bytes(b"x")
    (tmp_path / "done.ogg").write_bytes(b"x")

    assert scan_inbox(tmp_path) == [tmp_path / "done.ogg"]


def test_is_file_stable_true_for_unchanging_file(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with patch("time.sleep"):
        assert is_file_stable(audio_path, check_interval=0) is True


def test_is_file_stable_false_for_empty_file(tmp_path):
    audio_path = tmp_path / "empty.mp3"
    audio_path.write_bytes(b"")

    with patch("time.sleep"):
        assert is_file_stable(audio_path, check_interval=0) is False


def test_watch_loop_processes_stable_files_once_with_single_pass(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=None, whisper_model=None
    )


def test_watch_loop_threads_diarization_pipeline_to_process_file(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    sentinel_pipeline = object()

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True, diarization_pipeline=sentinel_pipeline)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=sentinel_pipeline, whisper_model=None
    )


def test_watch_loop_threads_whisper_model_to_process_file(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    sentinel_model = object()

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True, whisper_model=sentinel_model)

    mock_process_file.assert_called_once_with(
        audio_path, config, diarization_pipeline=None, whisper_model=sentinel_model
    )


def test_watch_loop_skips_unstable_files(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    with (
        patch("src.watcher.is_file_stable", return_value=False),
        patch("src.watcher.process_file") as mock_process_file,
    ):
        watch_loop(config, single_pass=True)

    mock_process_file.assert_not_called()


def make_enroll_audio(tmp_path, name="สมชาย.ogg"):
    directory = tmp_path / "enroll"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"fake audio")
    return path


def test_watch_loop_analyzes_a_requested_enrollment_clip(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    sentinel_pipeline = object()

    analyzed = {"status": "ok", "embedding": [0.5], "speaker_count": 1}
    with patch("src.watcher.enroll.analyze", return_value=analyzed) as mock_analyze:
        watch_loop(
            config, single_pass=True, diarization_pipeline=sentinel_pipeline
        )

    assert mock_analyze.call_count == 1
    assert mock_analyze.call_args.kwargs["pipeline"] is sentinel_pipeline
    written = json.loads(
        (tmp_path / "enroll" / "สมชาย.ogg.result.json").read_text(encoding="utf-8")
    )
    assert written["status"] == "ok"
    assert written["audio_file"] == "สมชาย.ogg"


def test_watch_loop_does_not_analyze_a_clip_nobody_requested(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    make_enroll_audio(tmp_path)

    with patch("src.watcher.enroll.analyze") as mock_analyze:
        watch_loop(config, single_pass=True)

    mock_analyze.assert_not_called()


def test_watch_loop_never_writes_the_speaker_registry(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")

    with patch(
        "src.watcher.enroll.analyze",
        return_value={"status": "ok", "embedding": [0.5]},
    ):
        watch_loop(config, single_pass=True)

    # ทะเบียนถูกเขียนโดย session_service เมื่อมีคนกดยืนยันเท่านั้น การจับคู่ที่ผิด
    # ต้องไม่ฝังตัวอย่างเสียงผิดคนลงโปรไฟล์ถาวรโดยไม่มีมนุษย์เห็นเลยสักครั้ง
    assert not (tmp_path / "speakers" / "registry.json").exists()


def test_a_failing_enroll_job_does_not_stop_the_inbox_from_being_processed(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "meeting.ogg"
    audio_path.write_bytes(b"fake audio data")
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
        patch("src.watcher.enroll.analyze", side_effect=OSError("disk on fire")),
    ):
        watch_loop(config, single_pass=True)

    # ประชุมที่อัดซ้ำไม่ได้ต้องไม่โดนงาน enroll ที่ทำใหม่ได้เสมอทำให้พัง
    mock_process_file.assert_called_once()


def test_a_failing_result_write_does_not_retry_the_same_clip_forever(tmp_path):
    """finding 3: write_result ที่ raise ตลอด (ดิสก์เต็ม/ไฟล์ถูกล็อกค้าง) ต้องไม่ทำให้
    ใบสั่งงานค้างอยู่ -- ไม่งั้น pending_requests จะยังเห็น "สั่งแล้วแต่ยังไม่มีผล" แล้ว
    วิเคราะห์ซ้ำ (GPU เต็มรอบ) ทุก poll ไปเรื่อย ๆ ไม่มีที่สิ้นสุด
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")

    with (
        patch(
            "src.watcher.enroll.analyze",
            return_value={"status": "ok", "embedding": [0.5]},
        ),
        patch("src.watcher.enroll.write_result", side_effect=OSError("disk full")),
    ):
        watch_loop(config, single_pass=True)

    assert enroll.pending_requests(tmp_path) == []


def test_a_failing_result_write_falls_back_to_a_minimal_failure_result(tmp_path):
    """finding 3: ถ้าเขียนผลจริงล้มเหลวแต่การเขียนผลล้มเหลวขั้นต่ำ (rejected) รอบสอง
    สำเร็จ ผู้ใช้ควรเห็นข้อความว่าวิเคราะห์ไม่สำเร็จ แทนที่จะเห็น "กำลังวิเคราะห์" ค้าง
    ตลอดไปหรือไฟล์เงียบหายกลับไปเป็น idle โดยไม่บอกอะไรเลย
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")

    real_write_result = enroll.write_result
    calls = []

    def flaky_write_result(base_dir, audio_file, analyzed, pre_analysis_stat=None, now=None):
        calls.append(analyzed)
        if len(calls) == 1:
            raise OSError("disk full")
        return real_write_result(
            base_dir, audio_file, analyzed, pre_analysis_stat=pre_analysis_stat, now=now
        )

    with (
        patch(
            "src.watcher.enroll.analyze",
            return_value={"status": "ok", "embedding": [0.5]},
        ),
        patch("src.watcher.enroll.write_result", side_effect=flaky_write_result),
    ):
        watch_loop(config, single_pass=True)

    result = enroll.read_result(tmp_path, "สมชาย.ogg")
    assert result is not None
    assert result["status"] == "rejected"
    assert result["reason"] == "analysis_failed"


def test_process_enroll_requests_clears_sidecars_for_a_result_whose_file_was_replaced_mid_analysis(
    tmp_path,
):
    """CRITICAL A + finding 4: watcher ต้อง stat ไฟล์ก่อนส่งเข้า analyze() แล้วส่ง
    (size, mtime) นั้นต่อให้ write_result -- ถ้าไม่ทำ ไฟล์ที่ถูกแทนที่ระหว่าง diarization
    (ใช้เวลานาน) จะถูก write_result stat() ใหม่ตอนเขียนผล แล้วผูก embedding ของไบต์ชุดเก่า
    เข้ากับไฟล์ใหม่เงียบ ๆ เพราะ record ที่เพิ่งเขียนกับไฟล์บนดิสก์ ณ ตอนนั้น "ตรง" กันเป๊ะ

    finding 4: จุดตันเดิมของการเขียน "rejected" ค้างไว้คือ audio_size/audio_mtime ที่บันทึก
    จะผูกกับไฟล์ใหม่ (ถูกต้องสมบูรณ์ คนละคน) ที่ยังอยู่บนดิสก์จริง ทำให้ rejected นั้น
    "ยืนยันได้" ตลอดกาลสำหรับไฟล์ใหม่ -- ต้องอ่าน sidecar ดิบเพื่อยืนยันว่าไม่มีทั้งคู่เหลือ
    อยู่เลย (ไม่ใช่แค่ผ่าน read_result ซึ่งจะคืน None ทั้งสองกรณีอยู่ดี ไม่บอกว่าไฟล์ถูกล้าง
    จริงหรือแค่ยัง "ตรง" กับไฟล์เดิมที่ไม่มีอยู่แล้ว)
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = make_enroll_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")

    def analyze_then_replace(path, pipeline, model):
        # จำลองผู้ใช้แทนที่ไฟล์ผ่าน Explorer ระหว่างที่ diarization กำลังทำงานอยู่พอดี
        audio_path.write_bytes(b"a completely different recording, replaced mid-analysis")
        return {
            "status": "ok",
            "embedding": [0.5],
            "speaker_count": 1,
            "suggested_name": "สมชาย",
        }

    with patch("src.watcher.enroll.analyze", side_effect=analyze_then_replace):
        watch_loop(config, single_pass=True)

    assert not (tmp_path / "enroll" / "สมชาย.ogg.result.json").exists()
    assert not (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()
    # การ์ดต้องกลับไป idle ให้วิเคราะห์ไฟล์ที่วางใหม่ได้เอง ไม่ใช่ทางตัน "ใช้ไม่ได้"
    assert enroll.list_entries(tmp_path)[0]["state"] == "idle"


def test_process_enroll_requests_logs_a_discard_not_a_success_when_binding_fails(
    tmp_path, caplog
):
    """finding 4 ของรีวิวรอบนี้: เดิม log INFO "Analyzed ... -> ok" ยิงแบบไม่มีเงื่อนไขเสมอ
    หลังเรียก write_result แม้ตอนที่ write_result คืน None เพราะผลถูกทิ้งไป (ผูกกับไฟล์เสียง
    ไม่ได้) -- ผู้คุมระบบจะเห็น WARNING เรื่องผูกไม่ได้ตามด้วย INFO อ้างว่าสำเร็จทันที ต้อง log
    ตามค่าที่ write_result คืนจริง ไม่ใช่ log "-> ok" ทุกครั้งที่ analyze() รายงาน status ok
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = make_enroll_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")

    def analyze_then_replace(path, pipeline, model):
        audio_path.write_bytes(b"a completely different recording, replaced mid-analysis")
        return {"status": "ok", "embedding": [0.5], "suggested_name": "สมชาย"}

    with (
        patch("src.watcher.enroll.analyze", side_effect=analyze_then_replace),
        caplog.at_level("INFO"),
    ):
        watch_loop(config, single_pass=True)

    assert not any(
        "Analyzed enrollment clip" in record.message and "-> ok" in record.message
        for record in caplog.records
    )
    assert any(
        record.levelname == "WARNING" and "Discarded" in record.message
        for record in caplog.records
    )


def test_process_enroll_requests_skips_analysis_when_the_pre_stat_fails(tmp_path, monkeypatch):
    """finding 1 (blocks merge): เดิมตรงนี้ pre_stat ที่ stat() ไม่สำเร็จถูกตั้งเป็น None
    แล้ว "เดินหน้าวิเคราะห์ต่อ" อยู่ดี -- write_result.เดิมเช็คแค่ pre_analysis_stat is not
    None ทำให้ None ข้ามการเปรียบเทียบไปเงียบ ๆ เท่ากับปิดการป้องกันทั้งหมดที่ CRITICAL A
    เพิ่มเข้ามาพอดีตอนที่ต้องการมันที่สุด (PermissionError ชั่วคราวมักคลายล็อกเองไม่กี่ร้อย
    ms ให้หลัง ทำให้ analyze() อ่านไฟล์สำเร็จตามปกติในสถานการณ์จริง)

    ต้องไม่วิเคราะห์ไฟล์ที่ผูกกับไบต์อะไรไม่ได้ตั้งแต่ต้นแทน -- ทดสอบว่า analyze() ไม่ถูก
    เรียกเลยเมื่อ pre-stat พัง, ไม่มีผลถูกเขียน, และใบสั่งงานยังอยู่ครบให้รอบ poll หน้าลองใหม่
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = make_enroll_audio(tmp_path, "สมชาย.ogg")
    enroll.write_request(tmp_path, "สมชาย.ogg")
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self == audio_path:
            raise PermissionError("locked by antivirus")
        return real_stat(self, *args, **kwargs)

    # pending_requests() เองก็เรียก scan_audio -> is_file() บนไฟล์เดียวกัน (finding 2/3) --
    # mock มันตรงนี้ให้คืนรายการตรง ๆ เพื่อแยกให้ชัดว่าเทสต์นี้ทดสอบเฉพาะ pre-stat ของ
    # watcher เอง (finding 1) ไม่ปนกับ guard อื่นที่ finding 2/3 เพิ่งเพิ่มเข้ามาในไฟล์เดียวกัน
    with patch("src.watcher.enroll.pending_requests", return_value=["สมชาย.ogg"]):
        monkeypatch.setattr(Path, "stat", flaky_stat)
        with patch("src.watcher.enroll.analyze") as mock_analyze:
            watch_loop(config, single_pass=True)

    mock_analyze.assert_not_called()
    assert not (tmp_path / "enroll" / "สมชาย.ogg.result.json").exists()
    # ใบสั่งงานต้องยังอยู่ครบ (ไม่ถูกแตะเลย) -- รอบ poll หน้าจะลองใหม่เองเมื่อล็อกคลาย
    assert (tmp_path / "enroll" / "สมชาย.ogg.request.json").exists()


def test_pending_requests_raising_does_not_stop_the_inbox_from_being_processed(tmp_path):
    """finding 3: pending_requests() นั่งอยู่ใน for header ของ process_enroll_requests
    นอก try ใด ๆ -- ถ้ามันพัง exception จะหลุดออกจาก watch_loop ทั้งก้อน ฆ่า watcher
    ทิ้งพร้อมงานประมวลผลไฟล์ประชุมใน inbox/ ไปด้วย
    """
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    audio_path = config.inbox_dir / "meeting.ogg"
    audio_path.write_bytes(b"fake audio data")

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file") as mock_process_file,
        patch(
            "src.watcher.enroll.pending_requests", side_effect=OSError("disk on fire")
        ),
    ):
        watch_loop(config, single_pass=True)  # ต้องไม่ raise

    mock_process_file.assert_called_once()


def test_the_inbox_is_processed_before_any_enroll_work(tmp_path):
    config = make_config(tmp_path)
    config.inbox_dir.mkdir(parents=True)
    (config.inbox_dir / "meeting.ogg").write_bytes(b"fake audio data")
    make_enroll_audio(tmp_path)
    enroll.write_request(tmp_path, "สมชาย.ogg")
    order = []

    with (
        patch("src.watcher.is_file_stable", return_value=True),
        patch("src.watcher.process_file", side_effect=lambda *a, **k: order.append("inbox")),
        patch(
            "src.watcher.enroll.analyze",
            side_effect=lambda *a, **k: order.append("enroll") or {"status": "ok"},
        ),
    ):
        watch_loop(config, single_pass=True)

    # ลำดับนี้ load-bearing: การประชุมที่อัดซ้ำไม่ได้ต้องได้ GPU ก่อนงานที่ทำใหม่ได้
    assert order == ["inbox", "enroll"]
