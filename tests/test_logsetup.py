import logging

import pytest

from src.logsetup import UI_LOG, WATCHER_LOG, configure_logging, log_path


@pytest.fixture(autouse=True)
def restore_root_logger():
    """configure_logging ใช้ force=True ซึ่งยึด root logger ทั้งตัว -- ไม่คืนให้ pytest
    แล้วเทสต์ตัวถัดไปที่พึ่ง caplog จะพังแบบงง ๆ"""
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    yield
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    for handler in saved_handlers:
        root.addHandler(handler)
    root.setLevel(saved_level)


def test_thai_log_lines_survive_the_round_trip(tmp_path):
    """กับดักที่ทำให้ต้องมีไฟล์นี้: Windows default เป็น cp1252 ซึ่ง encode ไทยไม่ได้
    handler ที่ไม่ระบุ encoding จะโยน UnicodeEncodeError ตอนเขียนบรรทัดแรก แปลว่ากลไก
    ที่สร้างมาเพื่อจับ error กลายเป็นตัวสร้าง error เสียเอง"""
    path = configure_logging(tmp_path, WATCHER_LOG)
    assert path == log_path(tmp_path, WATCHER_LOG)

    message = "สรุปช่วงนี้ล้มเหลว: หมดเวลารออ่านคำตอบ"
    logging.getLogger("src.summarize").error(message)
    for handler in logging.getLogger().handlers:
        handler.flush()

    written = path.read_text(encoding="utf-8")
    assert message in written
    # ชื่อโมดูลต้องติดมาด้วย ไม่งั้นเวลาสืบสวนจะไม่รู้ว่าบรรทัดไหนมาจากไหน
    assert "src.summarize" in written
    assert "ERROR" in written


def test_console_still_logs_when_the_file_cannot_be_opened(tmp_path):
    """ดิสก์เต็มหรือโฟลเดอร์เขียนไม่ได้ ต้องไม่ทำให้อัดประชุมไม่ได้ -- log ลงไฟล์เป็น
    เครื่องมือช่วยวินิจฉัย ไม่ใช่ความสามารถหลัก (เหตุผลเดียวกับ activity.append)"""
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("", encoding="utf-8")

    path = configure_logging(blocker, WATCHER_LOG)

    assert path is None
    stream_handlers = [
        h
        for h in logging.getLogger().handlers
        if isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
    ]
    assert stream_handlers, "console handler ต้องยังอยู่แม้เปิดไฟล์ไม่ได้"


def test_watcher_and_ui_write_to_separate_files():
    """สอง process เขียนไฟล์เดียวกันจะพังตอน RotatingFileHandler หมุนไฟล์ (rename บน
    Windows ล้มเมื่ออีกฝั่งถือ handle ค้าง) -- อาการโผล่เป็นครั้งคราวหลังใช้ไปหลายเดือน
    แล้วหาต้นเหตุยาก"""
    assert WATCHER_LOG != UI_LOG


def test_log_file_lands_next_to_the_activity_feed(tmp_path):
    from src.activity import activity_path

    assert log_path(tmp_path, WATCHER_LOG).parent == activity_path(tmp_path).parent
