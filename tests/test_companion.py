"""companion เป็นของเสริม -- ธีมเดียวของไฟล์นี้คือ "พังแล้วต้องเงียบ"."""

import subprocess

from src.companion import Companion


class FakeProc:
    def __init__(self, command, **kwargs):
        self.command = command
        self.kwargs = kwargs
        self.terminated = 0
        self.killed = 0
        self._alive = True

    def poll(self):
        return None if self._alive else 0

    def terminate(self):
        self.terminated += 1
        self._alive = False

    def kill(self):
        self.killed += 1
        self._alive = False

    def wait(self, timeout=None):
        if self._alive:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return 0


def _spy():
    made = []

    def launcher(command, **kwargs):
        proc = FakeProc(command, **kwargs)
        made.append(proc)
        return proc

    return launcher, made


def test_an_empty_command_never_launches_anything():
    launcher, made = _spy()

    Companion([], launcher=launcher).start()

    assert made == []


def test_start_launches_the_configured_command():
    launcher, made = _spy()

    Companion(["prog", "--flag"], launcher=launcher).start()

    assert len(made) == 1
    assert made[0].command == ["prog", "--flag"]


def test_start_twice_does_not_launch_a_second_process():
    """เปิดห้องซ้อนถูกปฏิเสธที่ endpoint อยู่แล้ว แต่ถ้าวันหนึ่งไม่ถูกปฏิเสธ ตัวที่สอง
    จะแย่งทรัพยากรกับตัวแรกโดยไม่มีใครถือ handle ตัวแรกไว้ปิด"""
    launcher, made = _spy()
    companion = Companion(["prog"], launcher=launcher)

    companion.start()
    companion.start()

    assert len(made) == 1


def test_stop_terminates_the_process():
    launcher, made = _spy()
    companion = Companion(["prog"], launcher=launcher)
    companion.start()

    companion.stop()

    assert made[0].terminated == 1
    assert companion.is_running() is False


def test_stop_twice_is_harmless():
    launcher, made = _spy()
    companion = Companion(["prog"], launcher=launcher)
    companion.start()

    companion.stop()
    companion.stop()

    assert made[0].terminated == 1


def test_stop_without_start_is_harmless():
    launcher, _ = _spy()

    Companion(["prog"], launcher=launcher).stop()


def test_a_process_that_ignores_terminate_gets_killed():
    """ทรัพยากรต้องถูกคืนให้ทันขั้นตอนถัดไป -- ตัวที่ไม่ยอมตายเองต้องถูกฆ่า
    ไม่ใช่ปล่อยค้างแล้วหวังว่าจะพอ"""
    launcher, made = _spy()
    companion = Companion(["prog"], launcher=launcher)
    companion.start()
    made[0].terminate = lambda: None  # เมินสัญญาณ แต่ยังมีชีวิตอยู่

    companion.stop()

    assert made[0].killed == 1


def test_a_launcher_that_explodes_does_not_raise():
    """กฎข้อสำคัญที่สุด: ห้องประชุมต้องเปิดได้แม้ companion จะเปิดไม่ติด"""

    def launcher(command, **kwargs):
        raise OSError("no such file")

    companion = Companion(["missing.exe"], launcher=launcher)

    companion.start()

    assert companion.is_running() is False


def test_a_terminate_that_explodes_does_not_raise():
    launcher, made = _spy()
    companion = Companion(["prog"], launcher=launcher)
    companion.start()

    def boom():
        raise OSError("access denied")

    made[0].terminate = boom
    made[0].kill = boom

    companion.stop()

    assert companion.is_running() is False


def test_it_can_start_again_after_stopping():
    """ประชุมถัดไปต้องได้ companion ตัวใหม่"""
    launcher, made = _spy()
    companion = Companion(["prog"], launcher=launcher)

    companion.start()
    companion.stop()
    companion.start()

    assert len(made) == 2


def test_extra_env_reaches_the_child_on_top_of_the_inherited_environment():
    """ส่ง context ให้ลูกทาง env เพื่อไม่ให้ลูกต้องไปไล่อ่าน state ของ service"""
    launcher, made = _spy()

    Companion(["prog"], launcher=launcher).start({"MEETING_ROOM": "standup"})

    assert made[0].kwargs["env"]["MEETING_ROOM"] == "standup"
    assert "PATH" in made[0].kwargs["env"]


def test_the_child_gets_no_console_window():
    """service รันใต้ pythonw ซึ่งไม่มี console -- ลูกที่เป็น console subsystem จะถูก
    Windows 11 โยนให้ Windows Terminal เปิดหน้าต่างใหม่ทุกครั้ง"""
    launcher, made = _spy()

    Companion(["prog"], launcher=launcher).start()

    assert made[0].kwargs["creationflags"] == getattr(
        subprocess, "CREATE_NO_WINDOW", 0
    )


def test_it_never_uses_a_shell():
    """shell=True ทำให้ terminate() ฆ่าแค่ shell ปล่อยลูกจริงค้างถือทรัพยากรไว้"""
    launcher, made = _spy()

    Companion(["prog"], launcher=launcher).start()

    assert made[0].kwargs.get("shell", False) is False
