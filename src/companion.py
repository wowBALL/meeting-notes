"""โปรเซสข้างเคียงที่มีอายุเท่ากับช่วงอัด

โมดูลนี้ตั้งใจไม่รู้ว่าโปรเซสที่รันคืออะไร รู้แค่ว่ามันเป็น "ของเสริม" ซึ่งแปลว่าความ
ล้มเหลวของมันต้องไม่เดินทางกลับไปหาผู้เรียกเลย -- ทุกทางออกของทั้ง start และ stop
จบแบบเงียบ เหลือไว้แค่บรรทัดใน log

ทำไมต้องเป็นโปรเซสลูกไม่ใช่ thread แบบตัวอัด: ตัวอัดแชร์ stop_event กับ service ได้
เพราะเป็นโค้ดใน repo เดียวกัน ส่วนของเสริมเป็นโปรแกรมภายนอกที่ service ไม่รู้จัก และ
อาจถือทรัพยากรระดับ process (เช่นหน่วยความจำการ์ดจอ) ที่คืนได้แน่นอนก็ต่อเมื่อโปรเซส
ตายเท่านั้น
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# เผื่อเวลาให้ปิดตัวเองอย่างสุภาพก่อนใช้กำลัง -- สั้นเพราะผู้เรียกกำลังรอคืนทรัพยากร
# ไปให้ขั้นตอนถัดไป ไม่ใช่รอความเรียบร้อย
_TERMINATE_GRACE_SECONDS = 5

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class Companion:
    def __init__(self, command, cwd=None, launcher=subprocess.Popen):
        self._command = list(command or [])
        self._cwd = cwd
        self._launcher = launcher
        self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, env_extra=None) -> None:
        if not self._command or self.is_running():
            return
        env = dict(os.environ)
        env.update(env_extra or {})
        try:
            self._proc = self._launcher(
                self._command,
                cwd=str(self._cwd) if self._cwd else None,
                env=env,
                shell=False,
                creationflags=_NO_WINDOW,
            )
        except Exception:
            # กว้างโดยตั้งใจ: launcher เป็นของที่ฉีดเข้ามาได้ และไม่มีความล้มเหลวแบบไหน
            # ของมันที่ควรทำให้ห้องประชุมเปิดไม่ได้
            logger.exception("เปิดโปรเซสข้างเคียงไม่สำเร็จ -- ประชุมเดินต่อโดยไม่มีมัน")
            self._proc = None

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            try:
                proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                # ตัวที่เมินสัญญาณต้องถูกฆ่า ไม่ใช่ปล่อยค้าง -- ทรัพยากรที่มันถืออยู่
                # เป็นของที่ขั้นตอนถัดไปกำลังรอใช้
                proc.kill()
                proc.wait(timeout=_TERMINATE_GRACE_SECONDS)
        except Exception:
            logger.exception("ปิดโปรเซสข้างเคียงไม่สำเร็จ")
