import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


class TyphoonVenvMissing(RuntimeError):
    pass


def _typhoon_python(base_dir: Path) -> Path:
    python_exe = base_dir / ".typhoon_venv" / "Scripts" / "python.exe"
    if not python_exe.is_file():
        raise TyphoonVenvMissing(
            f"ไม่พบ {python_exe} -- ต้องสร้าง .typhoon_venv ก่อนใช้ ASR_ENGINE=typhoon "
            "(pip install typhoon-asr silero-vad ใน venv แยกที่ base_dir/.typhoon_venv)"
        )
    return python_exe


def transcribe_audio_typhoon(wav_path: Path, base_dir: Path) -> list[dict]:
    """เหมือน transcribe_audio() แต่ใช้ Typhoon ASR (VAD-chunked) แทน large-v3

    รันเป็น subprocess แยกโดยเจตนา ไม่ import nemo/typhoon-asr เข้ามาใน process
    นี้ตรงๆ -- ทั้งสองแพ็กเกจอยู่ใน .typhoon_venv คนละตัวกับ .venv ที่ process นี้
    รันอยู่ (ดู tools/typhoon_worker.py สำหรับเหตุผลเต็ม) คืนรูปแบบเดียวกับ
    transcribe_audio() ทุกประการ (list ของ {"start","end","text"}) เพื่อให้
    diarize/merge/render ที่อยู่ปลายทางใช้ร่วมกันได้โดยไม่ต้องรู้ว่า engine ไหนถอด
    """
    python_exe = _typhoon_python(base_dir)
    worker = base_dir / "tools" / "typhoon_worker.py"

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_json = Path(tmp_dir) / "segments.json"
        result = subprocess.run(
            [str(python_exe), str(worker), str(wav_path), str(out_json)],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"typhoon_worker.py ล้มเหลว (exit {result.returncode}):\n{result.stderr[-4000:]}"
            )
        segments = json.loads(out_json.read_text(encoding="utf-8"))

    logger.info("Typhoon transcribed %d chunks", len(segments))
    return segments
