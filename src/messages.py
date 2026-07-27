"""ข้อความที่ผู้ใช้เห็นทั้งหมด แยกออกจาก logic เพื่อให้สลับภาษาได้โดยไม่แตะโค้ด

logic ส่งแค่รหัสเหตุการณ์กับค่าประกอบ ที่นี่เป็นที่เดียวที่รู้จักคำพูด ทั้ง CLI
และหน้าเว็บใช้ catalog ชุดเดียวกัน จึงเพิ่มภาษาได้ที่ไฟล์เดียว
"""

LANGUAGES = ("th", "en")
DEFAULT_LANGUAGE = "th"

MESSAGES = {
    "th": {
        # --- ตัวอัด ---
        "room_opened": "เปิดห้อง {room} · {model}",
        "devices_selected": "อัดจากไมค์ {mic} · เสียงคู่สนทนาจาก {loopback}",
        "recording_started": "กำลังอัดเสียง...",
        "press_ctrl_c": "กด Ctrl+C เพื่อหยุด",
        "part_closed": "ปิดไฟล์ส่วนที่ {count}",
        "device_changed": (
            'อุปกรณ์เสียงเปลี่ยนจาก "{old}" เป็น "{new}" — '
            "เครื่องอัดยังดักฟังตัวเดิมอยู่ เสียงคู่สนทนาจะไม่ถูกอัด"
        ),
        "room_closed": "ปิดห้องแล้ว กำลังบีบอัดไฟล์เสียง",
        "encode_started": "กำลังบีบอัดไฟล์เสียง",
        "encode_done": "บันทึกไปที่ {path}",
        "encode_failed": "บีบอัดไม่สำเร็จ ชิ้นส่วนเสียงยังอยู่ที่ {path} : {error}",
        "no_audio": (
            "ไม่มีเสียงถูกบันทึกในรอบนี้ (หยุดเร็วเกินไปหรือไมค์ไม่ส่งเสียงมา) "
            "ไม่ได้สร้างไฟล์"
        ),
        "device_open_failed": (
            "ไม่พบไมค์/ลำโพง default กรุณาตรวจสอบการตั้งค่าเสียงของ Windows: {error}"
        ),
        "stream_open_failed": (
            "อัดเสียงไม่สำเร็จ (อาจไม่มีสิทธิ์เข้าถึงไมค์ - ตรวจสอบที่ "
            "Settings > Privacy > Microphone): {error}"
        ),
        "samplerate_mismatch": "{error}",
        "orphan_found": "พบการอัดค้างจากรอบก่อน กำลังกู้ {name} ...",
        "orphan_recovered": "กู้สำเร็จ: {path}",
        "orphan_failed": "กู้ {name} ไม่สำเร็จ ข้ามไปก่อน (ชิ้นส่วนยังอยู่): {error}",
        "orphan_swept": "ลบเศษ session ที่ปิดงานไม่หมดจากรอบก่อน: {name}",
        # --- pipeline (watcher) ---
        "queued": "เข้าคิวรอประมวลผล",
        "transcribe_started": "กำลังถอดเสียง",
        "diarize_started": "กำลังแยกผู้พูด",
        "diarize_failed": "แยกผู้พูดไม่สำเร็จ ไปต่อโดยไม่มีชื่อผู้พูด: {error}",
        "summarize_started": "กำลังสรุปด้วย {model}",
        "meeting_done": "เสร็จแล้ว: {path}",
        "job_failed": "ประมวลผลล้มเหลว: {error}",
    },
    "en": {
        # --- recorder ---
        "room_opened": "Room opened: {room} · {model}",
        "devices_selected": "Recording mic {mic} · far end from {loopback}",
        "recording_started": "Recording...",
        "press_ctrl_c": "Press Ctrl+C to stop",
        "part_closed": "Closed part {count}",
        "device_changed": (
            'Audio device changed from "{old}" to "{new}" — the recorder is '
            "still listening to the old one, so the far end will not be captured."
        ),
        "room_closed": "Room ended, encoding the audio",
        "encode_started": "Encoding the audio",
        "encode_done": "Saved to {path}",
        "encode_failed": (
            "Encoding failed, the audio parts are still at {path} : {error}"
        ),
        "no_audio": (
            "Nothing was recorded this time (stopped too early, or the mic sent "
            "no audio). No file was created."
        ),
        "device_open_failed": (
            "No default mic/speaker found. Check the Windows sound settings: {error}"
        ),
        "stream_open_failed": (
            "Could not start recording (microphone permission? check "
            "Settings > Privacy > Microphone): {error}"
        ),
        "samplerate_mismatch": "{error}",
        "orphan_found": "Found an unfinished recording, recovering {name} ...",
        "orphan_recovered": "Recovered: {path}",
        "orphan_failed": "Could not recover {name}, skipping (parts kept): {error}",
        "orphan_swept": (
            "Removed leftover session debris from an earlier run: {name}"
        ),
        # --- pipeline (watcher) ---
        "queued": "Queued for processing",
        "transcribe_started": "Transcribing",
        "diarize_started": "Separating speakers",
        "diarize_failed": (
            "Speaker separation failed, continuing without labels: {error}"
        ),
        "summarize_started": "Summarizing with {model}",
        "meeting_done": "Done: {path}",
        "job_failed": "Processing failed: {error}",
    },
}


def render(code: str, params: dict | None = None, lang: str = DEFAULT_LANGUAGE) -> str:
    """ข้อความสำหรับรหัสนี้ ไม่ raise ไม่ว่าจะเจออะไร

    ข้อความเป็นของประดับเมื่อเทียบกับการอัดประชุมที่อัดซ้ำไม่ได้ รหัสที่ไม่รู้จัก
    หรือค่าที่เติมไม่ครบจึงต้องได้อะไรบางอย่างกลับไปเสมอ ไม่ใช่ exception
    """
    catalog = MESSAGES.get(lang) or MESSAGES[DEFAULT_LANGUAGE]
    template = catalog.get(code)
    if template is None:
        return code
    try:
        return template.format(**(params or {}))
    except (KeyError, IndexError, ValueError):
        return template
