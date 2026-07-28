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
        "speakers_matched": "จำเสียงได้ {count} คนจากทะเบียน",
        "speakers_pending": "มีผู้พูด {count} คนรอตั้งชื่อ",
        "speakers_failed": "ระบบจำเสียงทำงานไม่สำเร็จ ไฟล์ที่ได้ไม่ได้รับผลกระทบ: {error}",
        "summarize_started": "กำลังสรุปด้วย {model}",
        "meeting_done": "เสร็จแล้ว: {path}",
        "job_failed": "ประมวลผลล้มเหลว: {error}",
        # --- preflight ---
        # ข้อความชุดนี้ถูกยึดไว้ด้วย tests/test_preflight.py แบบตรงตัวอักษร
        # แก้คำเมื่อไหร่เทสต์จะบอกทันที ซึ่งเป็นสิ่งที่ต้องการ
        "check_mic": "ไมค์",
        "check_loopback": "ลำโพง (คู่สนทนา)",
        "check_samplerate": "sample rate",
        "check_model": "โมเดลที่ใช้สรุป",
        "peak_silent": "เงียบสนิท",
        "peak_level": "peak {db} dB",
        "passthrough": "{error}",
        "mic_ok": "{level} -- ระดับเสียงพูดปกติ",
        "mic_weak": (
            "{level} -- เบากว่าปกติมาก ลองขยับไมค์เข้าใกล้ / ดัน volume "
            "หรือเปิด Microphone Boost (mmsys.cpl)"
        ),
        "mic_none": (
            "{level} -- แทบไม่มีสัญญาณ เช็คว่าเสียบช่องไมค์ (สีชมพู) แน่นดี "
            "และไม่ได้ mute อยู่"
        ),
        "loopback_ok": "{level} จาก {device} -- เสียงไหลผ่านจริง",
        "loopback_silent": (
            "ไม่มีเสียงผ่าน {device} -- ถ้าตอนนี้เปิดเสียงอยู่ แปลว่าแอปส่งเสียง "
            "ออกอุปกรณ์อื่น ให้ตั้ง output ของแอปประชุมให้ตรงกับ default ของ Windows"
        ),
        "samplerate_ok": "ตรงกันที่ {rate} Hz",
        "model_ok": "รองรับ",
        "model_transcript_only": (
            "ตั้งใจปิดการสรุปไว้ (CLAUDE_MODEL={model}) -- อัดและถอดเสียงได้ปกติ "
            "จะไม่มี summary.md เกิดขึ้น เพราะเป็นโหมดที่เลือกไว้เอง ไม่ใช่การตั้งค่าที่ผิด"
        ),
        "model_unresolvable": (
            "ไม่รู้จักโมเดล {model} -- ที่รองรับตอนนี้: {known} -- อัดและถอดเสียงได้ปกติ "
            "แต่จะสรุปไม่สำเร็จ แก้ CLAUDE_MODEL ใน .env ให้เป็นหนึ่งในรายชื่อนี้"
        ),
        "mark_ok": "[ ผ่าน ]",
        "mark_warn": "[ เตือน ]",
        "mark_fail": "[ ไม่ผ่าน ]",
        "report_fail": "สรุป: ยังไม่ควรเริ่มอัด -- แก้ข้อที่ไม่ผ่านก่อน",
        "report_warn": "สรุป: พร้อมอัด แต่มีข้อเตือน อ่านให้ครบก่อนเริ่ม",
        "report_ok": "สรุป: พร้อมอัดประชุมได้เลย",
        "preflight_checking_model": "กำลังตรวจว่าโมเดลที่จะใช้สรุปใช้งานได้ไหม ({model}) ...",
        "preflight_checking_audio": (
            "กำลังตรวจเสียง {seconds} วินาที -- พูดใส่ไมค์ และเปิดเสียงอะไรก็ได้ออกลำโพง"
        ),
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
        "speakers_matched": "Recognized {count} speaker(s) from the registry",
        "speakers_pending": "{count} speaker(s) waiting to be named",
        "speakers_failed": "Speaker identification failed; the saved files are unaffected: {error}",
        "summarize_started": "Summarizing with {model}",
        "meeting_done": "Done: {path}",
        "job_failed": "Processing failed: {error}",
        # --- preflight ---
        "check_mic": "Microphone",
        "check_loopback": "Speaker (far end)",
        "check_samplerate": "sample rate",
        "check_model": "Summary model",
        "peak_silent": "completely silent",
        "peak_level": "peak {db} dB",
        "passthrough": "{error}",
        "mic_ok": "{level} -- normal speaking level",
        "mic_weak": (
            "{level} -- much quieter than normal. Move closer to the mic, raise the "
            "volume, or turn on Microphone Boost (mmsys.cpl)"
        ),
        "mic_none": (
            "{level} -- almost no signal. Check that the mic is plugged in firmly "
            "(pink jack) and not muted"
        ),
        "loopback_ok": "{level} from {device} -- audio is flowing",
        "loopback_silent": (
            "No audio through {device} -- if something is playing right now, the app "
            "is sending audio to a different device. Set the meeting app's output to "
            "match the Windows default"
        ),
        "samplerate_ok": "both at {rate} Hz",
        "model_ok": "supported",
        "model_transcript_only": (
            "Summarizing is switched off on purpose (CLAUDE_MODEL={model}) -- "
            "recording and transcription still work as normal. No summary.md will be "
            "produced, because this is a deliberate choice, not a broken setting"
        ),
        "model_unresolvable": (
            "unknown model {model} -- supported right now: {known} -- recording and "
            "transcription still work, but summarizing will fail. Fix CLAUDE_MODEL in "
            ".env to one of these"
        ),
        "mark_ok": "[ PASS ]",
        "mark_warn": "[ WARN ]",
        "mark_fail": "[ FAIL ]",
        "report_fail": "Verdict: do not start recording yet -- fix what failed first",
        "report_warn": "Verdict: ready to record, but read the warnings first",
        "report_ok": "Verdict: ready to record",
        "preflight_checking_model": "Checking whether the summary model is usable ({model}) ...",
        "preflight_checking_audio": (
            "Checking audio for {seconds} seconds -- talk into the mic, and play "
            "anything through the speakers"
        ),
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
