SUMMARY_SYSTEM_PROMPT = """คุณเป็นผู้ช่วยสรุปการประชุม อ่าน transcript ที่ให้มาแล้วสรุปเป็นภาษาไทยในรูปแบบ Markdown ประกอบด้วย:

## ประเด็นสำคัญ
(สรุปหัวข้อและประเด็นหลักที่พูดคุยกัน เป็น bullet point)

## Action Items
(รายการสิ่งที่ต้องทำ พร้อมระบุผู้รับผิดชอบถ้าอ้างอิงได้จากบทสนทนา ถ้าไม่ระบุชัดเจนให้เขียนว่า "ไม่ระบุผู้รับผิดชอบ")

ถ้าจากบริบทการสนทนาพอเดาชื่อจริงของผู้พูดแต่ละคนได้ (เช่นมีการเอ่ยชื่อกัน) ให้ใช้ชื่อจริงแทน label "ผู้พูด N" ในสรุป ถ้าเดาไม่ได้ให้คงป้าย "ผู้พูด N" ไว้"""


def summarize_transcript(
    transcript_markdown: str,
    model: str = "claude-opus-4-8",
    api_key: str | None = None,
) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key) if api_key else Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SUMMARY_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": transcript_markdown}],
    )
    return next(block.text for block in response.content if block.type == "text")
