"""ชั้นเดียวที่รู้ว่าโมเดลแต่ละตัวคุยด้วยโปรโตคอลอะไร

summarize.py เรียก provider.complete() แล้วได้ Completion กลับมา โดยไม่รู้ว่าปลายทาง
เป็น Anthropic หรือ endpoint ที่พูด OpenAI-compatible ค่าประจำ provider (budget,
ชื่อ env var ของ key, วิธีอ่านว่าคำตอบถูกตัด) อยู่ที่นี่ที่เดียว
"""

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

# Claude ใช้เท่านี้พอในงานเดียวกันที่ GLM ต้องใช้สี่เท่า -- อย่ารวมเป็นค่าเดียว
CLAUDE_MAP_MAX_TOKENS = 4096
CLAUDE_REDUCE_MAX_TOKENS = 8192

DEFAULT_LLM_BASE_URL = "https://your-llm-endpoint.example/v1"

# GLM-5.2 เป็น reasoning model: max_tokens คุมผลรวมของ reasoning + คำตอบ ไม่ใช่
# คำตอบเพียงอย่างเดียว วัดจริงบน transcript ไทยแล้ว reasoning กินได้ถึง 12,909
# ตัวอักษรบน chunk ขนาด 14,065 token แล้วยังต้องเขียนคำตอบอีก 2,752 -- ที่ 8192
# มันถูกตัด ค่าพวกนี้คือราวสองเท่าของกรณีแย่สุดที่วัดได้ ไม่ใช่เลขที่คิดเอาเอง
GLM_MAP_MAX_TOKENS = 16384
GLM_REDUCE_MAX_TOKENS = 24576

# หนึ่ง call ของ GLM ใช้เวลาได้ถึง ~155 วินาทีที่ราว 53 token/วินาที เผื่อไว้มาก
# เพราะการหมดเวลากลาง reduce แปลว่าเสียสรุปรายช่วงที่จ่ายไปแล้วทั้งหมด
LLM_TIMEOUT_SECONDS = 900

# ตัด detail ของ error/body ทุกจุดที่ 400 ตัวอักษรเท่ากัน -- ค่าเดียว ชื่อเดียว กันพลาด
# แบบตอนที่มันแยกกัน 3 ที่ในไฟล์นี้แล้วแก้ไม่ครบ
_RESPONSE_DETAIL_CHARS = 400


class UnknownModelError(ValueError):
    """model id ที่ไม่มีใน registry -- ล้มตรงนี้ก่อนจ่ายค่าเรียก API"""


class MissingApiKeyError(RuntimeError):
    """provider ที่เลือกไม่มี key ให้ใช้

    ข้อความต้องบอกชื่อ env var ที่ต้องตั้ง เพราะตอนนี้มีมากกว่าหนึ่งตัวให้สับสนได้
    """


class UnusableAnswerError(RuntimeError):
    """โมเดลตอบกลับมาแล้ว (HTTP 200) แต่คำตอบใช้เป็นสรุปไม่ได้ ไม่ว่าจะด้วยเหตุผลไหนใน
    สองกลุ่มนี้: (1) ไม่มี text block หรือ content ว่างเพราะ reasoning กิน budget หมด
    หรือ (2) รูปร่าง JSON ผิดคาดที่เป็นความผิดปกติถาวรของ proxy เอง เช่น top-level ไม่ใช่
    JSON object, choices ว่างเปล่า/ไม่มี, หรือ choice ไม่มี message -- ทั้งสองกลุ่มมีจุด
    ร่วมเดียวกัน: ยิงคำขอเดิมซ้ำก็ได้ผลเดิมเป๊ะ ไม่มีทางหายเอง

    แยกเป็นคลาสของตัวเองเพราะ is_retryable ต้องแยกมันออกจาก RuntimeError ทั่วไป: คำขอ
    เดิมยิงซ้ำก็ได้คำตอบเดิม ส่วน RuntimeError ที่มาจากที่อื่นอาจเป็นอาการชั่วคราวที่หายเอง

    หมายเหตุ: การเพิ่ม budget เป็นสองเท่าที่ _summarize ทำ (src/summarize.py) ทำงาน
    เฉพาะตอน complete() คืน Completion(truncated=True) มาเท่านั้น -- error นี้ raise
    ก่อนจะมี Completion ให้เห็นด้วยซ้ำ จึงไม่มีทางไปถึง path เพิ่ม budget นั้นเลย ความ
    ล้มเหลวแบบนี้จึงเป็นแบบถาวรจริงๆ ไม่มี retry ชั้นไหนกู้คืนให้ได้
    """


class UpstreamBodyError(RuntimeError):
    """proxy ตอบ HTTP 200 แต่ body ไม่ใช่ JSON เลย (เช่น gateway ส่งหน้า HTML error มา,
    body ว่างเปล่า, หรือ JSON ที่ถูกตัดกลางคัน) -- ต่างจาก UnusableAnswerError ตรงที่
    นี่มักเป็นอาการชั่วคราวของ gateway/proxy กลางทาง ไม่ใช่ความผิดปกติถาวรของคำขอ ลองใหม่
    รอบถัดไปมีโอกาสได้ JSON ที่ถูกต้องกลับมา

    ไม่มี status_code ติดมา (ไม่ใช่ urllib.error.HTTPError) และไม่ใช่ UnusableAnswerError
    หรือ MissingApiKeyError จึงถูก is_retryable ตัดสินว่า retryable โดยอัตโนมัติ -- ตรง
    ตามเจตนา: ค่า backoff ที่เสียไปถูกกว่าการทิ้งสรุปทั้ง chunk ไปเฉยๆ
    """


class HttpStatusError(RuntimeError):
    """พก status_code มาให้ is_retryable อ่านได้เหมือน exception ของ Anthropic SDK

    ไม่มี field นี้แล้ว is_retryable จะเดาว่า "ไปไม่ถึง API" แล้วลองใหม่ให้ทุกกรณี
    รวมถึง 401 ที่ลองอีกกี่ครั้งก็ได้คำตอบเดิม
    """

    def __init__(self, status_code: int, message: str):
        super().__init__(f"HTTP {status_code}: {message}")
        self.status_code = status_code


@dataclass(frozen=True)
class Completion:
    """คำตอบหนึ่งครั้งจากโมเดล

    truncated คือ "โมเดลใช้โควตาคำตอบหมดก่อนพูดจบ" ซึ่งแต่ละ provider บอกมาด้วย
    ชื่อ field คนละชื่อ -- แปลให้เป็นคำเดียวกันที่นี่ ผู้เรียกจึงไม่ต้องรู้
    """

    text: str
    truncated: bool


@dataclass(frozen=True)
class Provider:
    model_id: str
    map_max_tokens: int
    reduce_max_tokens: int
    # ชื่อ env var ที่ provider นี้อ่าน key มา -- completer แต่ละตัวใช้ตอนจะดึง key จริง
    # (ดู _require_key) ไม่ได้ผูกกับ preflight โดยเฉพาะ
    key_env: str
    complete: Callable[[str, str, int], Completion]


def _require_key(env_var: str, model_id: str) -> str:
    """อ่าน key ตอนจะใช้ ไม่ใช่ตอน import -- import โมดูลนี้ต้องไม่พังเมื่อยังไม่ตั้ง .env"""
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise MissingApiKeyError(
            f"ไม่ได้ตั้ง {env_var} ใน .env -- ต้องตั้งก่อนจึงจะสรุปด้วย {model_id} ได้"
        )
    return api_key


def _anthropic_completer(
    model_id: str, key_env: str
) -> Callable[[str, str, int], Completion]:
    def complete(system: str, content: str, max_tokens: int) -> Completion:
        from anthropic import Anthropic

        api_key = _require_key(key_env, model_id)
        # ปล่อยให้ SDK ใช้ timeout/retry default ของมันเอง -- retry เป็นหน้าที่ของ
        # is_retryable ใน summarize.py อยู่แล้ว ไม่ต้องมีชั้นซ้อนที่นี่อีก
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        stop_reason = getattr(response, "stop_reason", None)
        text = next(
            (block.text for block in response.content if block.type == "text"), None
        )
        if text is None:
            raise UnusableAnswerError(
                f"{model_id} returned no text block (stop_reason={stop_reason!r}); "
                "nothing to use as a summary"
            )
        return Completion(text=text, truncated=stop_reason == "max_tokens")

    return complete


def _claude(model_id: str) -> Provider:
    key_env = "ANTHROPIC_API_KEY"
    return Provider(
        model_id=model_id,
        map_max_tokens=CLAUDE_MAP_MAX_TOKENS,
        reduce_max_tokens=CLAUDE_REDUCE_MAX_TOKENS,
        key_env=key_env,
        complete=_anthropic_completer(model_id, key_env),
    )


def _response_detail(payload: object) -> str:
    """ข้อความอธิบายจากตัว proxy เอง ตัดที่ _RESPONSE_DETAIL_CHARS ตัวอักษรแบบเดียวกับ
    branch HTTPError

    LiteLLM proxy ตอบ HTTP 200 พร้อม error envelope {"error": {"message": ...}}
    แทนคำตอบได้ (เช่น budget ของ key หมด) -- ถ้ามีให้ใช้ข้อความนั้นตรงๆ เพราะเป็นสิ่งที่
    คนอ่าน log ต้องการเห็นจริง ไม่ใช่แค่ KeyError ที่บอกอะไรไม่ได้ ถ้าไม่มี ให้ dump
    payload ทั้งก้อนแทนเพื่อไม่ให้ข้อมูลหายไปเฉยๆ -- ใช้ได้แม้ payload จะไม่ใช่ dict เลย
    (null/list/string/number) เพราะ json.dumps รับได้หมด
    """
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:_RESPONSE_DETAIL_CHARS]
    return json.dumps(payload, ensure_ascii=False)[:_RESPONSE_DETAIL_CHARS]


def _openai_compat_completer(
    model_id: str, key_env: str, base_url_env: str, default_base_url: str
) -> Callable[[str, str, int], Completion]:
    def complete(system: str, content: str, max_tokens: int) -> Completion:
        api_key = _require_key(key_env, model_id)
        base_url = os.environ.get(base_url_env, "").strip() or default_base_url
        # ensure_ascii=False คือหัวใจ: transcript เป็นภาษาไทยทั้งไฟล์ ถ้า escape เป็น
        # \uXXXX ขนาด payload บวมและ proxy บางตัวส่งต่อเป็น ???? -- ทดสอบไว้แล้วว่า
        # แบบนี้ภาษาไทยกลับมาตรงทุกตัวอักษร
        body = json.dumps(
            {
                "model": model_id,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            },
            ensure_ascii=False,
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json; charset=utf-8",
            },
        )
        try:
            # งานสรุปทั้ง chunk ใช้เวลาได้นาน จึงเป็น LLM_TIMEOUT_SECONDS เต็ม ๆ
            # ไม่ต้องปิด retry ที่นี่: urlopen ไม่ retry เองอยู่แล้ว การลองใหม่เป็นหน้าที่
            # ของ is_retryable ใน summarize.py ที่ชั้นเหนือขึ้นไป
            with urllib.request.urlopen(
                request, timeout=LLM_TIMEOUT_SECONDS
            ) as response:
                try:
                    raw_body = response.read().decode("utf-8")
                except UnicodeDecodeError as e:
                    # gateway corruption กลางทาง: ไบต์ที่ไม่ใช่ UTF-8 ปกติเป็นอาการ
                    # ชั่วคราวเหมือนกับ HTML error page หรือ truncated stream ยิงซ้ำ
                    # รอบถัดไปมีโอกาสได้ข้อมูลที่ถูกต้องกลับมา
                    raise UpstreamBodyError(
                        f"{model_id} returned HTTP 200 with a body that is not valid UTF-8: {e}"
                    ) from e
        except urllib.error.HTTPError as e:
            # ตัดที่ _RESPONSE_DETAIL_CHARS ตัวอักษร: body ของ error บาง proxy ยัด HTML
            # มาทั้งหน้า
            detail = e.read().decode("utf-8", "replace")[:_RESPONSE_DETAIL_CHARS]
            raise HttpStatusError(e.code, detail) from e

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError as e:
            # HTTP 200 แต่ body ไม่ใช่ JSON เลย -- gateway ส่งหน้า HTML error มา, body
            # ว่างเปล่า, หรือ JSON ถูกตัดกลางคัน เดิมโผล่เป็น JSONDecodeError ดิบๆ
            # ("Expecting value: line 1 column 1") ที่ไม่บอกว่า proxy พูดอะไรจริงๆ --
            # ต้องจับเจาะจง (ไม่ใช่ Exception เฉยๆ) เพราะ JSONDecodeError เป็น
            # ValueError ซึ่งอาจบังเอิญไปคาบเกี่ยว exception อื่นที่ไม่เกี่ยวกัน
            # นี่มักเป็นอาการชั่วคราวของ gateway กลางทาง จึงยังต้อง retryable เหมือนเดิม
            # (ดู UpstreamBodyError) ไม่ใช่ UnusableAnswerError แบบ top-level ผิดรูป
            raise UpstreamBodyError(
                f"{model_id} returned HTTP 200 but the body is not valid JSON: "
                f"{raw_body[:_RESPONSE_DETAIL_CHARS]}"
            ) from e

        # ตรวจ top level ก่อน .get() เข้าไป: JSON ที่ถูกต้องแต่ top-level ไม่ใช่ object
        # เลยก็เป็นไปได้ (null/list/string/number) -- proxy พังแบบนี้ยิงซ้ำก็ได้รูปร่าง
        # เดิม จึงต้องเป็น UnusableAnswerError ไม่ใช่ retryable
        if not isinstance(payload, dict):
            raise UnusableAnswerError(
                f"{model_id} returned HTTP 200 but the top-level body is not a "
                f"JSON object: {_response_detail(payload)}"
            )

        # ตรวจรูปร่างก่อน index เข้าไป: proxy ตอบ 200 พร้อม error envelope แทนคำตอบได้
        # ({"error": {...}}), choices ว่างเปล่าก็ได้ ({"choices": []}), หรือ choice
        # ไม่มี message เลยก็ได้ -- ทั้งหมดนี้เดิมโผล่เป็น KeyError/IndexError ดิบๆ ที่
        # ทิ้งข้อความจริงของ proxy (เช่น "budget exceeded for key") ไปเฉยๆ แล้วยัง
        # ถูก is_retryable เดาว่า retryable เพราะไม่มี status_code (ดูไม่ต่างจาก
        # "ไปไม่ถึง API") ทั้งที่ยิงซ้ำก็ได้ผลเดิมทุกครั้ง
        choices = payload.get("choices")
        choice = choices[0] if isinstance(choices, list) and choices else None
        message = choice.get("message") if isinstance(choice, dict) else None
        if not isinstance(message, dict):
            raise UnusableAnswerError(
                f"{model_id} returned HTTP 200 but the body is not a usable "
                f"completion: {_response_detail(payload)}"
            )
        answer_content = message.get("content")
        if answer_content is not None and not isinstance(answer_content, str):
            # OpenAI content parts ([{"type": "text", "text": "..."}]) เป็น list ที่
            # truthy อยู่แล้ว ผ่าน `or ""` ไปได้ แล้วไปแตกที่ .strip() ด้านล่างแทน --
            # กันไว้ตรงนี้ก่อนเลย
            raise UnusableAnswerError(
                f"{model_id} returned non-string content "
                f"({type(answer_content).__name__}); cannot use as a summary: "
                f"{_response_detail(payload)}"
            )
        text = answer_content or ""
        finish_reason = choice.get("finish_reason")
        if not text.strip():
            # reasoning model ใช้ budget หมดไปกับ reasoning_content ได้ โดย content
            # เป็นสตริงว่าง เรียกซ้ำด้วย budget เดิมย่อมได้ผลเดิม จึงต้องไม่ retryable
            # -- UnusableAnswerError คือสิ่งที่ is_retryable แยกออกจาก RuntimeError
            # ทั่วไปโดยเจตนา (ดู src/summarize.py::is_retryable)
            raise UnusableAnswerError(
                f"{model_id} returned no text (finish_reason={finish_reason!r}); "
                "the entire output budget went to reasoning"
            )
        return Completion(text=text, truncated=finish_reason == "length")

    return complete


PROVIDERS: dict[str, Provider] = {
    "GLM-5.2": Provider(
        model_id="GLM-5.2",
        map_max_tokens=GLM_MAP_MAX_TOKENS,
        reduce_max_tokens=GLM_REDUCE_MAX_TOKENS,
        key_env="LLM_API_KEY",
        complete=_openai_compat_completer(
            "GLM-5.2", "LLM_API_KEY", "LLM_BASE_URL", DEFAULT_LLM_BASE_URL
        ),
    ),
    "claude-opus-5": _claude("claude-opus-5"),
    "claude-sonnet-5": _claude("claude-sonnet-5"),
}


def resolve(model_id: str) -> Provider:
    provider = PROVIDERS.get(model_id)
    if provider is None:
        known = ", ".join(sorted(PROVIDERS))
        raise UnknownModelError(
            f"ไม่รู้จักโมเดล {model_id!r} -- ที่รองรับตอนนี้: {known}"
        )
    return provider
