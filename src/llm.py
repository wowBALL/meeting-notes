"""ชั้นเดียวที่รู้ว่าโมเดลแต่ละตัวคุยด้วยโปรโตคอลอะไร

summarize.py เรียก provider.complete() แล้วได้ Completion กลับมา โดยไม่รู้ว่าปลายทาง
เป็น Anthropic หรือ endpoint ที่พูด OpenAI-compatible ค่าประจำ provider (budget,
ชื่อ env var ของ key, วิธีอ่านว่าคำตอบถูกตัด) อยู่ที่นี่ที่เดียว
"""

import os
from dataclasses import dataclass
from typing import Callable

# Claude ใช้เท่านี้พอในงานเดียวกันที่ GLM ต้องใช้สี่เท่า -- อย่ารวมเป็นค่าเดียว
CLAUDE_MAP_MAX_TOKENS = 4096
CLAUDE_REDUCE_MAX_TOKENS = 8192


class UnknownModelError(ValueError):
    """model id ที่ไม่มีใน registry -- ล้มตรงนี้ก่อนจ่ายค่าเรียก API"""


class MissingApiKeyError(RuntimeError):
    """provider ที่เลือกไม่มี key ให้ใช้

    ข้อความต้องบอกชื่อ env var ที่ต้องตั้ง เพราะตอนนี้มีมากกว่าหนึ่งตัวให้สับสนได้
    """


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
    complete: Callable[[str, str, int], Completion]


def _require_key(env_var: str, model_id: str) -> str:
    """อ่าน key ตอนจะใช้ ไม่ใช่ตอน import -- import โมดูลนี้ต้องไม่พังเมื่อยังไม่ตั้ง .env"""
    api_key = os.environ.get(env_var, "").strip()
    if not api_key:
        raise MissingApiKeyError(
            f"ไม่ได้ตั้ง {env_var} ใน .env -- ต้องตั้งก่อนจึงจะสรุปด้วย {model_id} ได้"
        )
    return api_key


def _anthropic_completer(model_id: str) -> Callable[[str, str, int], Completion]:
    def complete(system: str, content: str, max_tokens: int) -> Completion:
        from anthropic import Anthropic

        api_key = _require_key("ANTHROPIC_API_KEY", model_id)
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
            raise RuntimeError(
                f"{model_id} returned no text block (stop_reason={stop_reason!r}); "
                "nothing to use as a summary"
            )
        return Completion(text=text, truncated=stop_reason == "max_tokens")

    return complete


def _claude(model_id: str) -> Provider:
    return Provider(
        model_id=model_id,
        map_max_tokens=CLAUDE_MAP_MAX_TOKENS,
        reduce_max_tokens=CLAUDE_REDUCE_MAX_TOKENS,
        complete=_anthropic_completer(model_id),
    )


PROVIDERS: dict[str, Provider] = {
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
