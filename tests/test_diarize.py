import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from src.config import DEFAULT_DIARIZATION_MODEL
from src.diarize import (
    DiarizationResult,
    diarize_audio,
    load_diarization_pipeline,
)


def _write_wav(path, seconds=0.5, sample_rate=16000):
    """ไฟล์เสียงจริง ไม่ใช่ b"fake audio data"

    diarize_audio อ่านไบต์เองแล้ว (ดู src/waveform.py) เทสต์ที่ป้อนขยะจึงตายที่ตัวอ่าน
    ไฟล์ก่อนจะไปถึงพฤติกรรมที่มันตั้งใจวัด -- wav ครึ่งวินาทีเขียนเร็วพอที่จะไม่ต้องแลก
    ความเร็วของชุดเทสต์กับความจริงของอินพุต
    """
    frames = int(seconds * sample_rate)
    sf.write(path, np.sin(np.linspace(0.0, 40.0, frames, dtype="float32")), sample_rate)
    return path


def _fake_pyannote(mock_pipeline_cls):
    pyannote_pkg = ModuleType("pyannote")
    audio_mod = ModuleType("pyannote.audio")
    audio_mod.Pipeline = mock_pipeline_cls
    pyannote_pkg.audio = audio_mod
    return {"pyannote": pyannote_pkg, "pyannote.audio": audio_mod}


class FakeTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


def test_diarize_audio_hands_pyannote_an_in_memory_waveform_not_a_path(tmp_path):
    # pyannote 4.x อ่านไฟล์เองผ่าน torchcodec เท่านั้น และ torchcodec โหลดไม่ขึ้นบน
    # เครื่องนี้ -- การส่ง path เข้าไปตาย RuntimeError ทุกครั้ง แล้ว pipeline.py กลืนไว้
    # เดินต่อด้วย speaker_turns = [] ประชุมทั้งครั้งได้ป้าย "ผู้พูด 1" ก้อนเดียวเงียบ ๆ
    # เทสต์นี้คือสิ่งที่กันไม่ให้กลับไปส่ง path อีกโดยไม่ได้ตั้งใจ
    audio_path = _write_wav(tmp_path / "sample.wav", seconds=0.5)

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = []
    mock_pipeline = MagicMock(
        return_value=MagicMock(speaker_diarization=fake_diarization)
    )

    diarize_audio(audio_path, hf_token="t", pipeline=mock_pipeline)

    (passed,), kwargs = mock_pipeline.call_args
    assert kwargs == {}
    assert isinstance(passed, dict), f"pyannote ยังได้รับ {type(passed).__name__}"
    assert passed["sample_rate"] == 16000
    assert tuple(passed["waveform"].shape) == (1, 8000)


def test_diarize_audio_extracts_speaker_turns(tmp_path):
    audio_path = _write_wav(tmp_path / "sample.wav")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [
        (FakeTurn(0.0, 3.0), None, "SPEAKER_00"),
        (FakeTurn(3.0, 6.0), None, "SPEAKER_01"),
    ]
    mock_pipeline = MagicMock(
        return_value=MagicMock(speaker_diarization=fake_diarization)
    )

    result = diarize_audio(audio_path, hf_token="test-token", pipeline=mock_pipeline)

    assert result == DiarizationResult(
        turns=[
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
        ]
    )
    mock_pipeline.assert_called_once()
    fake_diarization.itertracks.assert_called_once_with(yield_label=True)


def test_diarization_result_no_longer_carries_embeddings():
    # voiceprint ย้ายไป src/voiceprint.py แล้ว การมี field นี้ค้างอยู่จะทำให้มีคนเขียนโค้ด
    # ที่อ่าน centroid ของ pipeline มาเทียบข้ามพื้นที่อีกครั้งโดยไม่รู้ตัว
    assert not hasattr(DiarizationResult(turns=[]), "embeddings")


def test_load_diarization_pipeline_moves_to_the_device_gpu_py_hands_it():
    loaded = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = loaded
    device = object()

    with patch.dict(sys.modules, _fake_pyannote(mock_pipeline_cls)), patch(
        "src.diarize.cuda_device", return_value=device
    ):
        result = load_diarization_pipeline("hf-test-token")

    mock_pipeline_cls.from_pretrained.assert_called_once_with(
        DEFAULT_DIARIZATION_MODEL, token="hf-test-token"
    )
    loaded.to.assert_called_once_with(device)
    assert result is loaded


def test_load_diarization_pipeline_honours_a_custom_checkpoint():
    """DIARIZATION_MODEL ใน .env ต้องไปถึง from_pretrained จริง ไม่ใช่ถูก default ทับ"""
    loaded = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = loaded

    with patch.dict(sys.modules, _fake_pyannote(mock_pipeline_cls)), patch(
        "src.diarize.cuda_device", return_value=None
    ):
        load_diarization_pipeline("hf-test-token", "pyannote/speaker-diarization-3.1")

    mock_pipeline_cls.from_pretrained.assert_called_once_with(
        "pyannote/speaker-diarization-3.1", token="hf-test-token"
    )


def test_load_diarization_pipeline_stays_on_cpu_when_there_is_no_device():
    loaded = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = loaded

    with patch.dict(sys.modules, _fake_pyannote(mock_pipeline_cls)), patch(
        "src.diarize.cuda_device", return_value=None
    ):
        result = load_diarization_pipeline("hf-test-token")

    loaded.to.assert_not_called()
    assert result is loaded


def test_diarize_audio_loads_pipeline_via_helper_when_none_given(tmp_path):
    audio_path = _write_wav(tmp_path / "sample.wav")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = []
    loaded = MagicMock(return_value=MagicMock(speaker_diarization=fake_diarization))

    with patch(
        "src.diarize.load_diarization_pipeline", return_value=loaded
    ) as mock_load:
        diarize_audio(audio_path, hf_token="hf-test-token", pipeline=None)

    mock_load.assert_called_once_with("hf-test-token", DEFAULT_DIARIZATION_MODEL)
    loaded.assert_called_once()
