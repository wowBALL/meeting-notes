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
    fake_diarization.labels.return_value = []
    mock_pipeline = MagicMock(
        return_value=MagicMock(speaker_diarization=fake_diarization)
    )

    diarize_audio(audio_path, hf_token="t", pipeline=mock_pipeline)

    (passed,), kwargs = mock_pipeline.call_args
    assert kwargs == {}
    assert isinstance(passed, dict), f"pyannote ยังได้รับ {type(passed).__name__}"
    assert passed["sample_rate"] == 16000
    assert tuple(passed["waveform"].shape) == (1, 8000)


def test_diarize_audio_extracts_speaker_turns_and_embeddings(tmp_path):
    audio_path = _write_wav(tmp_path / "sample.wav")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [
        (FakeTurn(0.0, 3.0), None, "SPEAKER_00"),
        (FakeTurn(3.0, 6.0), None, "SPEAKER_01"),
    ]
    fake_diarization.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]
    fake_output = MagicMock(
        speaker_diarization=fake_diarization,
        speaker_embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    mock_pipeline = MagicMock(return_value=fake_output)

    result = diarize_audio(audio_path, hf_token="test-token", pipeline=mock_pipeline)

    assert result == DiarizationResult(
        turns=[
            {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
            {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
        ],
        embeddings={"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]},
    )
    mock_pipeline.assert_called_once()
    fake_diarization.itertracks.assert_called_once_with(yield_label=True)


def test_diarize_audio_keys_embeddings_by_label_not_by_position(tmp_path):
    # pyannote บอกไว้เองว่า array เรียงตาม diarization.labels() ไม่ใช่ตามลำดับที่
    # ผู้พูดโผล่ใน itertracks -- ผูกผิดหนึ่งตำแหน่งแปลว่าลงทะเบียนเสียงผิดคน
    audio_path = _write_wav(tmp_path / "sample.wav")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [
        (FakeTurn(0.0, 3.0), None, "SPEAKER_01"),
        (FakeTurn(3.0, 6.0), None, "SPEAKER_00"),
    ]
    fake_diarization.labels.return_value = ["SPEAKER_00", "SPEAKER_01"]
    fake_output = MagicMock(
        speaker_diarization=fake_diarization,
        speaker_embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )

    result = diarize_audio(audio_path, hf_token="t", pipeline=MagicMock(return_value=fake_output))

    assert result.embeddings == {"SPEAKER_00": [1.0, 0.0], "SPEAKER_01": [0.0, 1.0]}


def test_diarize_audio_returns_no_embeddings_when_pyannote_gives_none(tmp_path):
    # เกิดขึ้นจริงกับ OracleClustering (ไลบรารี pyannote/audio/pipelines/
    # speaker_diarization.py ~บรรทัด 745)
    audio_path = _write_wav(tmp_path / "sample.wav")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [(FakeTurn(0.0, 3.0), None, "SPEAKER_00")]
    fake_diarization.labels.return_value = ["SPEAKER_00"]
    fake_output = MagicMock(speaker_diarization=fake_diarization, speaker_embeddings=None)

    result = diarize_audio(audio_path, hf_token="t", pipeline=MagicMock(return_value=fake_output))

    assert result.turns == [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    assert result.embeddings == {}


def test_diarize_audio_survives_an_unreadable_embedding_array(tmp_path):
    audio_path = _write_wav(tmp_path / "sample.wav")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [(FakeTurn(0.0, 3.0), None, "SPEAKER_00")]
    fake_diarization.labels.return_value = ["SPEAKER_00"]
    fake_output = MagicMock(
        speaker_diarization=fake_diarization, speaker_embeddings="ไม่ใช่ array"
    )

    result = diarize_audio(audio_path, hf_token="t", pipeline=MagicMock(return_value=fake_output))

    # การจำเสียงหายไป แต่การแยกผู้พูดยังทำงาน -- นี่คือสิ่งที่ยอมเสียได้
    assert result.turns == [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    assert result.embeddings == {}


def test_diarize_audio_keeps_the_turns_when_reading_embeddings_raises_anything(tmp_path):
    # _speaker_embeddings runs AFTER turns is already built successfully; any
    # exception type escaping it -- not just TypeError/ValueError/IndexError/KeyError --
    # must not propagate out of diarize_audio, or pipeline.py's broad except will wipe
    # out speaker_turns for a recording that cannot be made again.
    audio_path = _write_wav(tmp_path / "sample.wav")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [(FakeTurn(0.0, 3.0), None, "SPEAKER_00")]
    fake_diarization.labels.side_effect = AttributeError("boom")
    fake_output = MagicMock(
        speaker_diarization=fake_diarization, speaker_embeddings=[[1.0, 0.0]]
    )

    result = diarize_audio(audio_path, hf_token="t", pipeline=MagicMock(return_value=fake_output))

    assert result.turns == [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    assert result.embeddings == {}


def test_diarize_audio_keeps_the_turns_when_the_embeddings_attribute_itself_raises(tmp_path):
    # getattr(result, "speaker_embeddings", None) only swallows AttributeError from
    # the lookup itself -- if speaker_embeddings were a property that raised anything
    # else, the bare getattr used to sit outside the try and that exception would
    # escape _speaker_embeddings entirely, hitting pipeline.py's broad except and
    # wiping out speaker_turns for a recording that cannot be made again.
    audio_path = _write_wav(tmp_path / "sample.wav")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [(FakeTurn(0.0, 3.0), None, "SPEAKER_00")]
    fake_diarization.labels.return_value = ["SPEAKER_00"]

    class _ResultWithExplodingEmbeddings:
        speaker_diarization = fake_diarization

        @property
        def speaker_embeddings(self):
            raise RuntimeError("boom")

    result = diarize_audio(
        audio_path,
        hf_token="t",
        pipeline=MagicMock(return_value=_ResultWithExplodingEmbeddings()),
    )

    assert result.turns == [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    assert result.embeddings == {}


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
    fake_diarization.labels.return_value = []
    loaded = MagicMock(return_value=MagicMock(speaker_diarization=fake_diarization))

    with patch(
        "src.diarize.load_diarization_pipeline", return_value=loaded
    ) as mock_load:
        diarize_audio(audio_path, hf_token="hf-test-token", pipeline=None)

    mock_load.assert_called_once_with("hf-test-token", DEFAULT_DIARIZATION_MODEL)
    loaded.assert_called_once()
