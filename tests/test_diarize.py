import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from src.config import DEFAULT_DIARIZATION_MODEL
from src.diarize import (
    DiarizationResult,
    diarize_audio,
    load_diarization_pipeline,
)


def _fake_pyannote(mock_pipeline_cls):
    pyannote_pkg = ModuleType("pyannote")
    audio_mod = ModuleType("pyannote.audio")
    audio_mod.Pipeline = mock_pipeline_cls
    pyannote_pkg.audio = audio_mod
    return {"pyannote": pyannote_pkg, "pyannote.audio": audio_mod}


def _fake_torch(cuda_available: bool, device_sentinel):
    torch_mod = ModuleType("torch")
    torch_mod.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    torch_mod.device = MagicMock(return_value=device_sentinel)
    torch_mod.backends = SimpleNamespace(cudnn=SimpleNamespace(enabled=True))
    return torch_mod


class FakeTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


def test_diarize_audio_extracts_speaker_turns_and_embeddings(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

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
    mock_pipeline.assert_called_once_with(str(audio_path))
    fake_diarization.itertracks.assert_called_once_with(yield_label=True)


def test_diarize_audio_keys_embeddings_by_label_not_by_position(tmp_path):
    # pyannote บอกไว้เองว่า array เรียงตาม diarization.labels() ไม่ใช่ตามลำดับที่
    # ผู้พูดโผล่ใน itertracks -- ผูกผิดหนึ่งตำแหน่งแปลว่าลงทะเบียนเสียงผิดคน
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

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
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [(FakeTurn(0.0, 3.0), None, "SPEAKER_00")]
    fake_diarization.labels.return_value = ["SPEAKER_00"]
    fake_output = MagicMock(speaker_diarization=fake_diarization, speaker_embeddings=None)

    result = diarize_audio(audio_path, hf_token="t", pipeline=MagicMock(return_value=fake_output))

    assert result.turns == [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"}]
    assert result.embeddings == {}


def test_diarize_audio_survives_an_unreadable_embedding_array(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

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
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

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
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

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


def test_load_diarization_pipeline_moves_to_gpu_when_available():
    loaded = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = loaded
    cuda_device = object()

    with patch.dict(
        sys.modules,
        {**_fake_pyannote(mock_pipeline_cls), "torch": _fake_torch(True, cuda_device)},
    ):
        result = load_diarization_pipeline("hf-test-token")

    mock_pipeline_cls.from_pretrained.assert_called_once_with(
        DEFAULT_DIARIZATION_MODEL, token="hf-test-token"
    )
    # pyannote defaults to CPU; without this .to() a 50-minute meeting spends
    # 15+ minutes in diarization instead of ~2
    loaded.to.assert_called_once_with(cuda_device)
    assert result is loaded


def test_load_diarization_pipeline_honours_a_custom_checkpoint():
    """DIARIZATION_MODEL ใน .env ต้องไปถึง from_pretrained จริง ไม่ใช่ถูก default ทับ"""
    loaded = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = loaded

    with patch.dict(
        sys.modules,
        {**_fake_pyannote(mock_pipeline_cls), "torch": _fake_torch(False, object())},
    ):
        load_diarization_pipeline("hf-test-token", "pyannote/speaker-diarization-3.1")

    mock_pipeline_cls.from_pretrained.assert_called_once_with(
        "pyannote/speaker-diarization-3.1", token="hf-test-token"
    )


def test_load_diarization_pipeline_disables_torch_cudnn_on_gpu():
    # faster-whisper's ctranslate2 loads the cu12 cuDNN DLLs while torch ships
    # its own cu13 build under the same DLL basenames. Windows dedupes by name,
    # so mixing them dies with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH the
    # moment pyannote's first GPU forward runs after whisper (observed on the
    # real watcher, 2026-07-24). torch must not touch cuDNN at all.
    loaded = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = loaded
    fake_torch = _fake_torch(True, object())

    with patch.dict(sys.modules, {**_fake_pyannote(mock_pipeline_cls), "torch": fake_torch}):
        load_diarization_pipeline("hf-test-token")

    assert fake_torch.backends.cudnn.enabled is False


def test_load_diarization_pipeline_stays_on_cpu_without_cuda():
    loaded = MagicMock()
    mock_pipeline_cls = MagicMock()
    mock_pipeline_cls.from_pretrained.return_value = loaded
    fake_torch = _fake_torch(False, object())

    with patch.dict(
        sys.modules,
        {**_fake_pyannote(mock_pipeline_cls), "torch": fake_torch},
    ):
        result = load_diarization_pipeline("hf-test-token")

    loaded.to.assert_not_called()
    # no GPU -> no DLL clash possible; leave torch's defaults alone
    assert fake_torch.backends.cudnn.enabled is True
    assert result is loaded


def test_diarize_audio_loads_pipeline_via_helper_when_none_given(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = []
    fake_diarization.labels.return_value = []
    loaded = MagicMock(return_value=MagicMock(speaker_diarization=fake_diarization))

    with patch(
        "src.diarize.load_diarization_pipeline", return_value=loaded
    ) as mock_load:
        diarize_audio(audio_path, hf_token="hf-test-token", pipeline=None)

    mock_load.assert_called_once_with("hf-test-token", DEFAULT_DIARIZATION_MODEL)
    loaded.assert_called_once_with(str(audio_path))
