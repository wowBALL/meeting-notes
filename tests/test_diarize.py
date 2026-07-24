import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from src.diarize import diarize_audio, load_diarization_pipeline


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


def test_diarize_audio_extracts_speaker_turns(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"fake audio data")

    fake_diarization = MagicMock()
    fake_diarization.itertracks.return_value = [
        (FakeTurn(0.0, 3.0), None, "SPEAKER_00"),
        (FakeTurn(3.0, 6.0), None, "SPEAKER_01"),
    ]
    fake_output = MagicMock(speaker_diarization=fake_diarization)
    mock_pipeline = MagicMock(return_value=fake_output)

    result = diarize_audio(audio_path, hf_token="test-token", pipeline=mock_pipeline)

    assert result == [
        {"start": 0.0, "end": 3.0, "speaker": "SPEAKER_00"},
        {"start": 3.0, "end": 6.0, "speaker": "SPEAKER_01"},
    ]
    mock_pipeline.assert_called_once_with(str(audio_path))
    fake_diarization.itertracks.assert_called_once_with(yield_label=True)


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
        "pyannote/speaker-diarization-3.1", token="hf-test-token"
    )
    # pyannote defaults to CPU; without this .to() a 50-minute meeting spends
    # 15+ minutes in diarization instead of ~2
    loaded.to.assert_called_once_with(cuda_device)
    assert result is loaded


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
    loaded = MagicMock(return_value=MagicMock(speaker_diarization=fake_diarization))

    with patch(
        "src.diarize.load_diarization_pipeline", return_value=loaded
    ) as mock_load:
        diarize_audio(audio_path, hf_token="hf-test-token", pipeline=None)

    mock_load.assert_called_once_with("hf-test-token")
    loaded.assert_called_once_with(str(audio_path))
