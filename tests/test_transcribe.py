import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import src.transcribe as transcribe_module
from src.transcribe import load_whisper_model, transcribe_audio


def _segment(start, end, text):
    # faster-whisper yields objects with .start/.end/.text attributes
    return SimpleNamespace(start=start, end=end, text=text)


def test_transcribe_audio_extracts_segments_from_injected_model(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = (
        [_segment(0.0, 2.5, "สวัสดีครับ"), _segment(2.5, 5.0, "วันนี้เรามาคุยกัน")],
        SimpleNamespace(language="th"),
    )

    result = transcribe_audio(audio_path, model=mock_model)

    assert result == [
        {"start": 0.0, "end": 2.5, "text": "สวัสดีครับ"},
        {"start": 2.5, "end": 5.0, "text": "วันนี้เรามาคุยกัน"},
    ]


def test_transcribe_audio_calls_model_with_thai_language(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], SimpleNamespace(language="th"))

    transcribe_audio(audio_path, model=mock_model)

    call_args = mock_model.transcribe.call_args
    assert call_args.args[0] == str(audio_path)
    assert call_args.kwargs["language"] == "th"


def test_transcribe_audio_loads_model_by_size_when_none_given(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], SimpleNamespace(language="th"))

    with patch(
        "src.transcribe.load_whisper_model", return_value=mock_model
    ) as mock_load:
        transcribe_audio(audio_path, model_size="medium")

    mock_load.assert_called_once_with("medium")
    mock_model.transcribe.assert_called_once()


def test_select_device_and_compute_prefers_cuda_when_available():
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        device, compute = transcribe_module._select_device_and_compute()
    assert device == "cuda"
    assert compute == "int8_float16"


def test_select_device_and_compute_falls_back_to_cpu():
    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)
    with patch.dict(sys.modules, {"torch": fake_torch}):
        device, compute = transcribe_module._select_device_and_compute()
    assert device == "cpu"
    assert compute == "int8"


def test_load_whisper_model_loads_and_caches_by_size(monkeypatch):
    monkeypatch.setattr(transcribe_module, "_MODEL_CACHE", {})

    constructed = []

    def fake_ctor(size, device=None, compute_type=None):
        constructed.append((size, device, compute_type))
        return f"model-{size}-{device}-{compute_type}"

    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(side_effect=fake_ctor)

    with (
        patch.dict(sys.modules, {"faster_whisper": fake_fw}),
        patch.object(transcribe_module, "_register_cuda_dll_dirs"),
        patch.object(
            transcribe_module, "_select_device_and_compute", return_value=("cuda", "int8_float16")
        ),
    ):
        model1 = load_whisper_model("large-v3")
        model2 = load_whisper_model("large-v3")

    assert model1 == "model-large-v3-cuda-int8_float16"
    assert model2 == model1
    assert constructed == [("large-v3", "cuda", "int8_float16")]  # constructed once


def test_load_whisper_model_registers_cuda_dll_dirs(monkeypatch):
    monkeypatch.setattr(transcribe_module, "_MODEL_CACHE", {})

    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(return_value="m")

    with (
        patch.dict(sys.modules, {"faster_whisper": fake_fw}),
        patch.object(transcribe_module, "_register_cuda_dll_dirs") as mock_reg,
        patch.object(
            transcribe_module, "_select_device_and_compute", return_value=("cpu", "int8")
        ),
    ):
        load_whisper_model("small")

    mock_reg.assert_called_once()
