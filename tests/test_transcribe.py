import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import src.transcribe as transcribe_module
from src.transcribe import _register_cuda_dll_dirs, load_whisper_model, transcribe_audio


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


def test_transcribe_audio_requests_batched_inference(tmp_path):
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake audio data")

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([], SimpleNamespace(language="th"))

    transcribe_audio(audio_path, model=mock_model)

    call_args = mock_model.transcribe.call_args
    assert call_args.kwargs["batch_size"] == transcribe_module.BATCH_SIZE
    assert transcribe_module.BATCH_SIZE >= 2


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


def _fake_faster_whisper(ctor):
    fake_fw = ModuleType("faster_whisper")
    fake_fw.WhisperModel = MagicMock(side_effect=ctor)
    # Same weights, batched decoding: the wrapper is what load_whisper_model
    # must hand back so every transcribe call gets batch parallelism.
    fake_fw.BatchedInferencePipeline = MagicMock(
        side_effect=lambda model: f"batched({model})"
    )
    return fake_fw


def test_load_whisper_model_returns_a_batched_pipeline_and_caches_it(monkeypatch):
    monkeypatch.setattr(transcribe_module, "_MODEL_CACHE", {})

    constructed = []

    def fake_ctor(size, device=None, compute_type=None):
        constructed.append((size, device, compute_type))
        return f"model-{size}-{device}-{compute_type}"

    fake_fw = _fake_faster_whisper(fake_ctor)

    with (
        patch.dict(sys.modules, {"faster_whisper": fake_fw}),
        patch.object(transcribe_module, "_register_cuda_dll_dirs"),
        patch.object(
            transcribe_module, "_select_device_and_compute", return_value=("cuda", "int8_float16")
        ),
    ):
        model1 = load_whisper_model("large-v3")
        model2 = load_whisper_model("large-v3")

    assert model1 == "batched(model-large-v3-cuda-int8_float16)"
    assert model2 == model1
    assert constructed == [("large-v3", "cuda", "int8_float16")]  # constructed once
    fake_fw.BatchedInferencePipeline.assert_called_once_with(
        model="model-large-v3-cuda-int8_float16"
    )


def test_load_whisper_model_registers_cuda_dll_dirs(monkeypatch):
    monkeypatch.setattr(transcribe_module, "_MODEL_CACHE", {})

    fake_fw = _fake_faster_whisper(lambda *a, **k: "m")

    with (
        patch.dict(sys.modules, {"faster_whisper": fake_fw}),
        patch.object(transcribe_module, "_register_cuda_dll_dirs") as mock_reg,
        patch.object(
            transcribe_module, "_select_device_and_compute", return_value=("cpu", "int8")
        ),
    ):
        load_whisper_model("small")

    mock_reg.assert_called_once()


def test_register_cuda_dll_dirs_is_noop_on_non_windows():
    with (
        patch.object(transcribe_module.os, "name", "posix"),
        patch.object(
            transcribe_module.os, "add_dll_directory", MagicMock(), create=True
        ) as mock_add_dll_directory,
    ):
        result = _register_cuda_dll_dirs()

    assert result is None
    mock_add_dll_directory.assert_not_called()


def test_register_cuda_dll_dirs_registers_existing_bin_dirs_on_windows(tmp_path):
    cublas_bin = tmp_path / "cublas" / "bin"
    cudnn_bin = tmp_path / "cudnn" / "bin"
    cublas_bin.mkdir(parents=True)
    cudnn_bin.mkdir(parents=True)
    # cuda_nvrtc/bin intentionally left absent to verify only existing dirs register.

    fake_nvidia = ModuleType("nvidia")
    fake_nvidia.__path__ = [str(tmp_path)]

    with (
        patch.dict(sys.modules, {"nvidia": fake_nvidia}),
        patch.object(transcribe_module.os, "name", "nt"),
        patch.object(
            transcribe_module.os, "add_dll_directory", MagicMock(), create=True
        ) as mock_add_dll_directory,
    ):
        _register_cuda_dll_dirs()

    registered = {call.args[0] for call in mock_add_dll_directory.call_args_list}
    assert registered == {str(cublas_bin), str(cudnn_bin)}
