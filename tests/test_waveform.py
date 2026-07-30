import numpy as np
import pytest
import soundfile as sf

from src.waveform import load_waveform


def _write_wav(path, seconds=0.5, sample_rate=16000, channels=1):
    frames = int(seconds * sample_rate)
    tone = np.sin(np.linspace(0.0, 40.0, frames, dtype="float32"))
    data = tone if channels == 1 else np.stack([tone] * channels, axis=1)
    sf.write(path, data, sample_rate)
    return path


def test_load_waveform_returns_the_shape_pyannote_expects(tmp_path):
    # pyannote รับ {'waveform': (channel, time), 'sample_rate': int} -- ผิดแกนเท่ากับ
    # ป้อนเสียง 1 sample ที่มี 8000 ช่อง
    path = _write_wav(tmp_path / "mono.wav", seconds=0.5)

    loaded = load_waveform(path)

    assert set(loaded) == {"waveform", "sample_rate"}
    assert loaded["sample_rate"] == 16000
    assert tuple(loaded["waveform"].shape) == (1, 8000)


def test_load_waveform_keeps_channels_first_for_stereo(tmp_path):
    # soundfile คืน (time, channel) ต้องสลับแกนเสมอ ไม่ใช่เฉพาะตอน mono ที่บังเอิญถูก
    path = _write_wav(tmp_path / "stereo.wav", seconds=0.25, channels=2)

    loaded = load_waveform(path)

    assert tuple(loaded["waveform"].shape) == (2, 4000)


def test_load_waveform_reports_the_real_sample_rate(tmp_path):
    # อย่า hardcode 16000: enroll/ รับไฟล์ที่ผู้ใช้วางเองได้ และ pyannote resample
    # ให้เองจาก sample_rate ที่เราบอกมันเท่านั้น -- โกหกตรงนี้คือเสียงถูกยืด/บีบ
    path = _write_wav(tmp_path / "8k.wav", seconds=0.5, sample_rate=8000)

    assert load_waveform(path)["sample_rate"] == 8000


def test_load_waveform_gives_float32_samples(tmp_path):
    # torch เจอ float64 แล้ว forward ตายด้วย dtype mismatch กลางทาง
    path = _write_wav(tmp_path / "mono.wav")

    import torch

    assert load_waveform(path)["waveform"].dtype is torch.float32


def test_load_waveform_raises_on_a_file_that_is_not_audio(tmp_path):
    # ต้องไม่คืน dict ว่างเงียบ ๆ -- ผู้เรียกคือ diarize_audio ซึ่ง pipeline.py ดัก
    # exception ไว้แล้วและบันทึกเหตุผลลง activity log ให้ผู้ใช้เห็น
    path = tmp_path / "notaudio.wav"
    path.write_bytes(b"fake audio data")

    with pytest.raises(Exception):
        load_waveform(path)
