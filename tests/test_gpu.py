import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from src.gpu import cuda_device


def _fake_torch(cuda_available: bool, device_sentinel=None):
    torch_mod = ModuleType("torch")
    torch_mod.cuda = SimpleNamespace(is_available=lambda: cuda_available)
    torch_mod.device = MagicMock(return_value=device_sentinel or object())
    torch_mod.backends = SimpleNamespace(cudnn=SimpleNamespace(enabled=True))
    return torch_mod


def test_cuda_device_returns_the_cuda_device_when_available():
    sentinel = object()
    fake = _fake_torch(True, sentinel)

    with patch.dict(sys.modules, {"torch": fake}):
        assert cuda_device() is sentinel

    fake.device.assert_called_once_with("cuda")


def test_cuda_device_disables_cudnn_before_handing_out_a_device():
    # ctranslate2 ของ faster-whisper กับ torch โหลด cuDNN คนละเวอร์ชันใต้ชื่อ DLL
    # เดียวกัน Windows เก็บ DLL ชื่อละหนึ่งตัวต่อ process -- ตัวที่ init ก่อนทำให้อีกตัว
    # ตายด้วย CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH (วัดจริง 2026-07-24)
    fake = _fake_torch(True)

    with patch.dict(sys.modules, {"torch": fake}):
        cuda_device()

    assert fake.backends.cudnn.enabled is False


def test_cuda_device_returns_none_without_cuda_and_leaves_cudnn_alone():
    # ไม่มี GPU = ไม่มีทางชน DLL อย่าไปแตะ default ของ torch
    fake = _fake_torch(False)

    with patch.dict(sys.modules, {"torch": fake}):
        assert cuda_device() is None

    assert fake.backends.cudnn.enabled is True


def test_cuda_device_returns_none_when_torch_cannot_be_imported():
    # README บอกว่ารันบน macOS/Linux ที่ไม่มี torch+CUDA ได้ -- ต้องตกไป CPU เงียบ ๆ
    with patch.dict(sys.modules, {"torch": None}):
        assert cuda_device() is None


def test_cuda_device_returns_none_when_asking_about_cuda_raises():
    # torch.cuda.is_available() โยน OSError ได้เมื่อไดรเวอร์เพี้ยน -- ต้องไม่พาโปรแกรมล้ม
    fake = _fake_torch(True)
    fake.cuda = SimpleNamespace(is_available=MagicMock(side_effect=OSError("driver")))

    with patch.dict(sys.modules, {"torch": fake}):
        assert cuda_device() is None
