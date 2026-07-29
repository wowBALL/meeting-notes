import ast
from pathlib import Path
from unittest.mock import patch

from src.main import main


def test_main_creates_required_directories_and_starts_watch_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    with (
        patch("src.main.watch_loop") as mock_watch_loop,
        patch("src.main.load_whisper_model", return_value=object()),
        patch("src.main.load_diarization_pipeline", return_value=object()),
    ):
        main(base_dir=tmp_path)

    assert (tmp_path / "inbox").is_dir()
    assert (tmp_path / "failed").is_dir()
    assert (tmp_path / "meetings").is_dir()
    # finding 5: enroll\ ไม่เคยถูกสร้างมาก่อน -- โฟลเดอร์ที่ README บอกให้วางไฟล์ลงไป
    # ไม่มีอยู่จริงบนเครื่องที่เพิ่ง checkout ใหม่
    assert (tmp_path / "enroll").is_dir()
    mock_watch_loop.assert_called_once()


def test_main_loads_diarization_pipeline_once_and_passes_to_watch_loop(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")

    loaded_pipeline = object()

    with (
        patch("src.main.watch_loop") as mock_watch_loop,
        patch("src.main.load_whisper_model", return_value=object()),
        patch(
            "src.main.load_diarization_pipeline", return_value=loaded_pipeline
        ) as mock_load,
    ):
        main(base_dir=tmp_path)

    # the GPU/CPU placement decision lives in load_diarization_pipeline, so the
    # watcher's long-lived pipeline must come from it -- not a bare from_pretrained
    mock_load.assert_called_once_with("hf-test-token")
    assert (
        mock_watch_loop.call_args.kwargs["diarization_pipeline"] is loaded_pipeline
    )


def test_main_loads_whisper_model_once_and_passes_to_watch_loop(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("HF_TOKEN", "hf-test-token")
    monkeypatch.setenv("WHISPER_MODEL", "medium")

    loaded_whisper_model = object()

    with (
        patch("src.main.watch_loop") as mock_watch_loop,
        patch(
            "src.main.load_whisper_model", return_value=loaded_whisper_model
        ) as mock_load_whisper,
        patch("src.main.load_diarization_pipeline", return_value=object()),
    ):
        main(base_dir=tmp_path)

    mock_load_whisper.assert_called_once_with("medium")
    assert mock_watch_loop.call_args.kwargs["whisper_model"] is loaded_whisper_model


# โมดูลที่ import pyaudiowpatch ซึ่งเป็น fork เฉพาะ Windows ของ PyAudio
WINDOWS_ONLY_MODULES = {"record", "preflight"}


def _src_imports_of(module_name: str) -> set[str]:
    tree = ast.parse(
        (Path(__file__).parent.parent / "src" / f"{module_name}.py").read_text(
            encoding="utf-8"
        )
    )
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("src."):
            found.add(node.module.split(".")[1])
        elif isinstance(node, ast.Import):
            found.update(
                alias.name.split(".")[1]
                for alias in node.names
                if alias.name.startswith("src.")
            )
    return found


def test_the_watcher_never_reaches_the_windows_only_modules():
    """src.main ต้องนำเข้าเฉพาะโมดูลที่รันได้ทุกระบบปฏิบัติการ

    ครึ่งประมวลผล (ถอดเสียง แยกผู้พูด สรุป) เป็น Python ล้วนและใช้บน macOS/Linux ได้
    ส่วนครึ่งอัดเสียงผูกกับ WASAPI ของ Windows ถ้าวันหนึ่งมีใคร import ข้ามฝั่งมา
    การติดตั้งบนเครื่องที่ไม่ใช่ Windows จะพังทันทีโดยไม่มีอะไรเตือน -- เทสนี้คือตัวเตือน
    """
    reachable, pending = set(), ["main"]
    while pending:
        module = pending.pop()
        if module in reachable:
            continue
        reachable.add(module)
        pending.extend(_src_imports_of(module))

    assert WINDOWS_ONLY_MODULES.isdisjoint(reachable), (
        f"src.main ไปถึง {sorted(WINDOWS_ONLY_MODULES & reachable)} "
        "ซึ่งต้องใช้ pyaudiowpatch (Windows เท่านั้น)"
    )
