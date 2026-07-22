import subprocess
from pathlib import Path


def convert_to_wav(audio_path: Path, output_path: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ac", "1", "-ar", "16000", str(output_path)],
        check=True,
        capture_output=True,
    )
    return output_path
