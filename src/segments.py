import json
import os
import shutil
import subprocess
from pathlib import Path

# A session directory lives inside inbox/ so the finished file can be moved with a
# same-volume os.replace(). watcher.scan_inbox() only looks at files, so a directory
# sitting here is invisible to it and cannot be picked up half-recorded.
SESSION_PREFIX = ".session-"
MANIFEST_NAME = "session.json"
OPUS_BITRATE = "48k"


def session_dir_for(inbox_dir: Path, stem: str) -> Path:
    return inbox_dir / f"{SESSION_PREFIX}{stem}"


def part_filename(index: int) -> str:
    return f"part{index:04d}.wav"


def write_manifest(
    session_dir: Path,
    stem: str,
    started_at: str,
    samplerate: int,
    parts: list[str],
    status: str,
) -> None:
    manifest = {
        "stem": stem,
        "started_at": started_at,
        "samplerate": samplerate,
        "parts": list(parts),
        "status": status,
    }
    (session_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def read_manifest(session_dir: Path) -> dict:
    return json.loads((session_dir / MANIFEST_NAME).read_text(encoding="utf-8"))


def build_concat_list(session_dir: Path, parts: list[str]) -> str:
    # ffmpeg's concat demuxer format; absolute paths paired with -safe 0 so the
    # command does not depend on the working directory.
    return "".join(f"file '{(session_dir / part).resolve()}'\n" for part in parts)


def ffmpeg_concat_command(
    concat_list_path: Path, output_path: Path, bitrate: str = OPUS_BITRATE
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list_path),
        "-c:a",
        "libopus",
        "-b:a",
        bitrate,
        str(output_path),
    ]


def find_orphan_sessions(inbox_dir: Path) -> list[Path]:
    # A session directory that still exists means the previous run never finished:
    # a successful finish_session() removes it.
    if not inbox_dir.exists():
        return []
    orphans = []
    for entry in sorted(inbox_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith(SESSION_PREFIX):
            continue
        if not (entry / MANIFEST_NAME).exists():
            continue
        if not any(entry.glob("part*.wav")):
            continue
        orphans.append(entry)
    return orphans


def finish_session(
    session_dir: Path, inbox_dir: Path, bitrate: str = OPUS_BITRATE
) -> Path:
    manifest = read_manifest(session_dir)
    stem = manifest["stem"]
    parts = [name for name in manifest["parts"] if (session_dir / name).exists()]
    if not parts:
        raise RuntimeError(f"ไม่พบชิ้นส่วนเสียงใน {session_dir}")

    write_manifest(
        session_dir,
        stem,
        manifest["started_at"],
        manifest["samplerate"],
        parts,
        "encoding",
    )

    concat_list_path = session_dir / "concat.txt"
    concat_list_path.write_text(build_concat_list(session_dir, parts), encoding="utf-8")

    encoded_path = session_dir / f"{stem}.ogg"
    # check=True: on failure the session directory is left alone so the raw parts
    # survive for the next recovery attempt.
    subprocess.run(
        ffmpeg_concat_command(concat_list_path, encoded_path, bitrate),
        check=True,
        capture_output=True,
    )

    destination = inbox_dir / f"{stem}.ogg"
    # Atomic within the volume: the watcher never sees a partially written file.
    os.replace(encoded_path, destination)
    shutil.rmtree(session_dir)
    return destination
