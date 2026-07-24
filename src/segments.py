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

# A libsndfile WAV header is well under this; anything larger holds real samples.
# A part left behind by a killed process still has its audio on disk even though
# its RIFF header claims zero length -- ffmpeg reads such a file to EOF.
WAV_HEADER_ALLOWANCE = 100


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


def _concat_quote(path: Path) -> str:
    # The concat demuxer ends a single-quoted token at the next "'", so an
    # apostrophe in a meeting name would truncate the path. Close the quote,
    # emit an escaped quote, reopen -- the shell-style idiom the demuxer accepts.
    return str(path).replace("'", "'\\''")


def build_concat_list(session_dir: Path, parts: list[str]) -> str:
    # ffmpeg's concat demuxer format; absolute paths paired with -safe 0 so the
    # command does not depend on the working directory.
    return "".join(
        f"file '{_concat_quote((session_dir / part).resolve())}'\n" for part in parts
    )


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
    # The manifest lists only parts that have already CLOSED -- it is updated on
    # rotation, never mid-part. A crash is by definition mid-part, so the part
    # being written when the process died is absent from the manifest even though
    # its audio is sitting right there on disk. The directory listing is therefore
    # authoritative and the manifest is merely advisory; a header-only file (still
    # showing the RIFF/data size ffmpeg writes only on clean close) is excluded as
    # not worth encoding. 4-digit zero-padded names sort lexicographically in the
    # same order they were recorded.
    parts = sorted(
        path.name
        for path in session_dir.glob("part*.wav")
        if path.stat().st_size > WAV_HEADER_ALLOWANCE
    )
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
    try:
        shutil.rmtree(session_dir)
    except Exception:
        # The encode+move already succeeded -- the user's audio is safe at
        # `destination`. A lingering handle or an AV scanner (common on Windows)
        # can make rmtree fail after the fact; that must not be reported as an
        # encode failure, and the caller must still get the destination back.
        # Note: the surviving directory will be picked up and re-encoded by the
        # next find_orphan_sessions pass (producing a duplicate .ogg) unless it
        # is cleaned up manually -- but the recording itself is never lost.
        pass
    return destination
