import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Callable

from src.job import write_job
from src.retry import retry_with_backoff

logger = logging.getLogger(__name__)

# A session directory lives inside inbox/ so the finished file can be moved with a
# same-volume os.replace(). watcher.scan_inbox() only looks at files, so a directory
# sitting here is invisible to it and cannot be picked up half-recorded.
SESSION_PREFIX = ".session-"
MANIFEST_NAME = "session.json"
OPUS_BITRATE = "48k"

# Something outside this process opens each freshly written part file the moment
# ffmpeg is done with it and holds it for around a second (measured 2026-07-26:
# WinError 32, released after ~900 ms). One immediate delete attempt therefore
# lost that race on every real recording. Four attempts at 0.5s doubling waits
# out ~3.5s, which covers the observed delay with room to spare while still
# bounding how long the user waits at the console after a meeting.
CLEANUP_ATTEMPTS = 4
CLEANUP_BASE_DELAY = 0.5

# A libsndfile WAV header is well under this; anything larger holds real samples.
# A part left behind by a killed process still has its audio on disk even though
# its RIFF header claims zero length -- ffmpeg reads such a file to EOF.
WAV_HEADER_ALLOWANCE = 100


class NoAudioRecorded(RuntimeError):
    """A session whose parts hold no samples at all -- there is nothing to encode.
    Distinct from an encode failure: the parts are worthless rather than at risk,
    so the caller may discard the session instead of keeping it for recovery."""


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
    devices: dict | None = None,
    claude_model: str | None = None,
) -> None:
    manifest = {
        "stem": stem,
        "started_at": started_at,
        "samplerate": samplerate,
        "parts": list(parts),
        "status": status,
        # Which mic and which loopback this meeting was captured from. Nothing in
        # the audio itself reveals whether the right speaker was tapped, so a
        # recording with a silent far end is otherwise impossible to diagnose.
        "devices": dict(devices or {}),
        # Which model the user picked for THIS meeting at record time. The watcher
        # is a separate long-lived process that read CLAUDE_MODEL once at its own
        # startup, so the choice cannot live in .env -- it has to travel with the
        # recording. finish_session turns this into the inbox job file.
        "claude_model": claude_model,
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


def audio_parts(session_dir: Path) -> list[str]:
    """Part files that actually hold samples, in recording order.

    The manifest lists only parts that have already CLOSED -- it is updated on
    rotation, never mid-part. A crash is by definition mid-part, so the part being
    written when the process died is absent from the manifest even though its audio
    is sitting right there on disk. The directory listing is therefore authoritative
    and the manifest merely advisory; a header-only file (still showing the RIFF/data
    size ffmpeg writes only on clean close) holds nothing and is excluded. 4-digit
    zero-padded names sort lexicographically in the same order they were recorded.
    """
    return sorted(
        path.name
        for path in session_dir.glob("part*.wav")
        if path.stat().st_size > WAV_HEADER_ALLOWANCE
    )


def sweep_unrecoverable_sessions(inbox_dir: Path) -> list[Path]:
    """Delete session directories that can never be finished.

    A session directory without a manifest is either debris from a cleanup whose
    part file stayed locked for longer than finish_session was willing to retry,
    or a crash inside the instant between mkdir and the first manifest write.
    finish_session cannot run without the manifest, so these can only accumulate;
    sweep them at startup. A directory whose files are still locked simply
    survives until the next sweep.

    Until 2026-07-26 this ran on every single recording, because the cleanup made
    one delete attempt into a lock that takes about a second to clear. It is a
    genuine last resort again now that finish_session waits that lock out.
    """
    if not inbox_dir.exists():
        return []
    swept = []
    for entry in sorted(inbox_dir.iterdir()):
        if not entry.is_dir() or not entry.name.startswith(SESSION_PREFIX):
            continue
        if (entry / MANIFEST_NAME).exists():
            continue
        shutil.rmtree(entry, ignore_errors=True)
        if not entry.exists():
            swept.append(entry)
    return swept


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
        # Same size floor finish_session applies: a session holding only header-only
        # parts can never be finished, so reporting it as recoverable would make
        # every future startup attempt it and print the same failure forever.
        if not audio_parts(entry):
            continue
        orphans.append(entry)
    return orphans


def finish_session(
    session_dir: Path, inbox_dir: Path, bitrate: str = OPUS_BITRATE
) -> Path:
    manifest = read_manifest(session_dir)
    stem = manifest["stem"]
    parts = audio_parts(session_dir)
    if not parts:
        raise NoAudioRecorded(f"ไม่พบชิ้นส่วนเสียงใน {session_dir}")

    # Carry the metadata forward: this rewrite is only here to record the
    # "encoding" status, and dropping fields it did not set would lose the model
    # choice that the job file below is built from.
    write_manifest(
        session_dir,
        stem,
        manifest["started_at"],
        manifest["samplerate"],
        parts,
        "encoding",
        devices=manifest.get("devices"),
        claude_model=manifest.get("claude_model"),
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

    # BEFORE the audio is published, not after: os.replace below is the moment the
    # watcher can first see this recording, and it must already be able to find
    # the model choice by then. .get() because a manifest written before this
    # feature existed has no such key.
    claude_model = manifest.get("claude_model")
    if claude_model:
        write_job(inbox_dir, stem, claude_model)

    destination = inbox_dir / f"{stem}.ogg"
    # Atomic within the volume: the watcher never sees a partially written file.
    os.replace(encoded_path, destination)

    # The encode+move already succeeded -- the user's audio is safe at
    # `destination`, so nothing below may raise. A lingering handle or an AV
    # scanner (common on Windows) can make ANY of these deletions fail after the
    # fact -- the first real recording lost both the part unlink and the rmtree
    # to a transient lock. The manifest goes FIRST because find_orphan_sessions
    # requires it: once it is gone, whatever else survives can never qualify as
    # a recoverable session, so the next run cannot re-encode the leftovers into
    # a duplicate meeting. (A .json is also far less likely to be lock-scanned
    # than a fresh media file.)
    for name in [MANIFEST_NAME, "concat.txt", *parts]:
        _remove_with_retry(lambda p=session_dir / name: p.unlink(missing_ok=True), name)
    _remove_with_retry(lambda: shutil.rmtree(session_dir), session_dir.name)
    return destination


def _remove_with_retry(remove: Callable[[], None], target: str) -> None:
    """Delete something a scanner may still be holding, without ever failing the run.

    Nothing here may raise: by this point the encode and the move have succeeded and
    the user's audio is already safe in inbox/, so a deletion that cannot happen must
    not turn a finished recording into an error. What it must NOT do is stay quiet
    about it -- the previous single attempt swallowed the failure, so leftovers piled
    up in inbox/ from 2026-07-24 with nothing anywhere saying why. Whatever survives
    is cleared by sweep_unrecoverable_sessions on the next startup.
    """
    try:
        retry_with_backoff(
            remove, max_retries=CLEANUP_ATTEMPTS, base_delay=CLEANUP_BASE_DELAY
        )
    except OSError as e:
        logger.warning(
            "Could not remove %s after %d attempts, leaving it for the next startup "
            "sweep: %s",
            target,
            CLEANUP_ATTEMPTS,
            e,
        )
