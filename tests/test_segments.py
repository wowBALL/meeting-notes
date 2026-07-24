import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from src.segments import (
    MANIFEST_NAME,
    OPUS_BITRATE,
    SESSION_PREFIX,
    WAV_HEADER_ALLOWANCE,
    build_concat_list,
    ffmpeg_concat_command,
    finish_session,
    find_orphan_sessions,
    part_filename,
    read_manifest,
    session_dir_for,
    write_manifest,
)

# finish_session now decides which parts are "real" by size on disk (see
# WAV_HEADER_ALLOWANCE), so fixture parts representing genuine recorded audio
# must be comfortably larger than the allowance -- a bare `b"fake wav bytes"`
# (14 bytes) would now be mistaken for a header-only, in-progress file.
_FAKE_WAV_BYTES = b"fake wav bytes " * (WAV_HEADER_ALLOWANCE // 16 + 2)


def _make_session(inbox: Path, stem: str, part_count: int, status: str = "recording") -> Path:
    session_dir = session_dir_for(inbox, stem)
    session_dir.mkdir(parents=True)
    parts = []
    for index in range(1, part_count + 1):
        name = part_filename(index)
        (session_dir / name).write_bytes(_FAKE_WAV_BYTES)
        parts.append(name)
    write_manifest(session_dir, stem, "2026-07-24T14:30:05", 48000, parts, status)
    return session_dir


def test_session_dir_for_uses_the_prefix_and_stem(tmp_path):
    assert session_dir_for(tmp_path, "meet1-14-30-05") == tmp_path / f"{SESSION_PREFIX}meet1-14-30-05"


def test_part_filename_is_four_digit_and_one_based():
    assert part_filename(1) == "part0001.wav"
    assert part_filename(42) == "part0042.wav"
    assert part_filename(1234) == "part1234.wav"


def test_manifest_round_trips_every_field(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    write_manifest(
        session_dir, "meet1", "2026-07-24T14:30:05", 48000, ["part0001.wav"], "recording"
    )
    manifest = read_manifest(session_dir)

    assert manifest == {
        "stem": "meet1",
        "started_at": "2026-07-24T14:30:05",
        "samplerate": 48000,
        "parts": ["part0001.wav"],
        "status": "recording",
    }


def test_manifest_is_written_where_read_manifest_looks(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    write_manifest(session_dir, "meet1", "t", 48000, [], "recording")

    assert (session_dir / MANIFEST_NAME).exists()


def test_build_concat_list_lists_every_part_in_order(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    listing = build_concat_list(session_dir, ["part0001.wav", "part0002.wav"])

    lines = listing.strip().splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("file '") and lines[0].endswith("part0001.wav'")
    assert lines[1].endswith("part0002.wav'")


def test_build_concat_list_escapes_apostrophes_in_the_path(tmp_path):
    # ffmpeg's concat demuxer ends a single-quoted token at the next "'", so an
    # apostrophe in a meeting name (straight from sys.argv[1]) must be escaped
    # or the emitted path gets truncated and ffmpeg fails to find the file.
    session_dir = tmp_path / ".session-Bob's standup"
    session_dir.mkdir()
    real_path = str((session_dir / "part0001.wav").resolve())

    listing = build_concat_list(session_dir, ["part0001.wav"])

    line = listing.strip()
    assert line.startswith("file '") and line.endswith("'")
    escaped = line[len("file '") : -1]
    # each real "'" must have become the shell-style close-escape-reopen idiom
    assert escaped == real_path.replace("'", "'\\''")
    # round-trip: undoing that idiom recovers the exact real path
    assert escaped.replace("'\\''", "'") == real_path


def test_ffmpeg_concat_command_has_the_required_flags(tmp_path):
    command = ffmpeg_concat_command(tmp_path / "concat.txt", tmp_path / "out.ogg", "48k")

    assert command[0] == "ffmpeg"
    assert "-f" in command and command[command.index("-f") + 1] == "concat"
    assert "-safe" in command and command[command.index("-safe") + 1] == "0"
    assert command[command.index("-c:a") + 1] == "libopus"
    assert command[command.index("-b:a") + 1] == "48k"
    assert command[-1] == str(tmp_path / "out.ogg")


def test_find_orphan_sessions_returns_sessions_with_manifest_and_parts(tmp_path):
    _make_session(tmp_path, "meet1", part_count=2)

    orphans = find_orphan_sessions(tmp_path)

    assert [d.name for d in orphans] == [f"{SESSION_PREFIX}meet1"]


def test_find_orphan_sessions_ignores_plain_files_and_other_directories(tmp_path):
    (tmp_path / "loose.wav").write_bytes(b"x")
    (tmp_path / "not-a-session").mkdir()
    _make_session(tmp_path, "meet1", part_count=1)

    orphans = find_orphan_sessions(tmp_path)

    assert [d.name for d in orphans] == [f"{SESSION_PREFIX}meet1"]


def test_find_orphan_sessions_skips_sessions_without_a_manifest(tmp_path):
    session_dir = tmp_path / f"{SESSION_PREFIX}broken"
    session_dir.mkdir()
    (session_dir / part_filename(1)).write_bytes(b"x")

    assert find_orphan_sessions(tmp_path) == []


def test_find_orphan_sessions_skips_sessions_without_parts(tmp_path):
    _make_session(tmp_path, "empty", part_count=0)

    assert find_orphan_sessions(tmp_path) == []


def test_find_orphan_sessions_returns_empty_when_inbox_missing(tmp_path):
    assert find_orphan_sessions(tmp_path / "nope") == []


def test_find_orphan_sessions_skips_a_session_whose_parts_are_all_header_only(tmp_path):
    # Stopping the recorder before a single block is written leaves part0001.wav
    # holding nothing but a header. finish_session refuses such a session, so
    # reporting it as recoverable makes every later startup print a failure for a
    # directory that can never be finished.
    session_dir = _make_session(tmp_path, "silent", part_count=0)
    (session_dir / part_filename(1)).write_bytes(b"x" * (WAV_HEADER_ALLOWANCE - 20))

    assert find_orphan_sessions(tmp_path) == []


def test_finish_session_leaves_nothing_recoverable_when_cleanup_fails(tmp_path):
    # rmtree can fail on Windows (lingering handle, AV scanner) after the encode
    # and move already succeeded. Whatever survives must not look like a session
    # worth recovering, or the next run re-encodes it into a duplicate meeting.
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = _make_session(inbox, "meet1", part_count=2)

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with (
        patch("src.segments.subprocess.run", side_effect=fake_run),
        patch("src.segments.shutil.rmtree", side_effect=OSError("file in use")),
    ):
        destination = finish_session(session_dir, inbox)

    assert destination.read_bytes() == b"fake opus"
    assert find_orphan_sessions(inbox) == []


def test_finish_session_encodes_moves_and_cleans_up(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = _make_session(inbox, "meet1", part_count=2)

    def fake_run(command, **kwargs):
        # ffmpeg's job: produce the output file the command names
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with patch("src.segments.subprocess.run", side_effect=fake_run):
        destination = finish_session(session_dir, inbox)

    assert destination == inbox / "meet1.ogg"
    assert destination.read_bytes() == b"fake opus"
    assert not session_dir.exists()


def test_finish_session_passes_every_part_to_ffmpeg(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = _make_session(inbox, "meet1", part_count=3)
    seen = {}

    def fake_run(command, **kwargs):
        concat_path = Path(command[command.index("-i") + 1])
        seen["listing"] = concat_path.read_text(encoding="utf-8")
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with patch("src.segments.subprocess.run", side_effect=fake_run):
        finish_session(session_dir, inbox)

    assert seen["listing"].count("file '") == 3


def test_finish_session_keeps_the_session_when_ffmpeg_fails(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = _make_session(inbox, "meet1", part_count=1)

    with patch(
        "src.segments.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "ffmpeg"),
    ):
        with pytest.raises(subprocess.CalledProcessError):
            finish_session(session_dir, inbox)

    assert session_dir.exists()
    assert (session_dir / part_filename(1)).exists()
    assert not (inbox / "meet1.ogg").exists()


def test_finish_session_raises_when_no_parts_survive(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = _make_session(inbox, "meet1", part_count=1)
    (session_dir / part_filename(1)).unlink()

    with pytest.raises(RuntimeError, match="ไม่พบชิ้นส่วนเสียง"):
        finish_session(session_dir, inbox)


def test_finish_session_includes_the_in_progress_part_missing_from_the_manifest(tmp_path):
    # A crash is always mid-part, so the part being written when the process died
    # was never recorded in the manifest (only closed parts are). It still holds
    # real audio on disk and must not be silently discarded on recovery.
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = _make_session(inbox, "meet1", part_count=1)
    in_progress = part_filename(2)
    (session_dir / in_progress).write_bytes(_FAKE_WAV_BYTES)
    # sanity check on the premise: the manifest (only updated when a part closes)
    # does not yet know about the in-progress part
    assert in_progress not in read_manifest(session_dir)["parts"]
    seen = {}

    def fake_run(command, **kwargs):
        concat_path = Path(command[command.index("-i") + 1])
        seen["listing"] = concat_path.read_text(encoding="utf-8")
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with patch("src.segments.subprocess.run", side_effect=fake_run):
        finish_session(session_dir, inbox)

    assert in_progress in seen["listing"]


def test_finish_session_succeeds_when_manifest_lists_no_parts_but_one_exists_on_disk(tmp_path):
    # The common real-world crash: killed before the first rotation, so
    # manifest["parts"] is still []. The directory listing must still find the
    # part and recover the whole recording instead of raising.
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = session_dir_for(inbox, "meet1")
    session_dir.mkdir(parents=True)
    (session_dir / part_filename(1)).write_bytes(_FAKE_WAV_BYTES)
    write_manifest(session_dir, "meet1", "2026-07-24T14:30:05", 48000, [], "recording")

    def fake_run(command, **kwargs):
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with patch("src.segments.subprocess.run", side_effect=fake_run):
        destination = finish_session(session_dir, inbox)

    assert destination == inbox / "meet1.ogg"
    assert destination.read_bytes() == b"fake opus"


def test_finish_session_excludes_a_header_only_part_from_the_concat_list(tmp_path):
    # A part file left by a killed process before any audio was flushed still has
    # only its (possibly zero-length) RIFF header on disk -- not worth encoding,
    # and including it would just hand ffmpeg a truncated/empty stream.
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = _make_session(inbox, "meet1", part_count=1)
    header_only = part_filename(2)
    (session_dir / header_only).write_bytes(b"RIFF")  # far below WAV_HEADER_ALLOWANCE
    seen = {}

    def fake_run(command, **kwargs):
        concat_path = Path(command[command.index("-i") + 1])
        seen["listing"] = concat_path.read_text(encoding="utf-8")
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with patch("src.segments.subprocess.run", side_effect=fake_run):
        finish_session(session_dir, inbox)

    assert header_only not in seen["listing"]
    assert seen["listing"].count("file '") == 1


def test_finish_session_marks_the_manifest_encoding_before_running_ffmpeg(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    session_dir = _make_session(inbox, "meet1", part_count=1)
    seen = {}

    def fake_run(command, **kwargs):
        seen["status"] = json.loads(
            (session_dir / MANIFEST_NAME).read_text(encoding="utf-8")
        )["status"]
        Path(command[-1]).write_bytes(b"fake opus")
        return subprocess.CompletedProcess(command, 0)

    with patch("src.segments.subprocess.run", side_effect=fake_run):
        finish_session(session_dir, inbox)

    assert seen["status"] == "encoding"


def test_opus_bitrate_default_is_48k():
    assert OPUS_BITRATE == "48k"
