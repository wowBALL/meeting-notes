import logging
import tempfile
from pathlib import Path
from typing import Any

from src import activity
from src.audio_convert import convert_to_wav
from src.config import Config
from src.diarize import diarize_audio
from src.job import (
    NO_SUMMARY_MODEL,
    discard_job,
    read_model,
    read_transcript,
    record_transcript,
)
from src.merge import merge_transcript_and_speakers
from src.render import render_transcript_markdown
from src.retry import retry_with_backoff
from src.storage import (
    archive_audio,
    create_meeting_folder,
    move_to_failed,
    save_summary,
    save_transcript,
)
from src.summarize import summarize_transcript
from src.transcribe import transcribe_audio

logger = logging.getLogger(__name__)


def _reuse_saved_transcript(audio_path: Path) -> tuple[Path, Path, str] | None:
    """(meeting dir, transcript path, transcript) left by an earlier run, or None.

    A recording only comes back through here after a run that got as far as saving
    the transcript and then failed -- almost always at the summary. Transcribing it
    again would spend another full GPU pass to reproduce a file that is already on
    disk, byte for byte. Anything unexpected returns None and the caller does the
    whole pipeline, which is exactly the old behaviour.
    """
    transcript_path = read_transcript(audio_path)
    if transcript_path is None:
        return None
    try:
        transcript_markdown = transcript_path.read_text(encoding="utf-8")
    except OSError as e:
        logger.warning(
            "Cannot read the saved transcript %s, transcribing again: %s",
            transcript_path,
            e,
        )
        return None
    if not transcript_markdown.strip():
        logger.warning(
            "The saved transcript %s is empty, transcribing again", transcript_path
        )
        return None
    logger.info("Reusing the transcript from an earlier run: %s", transcript_path)
    return transcript_path.parent, transcript_path, transcript_markdown


def process_file(
    audio_path: Path,
    config: Config,
    diarization_pipeline: Any = None,
    whisper_model: Any = None,
) -> Path:
    # The recorder wrote this next to the audio; the watcher's own config was read
    # once at startup and cannot know what this meeting asked for.
    claude_model = read_model(audio_path) or config.claude_model

    # ทุก activity.append ที่นี่ไม่มีทาง raise (ดู src/activity.py) จึงไม่ห่อ try --
    # ห่อแล้วจะบังคับให้คนอ่านคิดว่ามันอาจ raise ได้ ซึ่งไม่จริง
    job = audio_path.stem
    activity.append(config.base_dir, job, "queued")

    reused = _reuse_saved_transcript(audio_path)
    if reused is not None:
        meeting_dir, transcript_path, transcript_markdown = reused
        return _finish_meeting(
            audio_path,
            config,
            claude_model,
            meeting_dir,
            transcript_path,
            transcript_markdown,
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / f"{audio_path.stem}.wav"
        try:
            convert_to_wav(audio_path, wav_path)
        except Exception as e:
            activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
            move_to_failed(audio_path, config.failed_dir, f"Audio conversion failed: {e}")
            raise

        activity.append(config.base_dir, job, "transcribe_started")
        try:
            whisper_segments = retry_with_backoff(
                lambda: transcribe_audio(
                    wav_path, model_size=config.whisper_model, model=whisper_model
                )
            )
        except Exception as e:
            activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
            move_to_failed(audio_path, config.failed_dir, f"Transcription failed: {e}")
            raise

        activity.append(config.base_dir, job, "diarize_started")
        diarization_failed = False
        try:
            speaker_turns = diarize_audio(
                wav_path, hf_token=config.hf_token, pipeline=diarization_pipeline
            )
        except Exception as e:
            logger.warning("Diarization failed, continuing without speaker labels: %s", e)
            activity.append(
                config.base_dir, job, "diarize_failed", "warn", {"error": str(e)}
            )
            speaker_turns = []
            diarization_failed = True

    try:
        merged = merge_transcript_and_speakers(whisper_segments, speaker_turns)
        transcript_markdown = render_transcript_markdown(
            merged, diarization_failed=diarization_failed
        )
    except Exception as e:
        activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
        move_to_failed(audio_path, config.failed_dir, f"Rendering failed: {e}")
        raise

    # Written before summarizing: the transcript is the expensive artifact of
    # this pipeline (a full GPU pass over the recording), and summarization is
    # the step most likely to fail. Persisting it first means a failed summary
    # costs one summarizer call to redo, not another transcription.
    try:
        meeting_dir = create_meeting_folder(audio_path, config.meetings_dir)
        transcript_path = save_transcript(meeting_dir, transcript_markdown)
    except Exception as e:
        activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
        move_to_failed(audio_path, config.failed_dir, f"Save failed: {e}")
        raise

    # Saving it is not enough to reuse it: a retry has to be able to find it. The
    # folder name cannot be recomputed for a named recording, whose date comes
    # from the day the folder was made rather than from the file, so the path is
    # written down next to the audio and travels with it into failed/.
    record_transcript(audio_path, transcript_path)

    return _finish_meeting(
        audio_path,
        config,
        claude_model,
        meeting_dir,
        transcript_path,
        transcript_markdown,
    )


def _finish_meeting(
    audio_path: Path,
    config: Config,
    claude_model: str,
    meeting_dir: Path,
    transcript_path: Path,
    transcript_markdown: str,
) -> Path:
    """The half of the pipeline that runs whether or not the transcript is new."""
    # No retry here on purpose: summarize_transcript retries every API call it
    # makes, per chunk. Wrapping it again would re-run a whole map-reduce
    # (8 chunks + 1 reduce, up to 54 calls at worst -- see MAP_MAX_CONCURRENCY's
    # comment in src/summarize.py for how that bound is derived) because of one
    # permanently failing chunk.
    job = audio_path.stem
    summary_markdown = None
    if claude_model != NO_SUMMARY_MODEL:
        activity.append(
            config.base_dir, job, "summarize_started", params={"model": claude_model}
        )
        try:
            summary_markdown = summarize_transcript(
                transcript_markdown, model=claude_model
            )
        except Exception as e:
            activity.append(
                config.base_dir, job, "job_failed", "error", {"error": str(e)}
            )
            move_to_failed(
                audio_path,
                config.failed_dir,
                f"Summarization failed: {e}\nTranscript was saved to {transcript_path}",
            )
            raise

    try:
        # save_summary stamps the model name into the file's footer, so calling it
        # in transcript-only mode would write "สรุปด้วย transcript-only" under a
        # summary nobody asked for. Everything else below still runs: the meeting
        # is finished, just without that one file.
        if summary_markdown is not None:
            save_summary(meeting_dir, summary_markdown, claude_model)
        archive_audio(meeting_dir, audio_path)
        discard_job(audio_path)
    except Exception as e:
        activity.append(config.base_dir, job, "job_failed", "error", {"error": str(e)})
        move_to_failed(audio_path, config.failed_dir, f"Save failed: {e}")
        raise

    activity.append(
        config.base_dir, job, "meeting_done", params={"path": str(meeting_dir)}
    )
    return meeting_dir
