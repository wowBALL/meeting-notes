import logging
import tempfile
from pathlib import Path
from typing import Any

from src.audio_convert import convert_to_wav
from src.config import Config
from src.diarize import diarize_audio
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


def process_file(
    audio_path: Path,
    config: Config,
    diarization_pipeline: Any = None,
    whisper_model: Any = None,
) -> Path:
    with tempfile.TemporaryDirectory() as tmp_dir:
        wav_path = Path(tmp_dir) / f"{audio_path.stem}.wav"
        try:
            convert_to_wav(audio_path, wav_path)
        except Exception as e:
            move_to_failed(audio_path, config.failed_dir, f"Audio conversion failed: {e}")
            raise

        try:
            whisper_segments = retry_with_backoff(
                lambda: transcribe_audio(
                    wav_path, model_size=config.whisper_model, model=whisper_model
                )
            )
        except Exception as e:
            move_to_failed(audio_path, config.failed_dir, f"Transcription failed: {e}")
            raise

        diarization_failed = False
        try:
            speaker_turns = diarize_audio(
                wav_path, hf_token=config.hf_token, pipeline=diarization_pipeline
            )
        except Exception as e:
            logger.warning("Diarization failed, continuing without speaker labels: %s", e)
            speaker_turns = []
            diarization_failed = True

    try:
        merged = merge_transcript_and_speakers(whisper_segments, speaker_turns)
        transcript_markdown = render_transcript_markdown(
            merged, diarization_failed=diarization_failed
        )
    except Exception as e:
        move_to_failed(audio_path, config.failed_dir, f"Rendering failed: {e}")
        raise

    # Written before summarizing: the transcript is the expensive artifact of
    # this pipeline (a full GPU pass over the recording), and summarization is
    # the step most likely to fail. Persisting it first means a failed summary
    # costs one Claude call to redo, not another transcription.
    try:
        meeting_dir = create_meeting_folder(audio_path, config.meetings_dir)
        transcript_path = save_transcript(meeting_dir, transcript_markdown)
    except Exception as e:
        move_to_failed(audio_path, config.failed_dir, f"Save failed: {e}")
        raise

    # No retry here on purpose: summarize_transcript retries every API call it
    # makes, per chunk. Wrapping it again would re-run a whole map-reduce
    # (8 chunks, ~27 calls) because of one permanently failing chunk.
    try:
        summary_markdown = summarize_transcript(
            transcript_markdown, model=config.claude_model, api_key=config.anthropic_api_key
        )
    except Exception as e:
        move_to_failed(
            audio_path,
            config.failed_dir,
            f"Summarization failed: {e}\nTranscript was saved to {transcript_path}",
        )
        raise

    try:
        save_summary(meeting_dir, summary_markdown)
        archive_audio(meeting_dir, audio_path)
    except Exception as e:
        move_to_failed(audio_path, config.failed_dir, f"Save failed: {e}")
        raise

    return meeting_dir
