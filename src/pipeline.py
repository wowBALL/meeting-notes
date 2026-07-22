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
from src.storage import create_meeting_folder, move_to_failed, save_outputs
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

    try:
        summary_markdown = retry_with_backoff(
            lambda: summarize_transcript(
                transcript_markdown, model=config.claude_model, api_key=config.anthropic_api_key
            )
        )
    except Exception as e:
        move_to_failed(audio_path, config.failed_dir, f"Summarization failed: {e}")
        raise

    try:
        meeting_dir = create_meeting_folder(audio_path, config.meetings_dir)
        save_outputs(meeting_dir, audio_path, transcript_markdown, summary_markdown)
    except Exception as e:
        move_to_failed(audio_path, config.failed_dir, f"Save failed: {e}")
        raise

    return meeting_dir
