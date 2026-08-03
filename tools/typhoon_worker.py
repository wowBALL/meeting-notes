"""VAD-chunked Typhoon ASR worker -- runs inside .typhoon_venv, not the main .venv.

Typhoon (scb10x/typhoon-asr-realtime, NeMo FastConformer-Transducer) was trained on
utterances <=30s (max_duration in its own config) and has no built-in long-form
chunking, unlike faster-whisper. Feeding it a full meeting directly produces empty
or truncated output. This script does the chunking faster-whisper gets for free:
Silero VAD finds speech spans, adjacent spans are greedily merged into <=28s chunks,
each chunk is transcribed, and (start, end, text) triples are written out --
the same shape src/transcribe.transcribe_audio returns, so the rest of the
pipeline (diarize/merge/render) doesn't need to know which engine produced them.

Kept in a separate venv on purpose: nemo_toolkit pulls ~90 packages (lightning,
wandb, tensorboard, datasets, its own transformers/huggingface_hub pins) that have
no reason to share a dependency resolution with the CUDA faster-whisper/pyannote
stack the rest of this project depends on for every meeting, working today.
Invoked via subprocess from src/transcribe_typhoon.py instead.

Usage: <typhoon_venv python> typhoon_worker.py <16kHz mono wav path> <output json path>
"""

import json
import sys
from pathlib import Path

MAX_CHUNK_S = 28.0


def main() -> None:
    wav_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])

    import numpy as np
    import soundfile as sf
    import torch

    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
    )
    (get_speech_timestamps, *_rest) = utils

    data, sr_in = sf.read(str(wav_path), dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr_in != 16000:
        raise ValueError(f"expected 16kHz input, got {sr_in}")
    wav = torch.from_numpy(np.ascontiguousarray(data))
    speech_ts = get_speech_timestamps(wav, vad_model, sampling_rate=16000, return_seconds=True)

    chunks: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None
    for seg in speech_ts:
        if cur_start is None:
            cur_start, cur_end = seg["start"], seg["end"]
            continue
        if seg["end"] - cur_start <= MAX_CHUNK_S:
            cur_end = seg["end"]
        else:
            chunks.append((cur_start, cur_end))
            cur_start, cur_end = seg["start"], seg["end"]
    if cur_start is not None:
        chunks.append((cur_start, cur_end))

    if not chunks:
        out_path.write_text("[]", encoding="utf-8")
        return

    sr = 16000
    chunk_dir = out_path.parent / f"{out_path.stem}_chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunk_paths = []
    for i, (s, e) in enumerate(chunks):
        sub = wav[int(s * sr) : int(e * sr)].numpy()
        p = chunk_dir / f"chunk_{i:04d}.wav"
        sf.write(str(p), sub, sr)
        chunk_paths.append(str(p))

    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="scb10x/typhoon-asr-realtime", map_location="cpu"
    )
    hypotheses = model.transcribe(audio=chunk_paths, batch_size=8)
    texts = [h.text if hasattr(h, "text") else str(h) for h in hypotheses]

    segments = [
        {"start": s, "end": e, "text": t.strip()}
        for (s, e), t in zip(chunks, texts)
        if t.strip()
    ]
    out_path.write_text(json.dumps(segments, ensure_ascii=False), encoding="utf-8")

    for p in chunk_paths:
        Path(p).unlink(missing_ok=True)
    chunk_dir.rmdir()


if __name__ == "__main__":
    main()
