def merge_transcript_and_speakers(
    whisper_segments: list[dict], speaker_turns: list[dict]
) -> list[dict]:
    merged = []
    for seg in whisper_segments:
        seg_start, seg_end = seg["start"], seg["end"]
        best_speaker = None
        best_overlap = 0.0
        for turn in speaker_turns:
            overlap = min(seg_end, turn["end"]) - max(seg_start, turn["start"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]
        merged.append(
            {
                "start": seg_start,
                "end": seg_end,
                "speaker": best_speaker or "SPEAKER_UNKNOWN",
                "text": seg["text"].strip(),
            }
        )
    return merged
