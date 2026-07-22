from src.merge import merge_transcript_and_speakers


def test_merge_assigns_speaker_by_max_overlap():
    whisper_segments = [
        {"start": 0.0, "end": 2.5, "text": " สวัสดีครับ "},
        {"start": 2.5, "end": 5.0, "text": "ผมมีประเด็นเสนอ"},
    ]
    speaker_turns = [
        {"start": 0.0, "end": 2.6, "speaker": "SPEAKER_00"},
        {"start": 2.4, "end": 5.0, "speaker": "SPEAKER_01"},
    ]

    result = merge_transcript_and_speakers(whisper_segments, speaker_turns)

    assert result == [
        {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00", "text": "สวัสดีครับ"},
        {"start": 2.5, "end": 5.0, "speaker": "SPEAKER_01", "text": "ผมมีประเด็นเสนอ"},
    ]


def test_merge_falls_back_to_unknown_speaker_when_no_turns():
    whisper_segments = [{"start": 0.0, "end": 2.0, "text": "ทดสอบ"}]

    result = merge_transcript_and_speakers(whisper_segments, [])

    assert result == [
        {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_UNKNOWN", "text": "ทดสอบ"}
    ]


def test_merge_falls_back_to_unknown_speaker_when_no_overlap():
    whisper_segments = [{"start": 10.0, "end": 12.0, "text": "ทดสอบ"}]
    speaker_turns = [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]

    result = merge_transcript_and_speakers(whisper_segments, speaker_turns)

    assert result[0]["speaker"] == "SPEAKER_UNKNOWN"
