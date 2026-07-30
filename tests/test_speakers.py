import json
from datetime import date
from unittest.mock import patch

import pytest

from src.speakers import (
    DuplicateNameError,
    Match,
    add_sample,
    clean_name,
    cosine_similarity,
    is_usable_embedding,
    load_registry,
    match_known,
    registry_path,
    remove_speaker,
    rename_speaker,
    sample_embedding_model,
    save_registry,
)


def test_load_registry_returns_empty_list_when_file_missing(tmp_path):
    assert load_registry(tmp_path) == []


def test_save_then_load_registry_round_trips(tmp_path):
    speakers = [
        {
            "id": "abc123",
            "name": "สมหญิง็ม",
            "samples": [
                {"embedding": [1.0, 0.0], "source": "2026-07-28_10-30-standup", "added": "2026-07-28"}
            ],
        }
    ]

    save_registry(tmp_path, speakers)

    assert load_registry(tmp_path) == speakers
    assert registry_path(tmp_path) == tmp_path / "speakers" / "registry.json"


def test_save_registry_writes_utf8_json_with_a_version(tmp_path):
    save_registry(tmp_path, [{"id": "a", "name": "สมหญิง็ม", "samples": []}])

    payload = json.loads(registry_path(tmp_path).read_text(encoding="utf-8"))

    assert payload["version"] == 2
    # ห้าม escape เป็น \uXXXX -- ชื่อไทยต้องอ่านออกเมื่อเปิดไฟล์ดูด้วยตา
    assert "สมหญิง็ม" in registry_path(tmp_path).read_text(encoding="utf-8")


def test_load_registry_returns_empty_list_when_file_is_not_json(tmp_path):
    path = registry_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("ไม่ใช่ json", encoding="utf-8")

    assert load_registry(tmp_path) == []


def test_load_registry_drops_entries_with_the_wrong_shape(tmp_path):
    path = registry_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "speakers": [
                    {"id": "a", "name": "ดี", "samples": []},
                    {"id": "b", "samples": []},
                    "ไม่ใช่ dict",
                    {"id": "c", "name": "ไม่มี samples"},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert load_registry(tmp_path) == [{"id": "a", "name": "ดี", "samples": []}]


def test_save_registry_retries_when_windows_holds_the_old_file(tmp_path):
    # WinError 32: ตัวสแกนไวรัส/indexer จับไฟล์ที่เพิ่งปิดไปค้างได้ราวหนึ่งวินาที
    # การเขียนครั้งเดียวแล้วยอมแพ้คือวิธีทำให้ผู้ใช้เสียชื่อที่เพิ่งตั้งไปเฉย ๆ
    save_registry(tmp_path, [])
    attempts = {"count": 0}
    real_replace = type(registry_path(tmp_path)).replace

    def flaky_replace(self, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError("WinError 32")
        return real_replace(self, target)

    with patch("pathlib.Path.replace", flaky_replace), patch("time.sleep"):
        save_registry(tmp_path, [{"id": "a", "name": "ดี", "samples": []}])

    assert attempts["count"] == 3
    assert load_registry(tmp_path) == [{"id": "a", "name": "ดี", "samples": []}]


def test_save_registry_raises_when_the_file_stays_locked(tmp_path):
    with (
        patch("pathlib.Path.replace", side_effect=PermissionError("WinError 32")),
        patch("time.sleep"),
        pytest.raises(PermissionError),
    ):
        save_registry(tmp_path, [])


def test_clean_name_strips_markdown_and_newlines():
    assert clean_name("  สมหญิง็ม  ") == "สมหญิง็ม"
    assert clean_name("พี่\nเอ็ม") == "พี่ เอ็ม"
    assert clean_name("**สมหญิง็ม**") == "สมหญิง็ม"
    assert clean_name("[สมหญิง็ม]") == "สมหญิง็ม"
    assert clean_name("") == ""


def test_clean_name_caps_the_length():
    assert len(clean_name("ก" * 200)) == 60


def test_is_usable_embedding_rejects_the_shapes_pyannote_can_hand_back():
    assert is_usable_embedding([1.0, 0.0]) is True
    # pyannote pad ศูนย์เมื่อจำนวน label มากกว่าจำนวน centroid -- เวกเตอร์ศูนย์ล้วน
    # ไม่มีทิศทาง เก็บเข้าทะเบียนแล้วจะ match กับทุกอย่างที่เป็นศูนย์เหมือนกัน
    assert is_usable_embedding([0.0, 0.0]) is False
    assert is_usable_embedding([]) is False
    assert is_usable_embedding(None) is False
    assert is_usable_embedding("ไม่ใช่เวกเตอร์") is False
    assert is_usable_embedding([1.0, "x"]) is False


def test_is_usable_embedding_rejects_components_that_are_not_finite():
    # inf/nan เข้ามาถึงการ์ดนี้ได้จริง: registry.json กับไฟล์คิวถูกแก้มือได้ตามเจตนา
    # ของโปรเจกต์ (ดู docstring ของ save_registry) และ json.loads ของ Python รับ
    # `Infinity`/`NaN` เปล่า ๆ เป็นตัวเลข
    #
    # inf เคยผ่านการ์ดนี้เพราะ norm ของมันเป็น inf ซึ่ง > 0.0 จริง แต่ cosine ที่ได้
    # เป็น nan และ nan แพ้ `score > best.score` ทุกครั้ง -- ตัวอย่างพิษหนึ่งตัวจึงล็อก
    # best ไว้แล้วกลบตัวอย่างจริงของคนคนนั้นทั้งหมด ซึ่งคืออาการ "ลงทะเบียนแล้วระบบก็ยัง
    # จำไม่ได้" ที่การ์ดนี้มีไว้กันตั้งแต่แรก
    assert is_usable_embedding([float("inf"), 0.0]) is False
    assert is_usable_embedding([float("-inf"), 0.0]) is False
    assert is_usable_embedding([float("nan"), 0.0]) is False


def test_is_usable_embedding_rejects_values_too_large_to_square():
    # ค่าใหญ่แต่ยังจำกัด: `value ** 2` ล้นก่อนที่จะได้ norm แล้ว OverflowError ทะลุ
    # ออกไปจากการ์ด ผู้เรียกดัก ValueError กันทั้งนั้น (session_service แปลเป็น 400
    # bad_embedding) ของที่ไม่ใช่ ValueError จึงกลายเป็น 500 ที่ไม่มีใครอธิบาย --
    # การ์ดต้องตอบว่า "ใช้ไม่ได้" ไม่ใช่ระเบิดใส่ผู้เรียก
    assert is_usable_embedding([1e308, 1e308]) is False


def test_is_usable_embedding_rejects_a_vector_too_small_for_its_norm_to_be_exact():
    # ปลายล่างของปัญหาเดียวกับ 1e308 และร้ายกว่า: ค่าที่เล็กจนกำลังสองตกไปอยู่ในช่วง
    # subnormal (ต่ำกว่า sys.float_info.min) เหลือ precision แค่ไม่กี่บิต norm ที่ได้
    # จึงผิด และ cosine_similarity คืนค่าที่ออกนอก [-1, 1] ได้จริง -- วัดแล้ว
    # (2026-07-30) คู่ข้างล่างนี้ให้ -2.0 คู่ที่เป็นบวกในทำนองเดียวกันจะ >= high แปลว่า
    # match_known ใส่ชื่อผิดคนลง transcript.md ให้เองโดยไม่มีใครกดยืนยัน ซึ่งเป็นความ
    # เสียหายที่โมดูลนี้ทั้งโมดูลมีไว้กัน
    assert is_usable_embedding([-1e-163, 2.3e-162]) is False
    # เกณฑ์นี้ไม่แตะเวกเตอร์จริง: ของจาก pyannote มี norm ราว 3.3 (ดู enroll.py)
    assert is_usable_embedding([1e-150, 1e-150]) is True


def test_is_usable_embedding_rejects_an_integer_too_long_to_be_a_float():
    # JSON ที่แก้มือใส่จำนวนเต็มยาวเท่าไหร่ก็ได้ และ json.loads คืนมาเป็น int ของ Python
    # ที่ไม่จำกัดความยาว ปลายทางทุกทางแปลงเป็น float (add_sample เก็บ, cosine_similarity
    # เทียบ) แล้วได้ OverflowError -- การ์ดที่บอกว่า "ใช้ได้" ทั้งที่ปลายทางใช้ไม่ได้คือ
    # การเลื่อนความพังไปให้ที่ที่ไม่มีใครดักไว้
    assert is_usable_embedding([10**400, 0]) is False


def test_cosine_similarity_scores_identical_orthogonal_and_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
    # ขนาดไม่มีผล ทิศทางเท่านั้นที่นับ
    assert cosine_similarity([1.0, 1.0], [5.0, 5.0]) == pytest.approx(1.0)


def test_cosine_similarity_returns_zero_for_unusable_input():
    assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == 0.0
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == 0.0


EMBED = "pyannote/wespeaker-voxceleb-resnet34-LM"
OTHER_EMBED = "speechbrain/spkrec-ecapa-voxceleb"


def _person(
    name: str,
    embeddings: list[list[float]],
    speaker_id: str = "id-1",
    embedding_model: str = EMBED,
) -> dict:
    return {
        "id": speaker_id,
        "name": name,
        "samples": [
            {
                "embedding": embedding,
                "source": "meeting",
                "embedding_model": embedding_model,
                "added": "2026-07-28",
            }
            for embedding in embeddings
        ],
    }


def test_match_known_names_a_speaker_above_the_high_threshold():
    registry = [_person("สมหญิง็ม", [[1.0, 0.0]])]

    matches = match_known(
        {"SPEAKER_00": [1.0, 0.0]}, registry, high=0.7, low=0.5, embedding_model=EMBED
    )

    assert matches["SPEAKER_00"].name == "สมหญิง็ม"
    assert matches["SPEAKER_00"].speaker_id == "id-1"
    assert matches["SPEAKER_00"].confident is True


def test_match_known_only_suggests_between_the_two_thresholds():
    # cos = 0.6 -> อยู่ระหว่างเกณฑ์: เสนอให้คนยืนยัน แต่ยังไม่ใส่ชื่อให้เอง
    registry = [_person("สมหญิง็ม", [[1.0, 0.0]])]

    matches = match_known(
        {"SPEAKER_00": [0.6, 0.8]}, registry, high=0.7, low=0.5, embedding_model=EMBED
    )

    assert matches["SPEAKER_00"].confident is False
    assert matches["SPEAKER_00"].score == pytest.approx(0.6)


def test_match_known_ignores_anyone_below_the_low_threshold():
    registry = [_person("สมหญิง็ม", [[1.0, 0.0]])]

    matches = match_known(
        {"SPEAKER_00": [0.0, 1.0]}, registry, high=0.7, low=0.5, embedding_model=EMBED
    )

    assert matches == {}


def test_match_known_takes_the_best_sample_across_every_person():
    registry = [
        _person("สมหญิง็ม", [[0.0, 1.0]], speaker_id="id-1"),
        _person("พี่บี", [[0.6, 0.8], [1.0, 0.0]], speaker_id="id-2"),
    ]

    matches = match_known(
        {"SPEAKER_00": [1.0, 0.0]}, registry, high=0.7, low=0.5, embedding_model=EMBED
    )

    assert matches["SPEAKER_00"].name == "พี่บี"
    assert matches["SPEAKER_00"].score == pytest.approx(1.0)


def test_match_known_skips_unusable_embeddings_on_both_sides():
    registry = [_person("สมหญิง็ม", [[0.0, 0.0]])]

    assert (
        match_known({"SPEAKER_00": [1.0, 0.0]}, registry, high=0.7, low=0.5, embedding_model=EMBED)
        == {}
    )
    assert (
        match_known(
            {"SPEAKER_00": [0.0, 0.0]},
            [_person("สมหญิง็ม", [[1.0, 0.0]])],
            high=0.7,
            low=0.5,
            embedding_model=EMBED,
        )
        == {}
    )
    assert (
        match_known(
            {}, [_person("สมหญิง็ม", [[1.0, 0.0]])], high=0.7, low=0.5, embedding_model=EMBED
        )
        == {}
    )


def test_match_known_ignores_a_poisoned_sample_and_still_finds_the_person():
    # ตัวอย่างที่มี inf ให้ cosine เป็น nan และการเทียบใด ๆ กับ nan เป็นเท็จ ดังนั้น
    # `score > best.score` จะไม่ยอมแทนที่ best ที่เป็น nan ไปตลอด ตัวอย่างจริงที่มา
    # ทีหลังแม้จะได้ 1.0 ก็ไม่เคยถูกเลือก และคนคนนี้จะหายไปจากผลทั้งหมด
    #
    # นี่คือเหตุผลที่การ์ดต้องกรองที่ระดับ "ตัวอย่างเดียว" ให้ได้: ทะเบียนของคนหนึ่งคน
    # ที่มีตัวอย่างพิษปนอยู่หนึ่งตัวต้องยังใช้งานได้ ไม่ใช่ตายทั้งโปรไฟล์
    registry = [_person("สมหญิง็ม", [[float("inf"), 0.0], [1.0, 0.0]])]

    matches = match_known({"SPEAKER_00": [1.0, 0.0]}, registry, high=0.7, low=0.5, model=MODEL)

    assert matches["SPEAKER_00"].name == "สมหญิง็ม"
    assert matches["SPEAKER_00"].score == pytest.approx(1.0)
    assert matches["SPEAKER_00"].confident is True


def test_match_known_returns_nothing_for_an_empty_registry():
    assert match_known({"SPEAKER_00": [1.0, 0.0]}, [], high=0.7, low=0.5, embedding_model=EMBED) == {}


def test_match_known_ignores_a_person_with_no_samples():
    registry = [_person("สมหญิง็ม", [])]

    assert (
        match_known({"SPEAKER_00": [1.0, 0.0]}, registry, high=0.7, low=0.5, embedding_model=EMBED)
        == {}
    )


def test_match_known_breaks_a_tie_in_favour_of_whoever_is_first_in_the_registry():
    # ทั้งสองคนได้ cosine เท่ากันเป๊ะกับผู้พูดคนนี้ -- เงื่อนไข `score > best.score`
    # ใน match_known เป็น strict greater-than จึงไม่แทนที่ best เมื่อคะแนนเท่ากัน
    # กติกาคือ "ใครมาก่อนในทะเบียนชนะ" ถ้าวันหนึ่งเปลี่ยนเป็น `>=` พฤติกรรมนี้จะกลับกัน
    # และเทสต์นี้ต้องจับได้
    registry = [
        _person("สมหญิง็ม", [[1.0, 0.0]], speaker_id="id-1"),
        _person("พี่บี", [[1.0, 0.0]], speaker_id="id-2"),
    ]

    matches = match_known(
        {"SPEAKER_00": [1.0, 0.0]}, registry, high=0.7, low=0.5, embedding_model=EMBED
    )

    assert matches["SPEAKER_00"].speaker_id == "id-1"
    assert matches["SPEAKER_00"].name == "สมหญิง็ม"


def test_rename_speaker_changes_the_name_and_keeps_every_sample():
    registry = [_person("ตั้งชื่อผิด", [[1.0, 0.0], [0.9, 0.1]], speaker_id="id-1")]

    updated = rename_speaker(registry, "id-1", "  สมหญิง็ม  ")

    assert updated[0]["name"] == "สมหญิง็ม"
    assert updated[0]["id"] == "id-1"
    # เหตุผลทั้งหมดที่ทำปุ่มนี้: ไม่ต้องลบทิ้งแล้วอัดใหม่เพียงเพราะสะกดผิด
    assert updated[0]["samples"] == registry[0]["samples"]


def test_rename_speaker_cleans_the_name_like_enrolling_does():
    """ชื่อที่แก้ทีหลังต้องปลอดภัยกับ transcript.md เท่ากับชื่อที่ตั้งครั้งแรก"""
    registry = [_person("เดิม", [[1.0, 0.0]], speaker_id="id-1")]

    updated = rename_speaker(registry, "id-1", "**สมหญิง**\nเอ็ม")

    assert updated[0]["name"] == "สมหญิง เอ็ม"


def test_rename_speaker_does_not_mutate_the_list_it_was_given():
    registry = [_person("เดิม", [[1.0, 0.0]], speaker_id="id-1")]

    rename_speaker(registry, "id-1", "ใหม่")

    assert registry[0]["name"] == "เดิม"


def test_rename_speaker_returns_none_for_an_unknown_id():
    registry = [_person("สมหญิง็ม", [[1.0, 0.0]], speaker_id="id-1")]

    assert rename_speaker(registry, "ไม่มีจริง", "ชื่อใหม่") is None


def test_rename_speaker_rejects_a_name_that_cleans_down_to_nothing():
    registry = [_person("สมหญิง็ม", [[1.0, 0.0]], speaker_id="id-1")]

    with pytest.raises(ValueError):
        rename_speaker(registry, "id-1", "  **  ")


def test_rename_speaker_refuses_to_collide_with_someone_else():
    """add_sample ถือว่าชื่อซ้ำ = คนเดิม การรีเนมจึงยุบสองคนเข้าด้วยกันได้ถ้าไม่กัน

    ตรงนั้นผู้ใช้ตั้งใจบอกว่าเป็นคนเดิม ส่วนตรงนี้เขากำลังแก้ตัวสะกด -- การยุบตัวอย่าง
    เสียงของคนละคนเข้าด้วยกันเพราะพิมพ์ผิดกู้กลับไม่ได้
    """
    registry = [
        _person("สมหญิง็ม", [[1.0, 0.0]], speaker_id="id-1"),
        _person("พี่บี", [[0.0, 1.0]], speaker_id="id-2"),
    ]

    with pytest.raises(DuplicateNameError):
        rename_speaker(registry, "id-2", "สมหญิง็ม")


def test_rename_speaker_allows_renaming_to_the_same_name_after_cleaning():
    """แก้แค่ช่องว่าง/อักขระที่ถูกตัดออก ต้องไม่ถูกมองว่าชนกับตัวเอง"""
    registry = [_person("สมหญิง็ม", [[1.0, 0.0]], speaker_id="id-1")]

    updated = rename_speaker(registry, "id-1", "  สมหญิง็ม  ")

    assert updated[0]["name"] == "สมหญิง็ม"


def test_duplicate_name_error_is_a_value_error():
    """ผู้เรียกที่ดัก ValueError แบบกว้าง ๆ อยู่แล้วต้องไม่พังเพราะชนิดใหม่นี้"""
    assert issubclass(DuplicateNameError, ValueError)


def test_match_known_ignores_samples_recorded_under_a_different_model():
    """เวกเตอร์ของสองโมเดลอยู่คนละพื้นที่ -- cosine ข้ามพื้นที่ให้เลขที่ไม่มีความหมาย

    ตัวอย่างนี้เหมือนกันเป๊ะ (cos = 1.0) ซึ่งจะผ่านเกณฑ์ HIGH ไปใส่ชื่อให้อัตโนมัติ
    ถ้าไม่มีการกรองตามโมเดล -- เป็นรูปทรงเดียวกับอันตรายที่เกณฑ์ 0.80 ตั้งมากัน
    """
    registry = [_person("สมหญิง็ม", [[1.0, 0.0]], embedding_model=OTHER_EMBED)]

    assert (
        match_known({"SPEAKER_00": [1.0, 0.0]}, registry, high=0.7, low=0.5, embedding_model=EMBED)
        == {}
    )


def test_match_known_still_uses_samples_from_the_matching_model_alongside_others():
    """สลับโมเดลแล้วคนที่ enroll ไว้ฝั่งนั้นต้องยังถูกจำ ไม่ใช่ทะเบียนตายทั้งใบ"""
    registry = [
        _person("คนของโมเดลอื่น", [[1.0, 0.0]], speaker_id="id-1", embedding_model=OTHER_EMBED),
        _person("คนของโมเดลนี้", [[0.9, 0.1]], speaker_id="id-2", embedding_model=EMBED),
    ]

    matches = match_known(
        {"SPEAKER_00": [1.0, 0.0]}, registry, high=0.7, low=0.5, embedding_model=EMBED
    )

    # 1.0 ของคนแรกชนะถ้านับ แต่ต้องถูกข้าม เหลือ 0.9939 ของคนที่สอง
    assert matches["SPEAKER_00"].name == "คนของโมเดลนี้"


def test_add_sample_records_the_diarization_model_alongside_the_embedding_stamp():
    """`model` (โมเดลแยกผู้พูด) เป็น provenance เสริม ไม่ใช่ป้ายที่ match_known ใช้กรอง"""
    updated = add_sample(
        [],
        "สมหญิง็ม",
        {"embedding": [1.0, 0.0], "embedding_model": EMBED, "model": OTHER_EMBED},
        source="m1",
    )

    assert updated[0]["samples"][0]["model"] == OTHER_EMBED
    assert updated[0]["samples"][0]["embedding_model"] == EMBED


def test_add_sample_creates_a_new_person_with_an_id():
    updated = add_sample(
        [],
        "สมหญิง็ม",
        {"embedding": [1.0, 0.0], "embedding_model": EMBED},
        source="m1",
        today=date(2026, 7, 28),
    )

    assert len(updated) == 1
    assert updated[0]["name"] == "สมหญิง็ม"
    assert updated[0]["id"]
    assert updated[0]["samples"] == [
        {"embedding": [1.0, 0.0], "embedding_model": EMBED, "source": "m1", "added": "2026-07-28"}
    ]


def test_add_sample_appends_to_the_existing_person_when_the_name_matches():
    existing = add_sample(
        [],
        "สมหญิง็ม",
        {"embedding": [1.0, 0.0], "embedding_model": EMBED},
        source="m1",
        today=date(2026, 7, 28),
    )

    updated = add_sample(
        existing,
        "  สมหญิง็ม  ",
        {"embedding": [0.9, 0.1], "embedding_model": EMBED},
        source="m2",
        today=date(2026, 7, 29),
    )

    assert len(updated) == 1
    assert len(updated[0]["samples"]) == 2
    assert updated[0]["id"] == existing[0]["id"]


def test_add_sample_keeps_only_the_most_recent_samples():
    speakers = []
    for index in range(12):
        speakers = add_sample(
            speakers,
            "สมหญิง็ม",
            {"embedding": [float(index), 1.0], "embedding_model": EMBED},
            source=f"m{index}",
            today=date(2026, 7, 28),
        )

    assert len(speakers[0]["samples"]) == 10
    assert speakers[0]["samples"][0]["source"] == "m2"
    assert speakers[0]["samples"][-1]["source"] == "m11"


def test_add_sample_does_not_mutate_the_list_it_was_given():
    original = add_sample(
        [], "สมหญิง็ม", {"embedding": [1.0, 0.0], "embedding_model": EMBED}, source="m1"
    )

    add_sample(original, "พี่บี", {"embedding": [0.0, 1.0], "embedding_model": EMBED}, source="m2")

    assert len(original) == 1


def test_add_sample_rejects_a_name_that_cleans_down_to_nothing():
    with pytest.raises(ValueError):
        add_sample(
            [], "  **  ", {"embedding": [1.0, 0.0], "embedding_model": EMBED}, source="m1"
        )


def _registry_sample(embedding, embedding_model=EMBED, **extra):
    sample = {"embedding": embedding, "source": "meeting", "added": "2026-07-30"}
    if embedding_model is not None:
        sample["embedding_model"] = embedding_model
    sample.update(extra)
    return sample


def _speaker(name, samples):
    return {"id": "id-" + name, "name": name, "samples": samples}


def test_sample_embedding_model_reads_the_stamp():
    assert sample_embedding_model({"embedding_model": EMBED}) == EMBED


def test_sample_embedding_model_is_none_for_an_unstamped_sample():
    # sample ที่เก็บไว้ก่อนมีฟีเจอร์นี้มาจาก centroid ของ pipeline ซึ่งไม่ใช่พื้นที่นี้แน่นอน
    # ต่างจาก sample_model() เดิมที่เดาเป็น 3.1 ได้เพราะตอนนั้นเป็นข้อเท็จจริงของ repo --
    # รอบนี้เดาไม่ได้ และการเดาผิดคือ cosine ข้ามพื้นที่ที่ "บังเอิญสูง" ได้พอ ๆ กับต่ำ
    assert sample_embedding_model({}) is None
    assert sample_embedding_model({"embedding_model": ""}) is None
    assert sample_embedding_model({"embedding_model": "   "}) is None
    assert sample_embedding_model({"embedding_model": 42}) is None


def test_match_known_skips_a_sample_with_no_embedding_model():
    registry = [_speaker("satit", [_registry_sample([1.0, 0.0], embedding_model=None)])]

    matches = match_known(
        {"SPEAKER_00": [1.0, 0.0]}, registry, high=0.6, low=0.4, embedding_model=EMBED
    )

    assert matches == {}


def test_match_known_skips_a_sample_from_another_embedding_space():
    registry = [_speaker("satit", [_registry_sample([1.0, 0.0], OTHER_EMBED)])]

    matches = match_known(
        {"SPEAKER_00": [1.0, 0.0]}, registry, high=0.6, low=0.4, embedding_model=EMBED
    )

    assert matches == {}


def test_match_known_uses_a_sample_whose_stamp_matches():
    registry = [_speaker("satit", [_registry_sample([1.0, 0.0])])]

    matches = match_known(
        {"SPEAKER_00": [1.0, 0.0]}, registry, high=0.6, low=0.4, embedding_model=EMBED
    )

    assert matches["SPEAKER_00"].name == "satit"
    assert matches["SPEAKER_00"].confident is True


def test_match_known_requires_the_embedding_model_by_keyword():
    # ส่งแบบตำแหน่งต้องพัง ไม่ใช่ผ่านไปเป็นค่าของพารามิเตอร์อื่น
    with pytest.raises(TypeError):
        match_known({}, [], 0.6, 0.4, EMBED)


def test_match_known_rejects_embedding_model_none():
    # None ทำให้ stamp is None ของ sample ที่ไม่มีป้ายผ่านเงื่อนไข `!=` ไปเทียบได้ --
    # เปิดช่องโหว่เดิมที่ฟีเจอร์นี้ปิดกลับมา ต้องพังตรงนี้ ไม่ใช่คืน match ที่ไม่มีความหมาย
    registry = [_speaker("legacy", [_registry_sample([1.0, 0.0], embedding_model=None)])]

    with pytest.raises(ValueError):
        match_known({"S": [1.0, 0.0]}, registry, 0.6, 0.4, embedding_model=None)


def test_match_known_rejects_blank_embedding_model():
    with pytest.raises(ValueError):
        match_known({}, [], 0.6, 0.4, embedding_model="")
    with pytest.raises(ValueError):
        match_known({}, [], 0.6, 0.4, embedding_model="   ")


def test_match_known_tolerates_surrounding_whitespace_on_a_valid_stamp():
    # ค่าที่เก็บถูก strip โดย sample_embedding_model อยู่แล้ว -- พารามิเตอร์ที่รับมาก็ต้อง
    # strip เหมือนกัน ไม่งั้นตัวอย่างที่ถูกต้องกลับไม่ถูกจับคู่เพราะช่องว่างรอบ ๆ ไม่ตรงเป๊ะ
    registry = [_speaker("satit", [_registry_sample([1.0, 0.0], embedding_model="embedA")])]

    matches = match_known(
        {"SPEAKER_00": [1.0, 0.0]}, registry, high=0.6, low=0.4, embedding_model=" embedA "
    )

    assert matches["SPEAKER_00"].name == "satit"


def test_add_sample_stores_the_stamp_and_the_evidence():
    updated = add_sample(
        [],
        "satit",
        {
            "embedding": [1.0, 0.0],
            "embedding_model": EMBED,
            "embedding_seconds": 21.4,
            "segment_count": 7,
            "model": "pyannote/speaker-diarization-community-1",
        },
        source="meeting",
        today=date(2026, 7, 30),
    )

    sample = updated[0]["samples"][0]
    assert sample["embedding_model"] == EMBED
    assert sample["embedding_seconds"] == 21.4
    assert sample["segment_count"] == 7
    assert sample["model"] == "pyannote/speaker-diarization-community-1"
    assert sample["source"] == "meeting"
    assert sample["added"] == "2026-07-30"


def test_add_sample_refuses_a_payload_with_no_embedding_model():
    # เขียน sample ที่ไม่มีป้ายลงทะเบียน = สร้าง sample ที่จะถูกข้ามตลอดกาลโดยไม่มีใครรู้
    with pytest.raises(ValueError):
        add_sample([], "satit", {"embedding": [1.0, 0.0]}, source="meeting")


def test_add_sample_refuses_a_payload_with_no_embedding():
    with pytest.raises(ValueError):
        add_sample([], "satit", {"embedding_model": EMBED}, source="meeting")


@pytest.mark.parametrize("bad_model", ["   ", 42, True, ["x"]])
def test_add_sample_refuses_values_the_reader_would_treat_as_unstamped(bad_model):
    # ตรวจผ่าน sample_embedding_model เอง: ตัวเขียนกับตัวอ่านต้องเห็นตรงกันว่าอะไรใช้ได้
    # ค่าพวกนี้ผ่าน `if not sample.get(key)` (truthiness) แต่ sample_embedding_model คืน
    # None ให้ทุกตัว -- ถ้า add_sample ยอมรับ จะได้ sample ที่ match_known ข้ามตลอดกาล
    with pytest.raises(ValueError):
        add_sample(
            [], "satit", {"embedding": [1.0, 0.0], "embedding_model": bad_model}, source="meeting"
        )


def test_add_sample_refuses_a_zero_vector_embedding():
    # เวกเตอร์ศูนย์ล้วนไม่มีทิศทาง cosine ไม่นิยาม -- เก็บเข้าทะเบียนแล้วจะกลายเป็น
    # ตัวอย่างที่ match กับ "ศูนย์" อื่นทุกตัวและไม่ match กับใครที่มีทิศทางจริงเลย
    with pytest.raises(ValueError):
        add_sample(
            [],
            "satit",
            {"embedding": [0.0, 0.0], "embedding_model": EMBED},
            source="meeting",
        )


def test_add_sample_stores_the_stripped_stamp_not_the_raw_payload_value():
    updated = add_sample(
        [],
        "satit",
        {"embedding": [1.0, 0.0], "embedding_model": "  " + EMBED + "  "},
        source="meeting",
    )

    assert updated[0]["samples"][0]["embedding_model"] == EMBED


def test_add_sample_omits_optional_evidence_that_was_not_given():
    updated = add_sample(
        [], "satit", {"embedding": [1.0, 0.0], "embedding_model": EMBED}, source="meeting"
    )

    assert "embedding_seconds" not in updated[0]["samples"][0]
    assert "segment_count" not in updated[0]["samples"][0]


def test_save_registry_writes_version_two(tmp_path):
    save_registry(tmp_path, [_speaker("satit", [_registry_sample([1.0, 0.0])])])

    written = json.loads((tmp_path / "speakers" / "registry.json").read_text(encoding="utf-8"))
    assert written["version"] == 2


def test_load_registry_still_reads_a_version_one_file(tmp_path):
    # ทะเบียนเก่าต้องอ่านได้ต่อ -- ตัวที่คัด sample ที่ใช้ไม่ได้คือ match_known ไม่ใช่ตัวโหลด
    path = tmp_path / "speakers" / "registry.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"version": 1, "speakers": [_speaker("satit", [{"embedding": [1.0]}])]}),
        encoding="utf-8",
    )

    assert len(load_registry(tmp_path)) == 1


def test_remove_speaker_drops_only_the_matching_id():
    speakers = [_person("สมหญิง็ม", [[1.0, 0.0]], "id-1"), _person("พี่บี", [[0.0, 1.0]], "id-2")]

    assert remove_speaker(speakers, "id-1") == [speakers[1]]
    assert remove_speaker(speakers, "ไม่มีจริง") == speakers
