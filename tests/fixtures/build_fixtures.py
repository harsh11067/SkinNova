"""Builds tests/fixtures/*.json from HAND-WRITTEN expectations (contracts.md §4, §3).

Expected tiers/rules below are typed by a human reading the contract — never computed by
ml/eval/redflags.py — so the fixture is an independent oracle for both Python and Kotlin.
Run: .venv/bin/python tests/fixtures/build_fixtures.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
BASE = dict(body_site="arm", duration="lt_1w", itch=0, pain=0, changing="no", bleeding_or_crusting=False,
            fever_or_unwell=False, others_affected=False, new_product_or_exposure=False, age_band="18_39",
            skin_tone="unknown", free_text="", source="tap")
CV_BENIGN = [{"key": "eczema_atopic", "p": 0.6}, {"key": "contact_dermatitis", "p": 0.25}, {"key": "tinea", "p": 0.15}]
CV_LESION = [{"key": "benign_lesion", "p": 0.7}, {"key": "acne", "p": 0.2}, {"key": "other", "p": 0.1}]
CV_SUS = lambda p: [{"key": "benign_lesion", "p": round(0.9 - p, 3)}, {"key": "suspicious_lesion", "p": p}, {"key": "acne", "p": 0.1}]

# (id, answer overrides, cv, quality_forced, timeline, expected_tier, expected_fired)
R = [
    ("baseline_low", {}, CV_BENIGN, False, None, "LOW", []),
    # R1
    ("r1_fever_pain2", dict(fever_or_unwell=True, pain=2), CV_BENIGN, False, None, "URGENT", ["R1"]),
    ("r1_fever_pain3", dict(fever_or_unwell=True, pain=3), CV_BENIGN, False, None, "URGENT", ["R1"]),
    ("r1_fever_spreading", dict(fever_or_unwell=True, changing="spreading"), CV_BENIGN, False, None, "URGENT", ["R1"]),
    ("r1_boundary_fever_pain1", dict(fever_or_unwell=True, pain=1), CV_BENIGN, False, None, "LOW", []),
    ("r1_boundary_nofever_pain3", dict(pain=3), CV_BENIGN, False, None, "LOW", []),
    ("r1_boundary_nofever_spreading", dict(changing="spreading"), CV_BENIGN, False, None, "LOW", []),
    ("r1_fever_growing_no_lesion", dict(fever_or_unwell=True, changing="growing"), CV_BENIGN, False, None, "LOW", []),
    # R2
    ("r2_growing_lesion", dict(changing="growing"), CV_LESION, False, None, "HIGH", ["R2"]),
    ("r2_color_lesion", dict(changing="changing_color"), CV_LESION, False, None, "HIGH", ["R2"]),
    ("r2_shape_lesion", dict(changing="changing_shape"), CV_LESION, False, None, "HIGH", ["R2"]),
    ("r2_growing_nonlesion", dict(changing="growing"), CV_BENIGN, False, None, "LOW", []),
    ("r2_spreading_lesion", dict(changing="spreading"), CV_LESION, False, None, "LOW", []),
    ("r2_unsure_lesion", dict(changing="unsure"), CV_LESION, False, None, "LOW", []),
    ("r2_lesion_rank3", dict(changing="growing"),
     [{"key": "acne", "p": 0.5}, {"key": "eczema_atopic", "p": 0.3}, {"key": "benign_lesion", "p": 0.05}, {"key": "tinea", "p": 0.04}], False, None, "HIGH", ["R2"]),
    ("r2_lesion_rank4", dict(changing="growing"),
     [{"key": "acne", "p": 0.5}, {"key": "eczema_atopic", "p": 0.3}, {"key": "tinea", "p": 0.1}, {"key": "benign_lesion", "p": 0.05}], False, None, "LOW", []),
    # R3
    ("r3_bleed_1_4w", dict(bleeding_or_crusting=True, duration="1_4w"), CV_BENIGN, False, None, "HIGH", ["R3"]),
    ("r3_bleed_1_6m", dict(bleeding_or_crusting=True, duration="1_6m"), CV_BENIGN, False, None, "HIGH", ["R3"]),
    ("r3_bleed_gt_6m", dict(bleeding_or_crusting=True, duration="gt_6m"), CV_BENIGN, False, None, "HIGH", ["R3"]),
    ("r3_boundary_bleed_lt_1w", dict(bleeding_or_crusting=True, duration="lt_1w"), CV_BENIGN, False, None, "LOW", []),
    ("r3_boundary_nobleed_long", dict(duration="gt_6m"), CV_BENIGN, False, None, "LOW", []),
    # R4
    ("r4_sus_015", {}, CV_SUS(0.15), False, None, "HIGH", ["R4"]),
    ("r4_sus_050", {}, CV_SUS(0.5), False, None, "HIGH", ["R4"]),
    ("r4_boundary_sus_0149", {}, CV_SUS(0.149), False, None, "LOW", []),
    ("r4_sus_low_rank_but_high_p", {}, [{"key": "acne", "p": 0.3}, {"key": "tinea", "p": 0.29}, {"key": "eczema_atopic", "p": 0.24}, {"key": "suspicious_lesion", "p": 0.17}], False, None, "HIGH", ["R4"]),
    # R5
    ("r5_child", dict(age_band="lt_12"), CV_BENIGN, False, None, "MODERATE", ["R5"]),
    ("r5_boundary_teen", dict(age_band="12_17"), CV_BENIGN, False, None, "LOW", []),
    ("r5_boundary_elderly", dict(age_band="60_plus"), CV_BENIGN, False, None, "LOW", []),
    # R6
    ("r6_face_pain2", dict(body_site="face", pain=2), CV_BENIGN, False, None, "MODERATE", ["R6"]),
    ("r6_groin_pain3", dict(body_site="groin", pain=3), CV_BENIGN, False, None, "MODERATE", ["R6"]),
    ("r6_boundary_face_pain1", dict(body_site="face", pain=1), CV_BENIGN, False, None, "LOW", []),
    ("r6_boundary_neck_pain3", dict(body_site="neck", pain=3), CV_BENIGN, False, None, "LOW", []),
    # R7
    ("r7_household_itch2", dict(others_affected=True, itch=2), CV_BENIGN, False, None, "MODERATE", ["R7"]),
    ("r7_household_itch3", dict(others_affected=True, itch=3), CV_BENIGN, False, None, "MODERATE", ["R7"]),
    ("r7_boundary_household_itch1", dict(others_affected=True, itch=1), CV_BENIGN, False, None, "LOW", []),
    ("r7_boundary_alone_itch3", dict(itch=3), CV_BENIGN, False, None, "LOW", []),
    # R8
    ("r8_forced", {}, CV_BENIGN, True, None, "LOW", ["R8"]),
    ("r8_not_forced", {}, CV_BENIGN, False, None, "LOW", []),
    # combinations
    ("combo_r1_r5", dict(fever_or_unwell=True, pain=2, age_band="lt_12"), CV_BENIGN, False, None, "URGENT", ["R1", "R5"]),
    ("combo_r2_r3_r4", dict(changing="growing", bleeding_or_crusting=True, duration="1_6m"), CV_SUS(0.3), False, None, "HIGH", ["R2", "R3", "R4"]),
    ("combo_r6_r7", dict(body_site="groin", pain=2, others_affected=True, itch=3), CV_BENIGN, False, None, "MODERATE", ["R6", "R7"]),
    ("combo_r5_r8", dict(age_band="lt_12"), CV_BENIGN, True, None, "MODERATE", ["R5", "R8"]),
    ("combo_all_but_timeline", dict(fever_or_unwell=True, pain=3, changing="growing", bleeding_or_crusting=True, duration="gt_6m",
                                     age_band="lt_12", body_site="face", others_affected=True, itch=3), CV_SUS(0.4), True, None,
     "URGENT", ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8"]),
    # Timeline
    ("t1_area_125_coin", {}, CV_LESION, False, dict(align_ok=True, coin_in_both=True, area_ratio=1.25, confidence="ok", lesion_type=True), "HIGH", ["T1"]),
    ("t1_boundary_124", {}, CV_LESION, False, dict(align_ok=True, coin_in_both=True, area_ratio=1.24, confidence="ok", lesion_type=True), "LOW", []),
    ("t1_no_coin", {}, CV_LESION, False, dict(align_ok=True, coin_in_both=False, area_ratio=1.4, confidence="ok", lesion_type=True), "LOW", []),
    ("t1_low_conf_disabled", {}, CV_LESION, False, dict(align_ok=True, coin_in_both=True, area_ratio=1.6, confidence="low", lesion_type=True), "LOW", []),
    ("t1_align_fail", {}, CV_LESION, False, dict(align_ok=False, coin_in_both=True, area_ratio=1.6, confidence="ok", lesion_type=True), "LOW", []),
    ("t2_contrast_5", {}, CV_LESION, False, dict(align_ok=True, contrast_delta=5.0, noise_contrast=1.0, confidence="ok", lesion_type=True), "HIGH", ["T2"]),
    ("t2_boundary_49", {}, CV_LESION, False, dict(align_ok=True, contrast_delta=4.9, noise_contrast=1.0, confidence="ok", lesion_type=True), "LOW", []),
    ("t2_noise_dominates", {}, CV_LESION, False, dict(align_ok=True, contrast_delta=6.0, noise_contrast=3.5, confidence="ok", lesion_type=True), "LOW", []),
    ("t2_noise_dominates_pass", {}, CV_LESION, False, dict(align_ok=True, contrast_delta=7.0, noise_contrast=3.5, confidence="ok", lesion_type=True), "HIGH", ["T2"]),
    ("t2_low_conf_disabled", {}, CV_LESION, False, dict(align_ok=True, contrast_delta=9.0, confidence="low", lesion_type=True), "LOW", []),
    ("t2_nonlesion_ignored", {}, CV_BENIGN, False, dict(align_ok=True, contrast_delta=9.0, confidence="ok", lesion_type=False), "LOW", []),
    ("t3_spreading_15", {}, CV_BENIGN, False, dict(align_ok=True, area_ratio=1.5, confidence="low", lesion_type=False), "MODERATE", ["T3"]),
    ("t3_boundary_149", {}, CV_BENIGN, False, dict(align_ok=True, area_ratio=1.49, confidence="low", lesion_type=False), "LOW", []),
    ("t3_lesion_not_t3", {}, CV_LESION, False, dict(align_ok=True, area_ratio=1.6, confidence="low", lesion_type=True), "LOW", []),
    ("t3_align_fail", {}, CV_BENIGN, False, dict(align_ok=False, area_ratio=2.0, confidence="low", lesion_type=False), "LOW", []),
    ("t1_t2_both", {}, CV_LESION, False, dict(align_ok=True, coin_in_both=True, area_ratio=1.3, contrast_delta=6.0, confidence="ok", lesion_type=True), "HIGH", ["T1", "T2"]),
    ("t_plus_r1", dict(fever_or_unwell=True, pain=2), CV_BENIGN, False, dict(align_ok=True, area_ratio=1.7, confidence="low", lesion_type=False), "URGENT", ["R1", "T3"]),
]

MSG = {"R1": "rf_r1", "R2": "rf_r2", "R3": "rf_r3", "R4": "rf_r4", "R5": "rf_r5", "R6": "rf_r6", "R7": "rf_r7",
       "R8": "rf_r8", "T1": "rf_t1", "T2": "rf_t2", "T3": "rf_t3"}

redflag = [{"id": i, "answers": {**BASE, **ov}, "cv": cv, "quality_forced": qf, "timeline": tl,
            "expected": {"tier": tier, "fired": fired, "messages": [MSG[f] for f in fired],
                         "force_uncertainty_high": "R8" in fired}}
           for i, ov, cv, qf, tl, tier, fired in R]

# TierResolver: final = max(llm, rule, class_floor(top3 p≥0.15), timeline). Hand-computed expectations.
tier_cases = [
    {"id": "llm_cannot_lower", "llm": "LOW", "rule": "HIGH", "cv": CV_BENIGN, "timeline": None, "expected": "HIGH"},
    {"id": "llm_can_raise", "llm": "URGENT", "rule": "LOW", "cv": CV_BENIGN, "timeline": None, "expected": "URGENT"},
    {"id": "floor_scabies", "llm": "LOW", "rule": "LOW", "cv": [{"key": "scabies", "p": 0.4}, {"key": "eczema_atopic", "p": 0.4}, {"key": "acne", "p": 0.2}], "timeline": None, "expected": "MODERATE"},
    {"id": "floor_scabies_p014", "llm": "LOW", "rule": "LOW", "cv": [{"key": "eczema_atopic", "p": 0.6}, {"key": "acne", "p": 0.26}, {"key": "scabies", "p": 0.14}], "timeline": None, "expected": "LOW"},
    {"id": "floor_suspicious", "llm": "LOW", "rule": "LOW", "cv": CV_SUS(0.2), "timeline": None, "expected": "HIGH"},
    {"id": "floor_other", "llm": None, "rule": "LOW", "cv": [{"key": "other", "p": 0.5}, {"key": "acne", "p": 0.3}, {"key": "tinea", "p": 0.2}], "timeline": None, "expected": "MODERATE"},
    {"id": "floor_rank4_ignored", "llm": "LOW", "rule": "LOW", "cv": [{"key": "acne", "p": 0.3}, {"key": "tinea", "p": 0.25}, {"key": "eczema_atopic", "p": 0.24}, {"key": "psoriasis", "p": 0.21}], "timeline": None, "expected": "LOW"},
    {"id": "timeline_raises", "llm": "LOW", "rule": "LOW", "cv": CV_BENIGN, "timeline": "HIGH", "expected": "HIGH"},
    {"id": "llm_missing_basic_mode", "llm": None, "rule": "MODERATE", "cv": CV_BENIGN, "timeline": None, "expected": "MODERATE"},
    {"id": "llm_garbage_ignored", "llm": "SEVERE", "rule": "LOW", "cv": CV_BENIGN, "timeline": None, "expected": "LOW"},
]

KEYS_OK = '"possible_categories":[{"key":"tinea","likelihood":"higher","why":"Ring edge and itching for 3 weeks."},{"key":"eczema_atopic","likelihood":"possible","why":"Itching fits, but the edge is sharp."}]'
TAIL_OK = '"uncertainty":{"level":"low","reasons":[]},"explanation":"You said it itches and has spread over three weeks. A ring with a scaly edge often fits a fungal infection. Eczema can look similar.","what_would_help":["A clearer photo in daylight"],"self_care_info":["Keep the area clean and dry."],"triage":{"tier":"LOW","advice":"Watch it; see a doctor if it spreads."},"disagreement_with_image_model":false'
VALID = "{" + KEYS_OK + "," + TAIL_OK + "}"
val = lambda vid, text, ok, errs=None, **kw: {"id": vid, "text": text, "expect_ok": ok, "expect_error_prefixes": errs or [], **kw}
validator_cases = [
    val("plain_valid", VALID, True, cv_top1_p=0.8, expect_uncertainty="low", expect_tier="LOW"),
    val("fenced", "```json\n" + VALID + "\n```", True, cv_top1_p=0.8),
    val("leading_trailing_text", "Sure! Here it is:\n" + VALID + "\nHope this helps.", True, cv_top1_p=0.8),
    val("no_json", "I think it is eczema.", False, ["no_json_object"]),
    val("truncated", VALID[:-40], False, ["no_json_object"]),
    val("unknown_key", VALID.replace('"key":"tinea"', '"key":"ringworm"'), False, ["unknown_key"]),
    val("duplicate_key", VALID.replace('"key":"eczema_atopic"', '"key":"tinea"'), False, ["duplicate_key"]),
    val("bad_likelihood", VALID.replace('"higher"', '"certain"'), False, ["bad_likelihood"]),
    val("bad_tier", VALID.replace('"tier":"LOW"', '"tier":"SEVERE"'), False, ["bad_triage"]),
    val("bad_uncertainty", VALID.replace('"level":"low"', '"level":"none"'), False, ["bad_uncertainty"]),
    val("dose_percent", VALID.replace("Keep the area clean and dry.", "Apply 1% cream twice daily."), False, ["dose_pattern"]),
    val("dose_mg", VALID.replace("Keep the area clean and dry.", "Take 250 mg daily."), False, ["dose_pattern"]),
    val("rx_term", VALID.replace("Keep the area clean and dry.", "Ask for terbinafine."), False, ["rx_term"]),
    val("diagnosis_phrase", VALID.replace("A ring with", "You have ringworm. A ring with"), False, ["diagnosis_phrasing"]),
    val("definitely", VALID.replace("Eczema can look similar.", "This is definitely tinea."), False, ["diagnosis_phrasing"]),
    val("too_many_categories", VALID.replace(']', ',{"key":"acne","likelihood":"less_likely","why":"x"},{"key":"psoriasis","likelihood":"less_likely","why":"y"}]', 1), False, ["categories_count"]),
    val("zero_categories", VALID.replace(KEYS_OK, '"possible_categories":[]'), False, ["categories_count"]),
    val("why_too_long", VALID.replace("Ring edge and itching for 3 weeks.", "x" * 201), False, ["why_too_long"]),
    val("explanation_7_sentences", VALID.replace("Eczema can look similar.", "A. B. C. D. E."), False, ["explanation_sentences"]),
    val("low_unc_upgraded_by_cv", VALID, True, cv_top1_p=0.4, expect_uncertainty="moderate"),
    val("low_unc_upgraded_by_disagree", VALID.replace('"disagreement_with_image_model":false', '"disagreement_with_image_model":true'), True, cv_top1_p=0.9, expect_uncertainty="moderate"),
    val("tier_raised_to_rule", VALID, True, cv_top1_p=0.9, rule_tier="HIGH", expect_tier="HIGH"),
    val("percent_word_not_dose", VALID.replace("Keep the area clean and dry.", "Most people improve within two weeks."), True, cv_top1_p=0.9),
]

for name, obj in [("redflag_cases.json", redflag), ("tier_cases.json", tier_cases), ("validator_cases.json", validator_cases)]:
    (HERE / name).write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n")
    print(name, len(obj))
