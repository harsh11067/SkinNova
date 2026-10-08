"""voice_test kit: card integrity, schedule coverage, and the V2–V6 scorer (ml/eval/eval_voice.py) on hand-made cases."""
import json

import pytest

from ml.common.schema import AGE_BANDS, BODY_SITES, CHANGING, DURATIONS
from ml.eval.eval_voice import cer, coverage_problems, gold_for, load_cards, norm_text, score_clip, script_of, summarize
from ml.voice.intake import FIELDS
from ml.voice.make_sheet import schedule

CARDS = load_cards()
ENUMS = {"body_site": BODY_SITES, "duration": DURATIONS, "changing": CHANGING, "age_band": AGE_BANDS}


def test_cards_valid():
    assert len(CARDS) == 20
    for c in CARDS.values():
        for k, v in c["fields"].items():
            assert k in FIELDS, (c["id"], k)
            kind = FIELDS[k][0]
            if kind == "enum":
                assert v in ENUMS[k], (c["id"], k, v)
            elif kind == "scale":
                assert isinstance(v, int) and 0 <= v <= 3
            else:
                assert isinstance(v, bool)
        assert set(c["negated"]) <= set(c["fields"]), c["id"]
        assert all(c[lang].strip() for lang in ("en", "hi", "hinglish"))
        assert script_of(c["hi"]) == "deva" and script_of(c["hinglish"]) == "latn"
    # every field and every age band is exercised; negation subset is large enough to measure V5
    assert set().union(*(c["fields"] for c in CARDS.values())) == set(FIELDS)
    assert {c["fields"]["age_band"] for c in CARDS.values() if "age_band" in c["fields"]} == set(AGE_BANDS)
    assert sum(len(c["negated"]) for c in CARDS.values()) >= 10
    assert any(not c["fields"] for c in CARDS.values())   # a no-facts card: anything extracted is a hallucination


def test_schedule_covers_every_cell():
    rows = schedule(6)
    assert len(rows) == 120 and coverage_problems(rows) == []
    for s in {r["speaker"] for r in rows}:
        mine = [r for r in rows if r["speaker"] == s]
        assert {(r["lang"], r["noise"]) for r in mine} == {(l, n) for l in ("en", "hi", "hinglish") for n in ("quiet", "noisy")}
        assert sorted(r["card_id"] for r in mine) == sorted(CARDS)


def test_text_metrics():
    assert norm_text("Teen  hafte se, haath par!") == "teen hafte se haath par"
    assert norm_text("खुजली  होती है।") == "खुजली होती है"     # danda removed, vowel signs kept
    assert cer("teen hafte se", "teen hafte se") == 0
    assert cer("teen hafte", "teen hafte se") == pytest.approx(3 / 13)


def _row(card, lang="hinglish", omitted="", ref=None):
    return {"clip": f"s1_{card}.wav", "speaker": "s1", "lang": lang, "noise": "quiet", "card_id": card, "mode": "free",
            "omitted": omitted, "reference_transcript": CARDS[card][lang] if ref is None else ref}


def _ext(fields):
    return json.dumps({"language": "hi", "fields": {k: {"value": v, "evidence": e} for k, (v, e) in fields.items()}, "unparsed_notes": ""})


def test_score_clip_correct_wrong_hallucinated():
    c = CARDS["C01"]; t = c["hinglish"]
    ext = _ext({"body_site": ("hand", "haath par"), "duration": ("1_4w", "teen hafte se"), "itch": (2, "khujli bahut zyada"),
                "others_affected": (True, "bhai ko bhi"), "fever_or_unwell": (True, "bhai ko bhi")})
    r = score_clip(_row("C01"), c, gold_for(_row("C01"), CARDS), t, ext)
    # fever_or_unwell quotes "bhai ko bhi" (brother has it): a real quote on the wrong field. Since the topic check
    # (decisions 2026-10-07) the validator drops it, so it is no longer a hallucinated field that reaches the user.
    assert r["correct"] == 3 and r["wrong"] == ["itch"] and r["hallucinated"] == [] and r["cer"] == 0


def test_evidence_not_in_transcript_is_dropped_not_counted():
    c = CARDS["C20"]; t = c["en"]
    r = score_clip(_row("C20", "en"), c, {}, t, _ext({"fever_or_unwell": (True, "I have a fever")}))
    assert r["hallucinated"] == []           # the validator removed it: the evidence quote is not in the transcript
    r = score_clip(_row("C20", "en"), c, {}, t, _ext({"body_site": ("other", "this spot")}))
    assert r["hallucinated"] == ["body_site"]


def test_omitted_facts_leave_gold_and_bad_names_fail():
    assert "age_band" not in gold_for(_row("C03", omitted="age_band"), CARDS)
    with pytest.raises(ValueError):
        gold_for(_row("C03", omitted="itch"), CARDS)


def test_summary_gates():
    c = CARDS["C10"]; t = c["en"]
    good = _ext({"body_site": ("nails", "toenail"), "duration": ("1_6m", "four months"), "changing": ("no", "hasn't changed"),
                 "pain": (0, "No pain"), "bleeding_or_crusting": (False, "no bleeding")})
    rows = [score_clip(_row("C10", "en"), c, gold_for(_row("C10", "en"), CARDS), t, good)]
    s = summarize(rows)
    assert s["V3_field_acc"]["en"]["field_acc"] == 1.0 and s["V4_pass"] and s["V5_negation_acc"] == 1.0 and s["V6_duration_acc"] == 1.0
    assert not s["V3_pass"]                  # hi / hinglish have no clips → cannot pass
    t2 = "lagbhag chaar mahine se naakhun peela hai"
    rows.append(score_clip(_row("C10", ref=t2), c, gold_for(_row("C10"), CARDS), "लगभग चार महीने से नाखून पीला है", "{}"))
    assert rows[-1]["script_mismatch"] and rows[-1]["cer"] is None
