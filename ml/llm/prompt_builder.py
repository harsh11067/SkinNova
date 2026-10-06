"""PromptBuilder — Python mirror of android `ml/PromptBuilder.kt`. Same template files, same rendering.

Rendering rules (Kotlin must match byte-for-byte; golden files in tests/fixtures/prompt_golden/):
- Template header line "# vN" is stripped; its version is returned for provenance.
- JSON values are compact (separators ",", ":"), keys in insertion order, non-ASCII kept.
- CV top-3 probabilities rounded to 2 decimals.
- Condition notes: only the cards for the CV top-3 keys, fields summary/typical_features/distinguishing_from/see_doctor_if.
- free_text: "<<<" / ">>>" removed from user text so it cannot close the delimiter.
"""
from __future__ import annotations

import json
from pathlib import Path

from ml.common.paths import CARDS, PROMPTS

SCHEMA_COMPACT = ('{"possible_categories":[{"key":"<label key>","likelihood":"higher|possible|less_likely","why":"<=200 chars"}] (1-3),'
                  '"uncertainty":{"level":"low|moderate|high","reasons":["..."]},"explanation":"2-5 sentences citing an answer",'
                  '"what_would_help":["..."],"self_care_info":["general, non-prescription"],'
                  '"triage":{"tier":"LOW|MODERATE|HIGH|URGENT","advice":"..."},"disagreement_with_image_model":false}')
CARD_FIELDS = ["summary", "typical_features", "distinguishing_from", "see_doctor_if"]
ANSWER_FIELDS = ["body_site", "duration", "itch", "pain", "changing", "bleeding_or_crusting", "fever_or_unwell",
                 "others_affected", "new_product_or_exposure", "age_band", "skin_tone"]


def cj(o) -> str:
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False)


def load_template(name: str, root: Path = PROMPTS) -> tuple[str, str]:
    text = (root / name).read_text(encoding="utf-8")
    first, _, rest = text.partition("\n")
    assert first.startswith("# v"), f"{name} missing version header"
    return first[2:].strip(), rest.rstrip("\n")


def load_cards(path: Path = CARDS) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["cards"]


def sanitize_free_text(s: str) -> str:
    return (s or "").replace("<<<", "").replace(">>>", "")[:200]


def top3(cv: list[dict]) -> list[dict]:
    return [{"key": s["key"], "p": round(float(s["p"]), 2)} for s in sorted(cv, key=lambda s: -s["p"])[:3]]


def build_analysis(answers: dict, cv: list[dict], rule_tier: str, rule_messages: list[str],
                   cards: dict | None = None, root: Path = PROMPTS) -> dict:
    cards = load_cards() if cards is None else cards
    v_sys, system = load_template("analyze_system.txt", root)
    v_usr, user_t = load_template("analyze_user.txt", root)
    t3 = top3(cv)
    notes = {s["key"]: {f: cards[s["key"]][f] for f in CARD_FIELDS} for s in t3 if s["key"] in cards}
    ans = {k: answers[k] for k in ANSWER_FIELDS if k in answers}
    user = (user_t.replace("{cv_top3_json}", cj(t3))
            .replace("{rule_tier}", rule_tier)
            .replace("{rule_messages_json}", cj(rule_messages))
            .replace("{answers_json}", cj(ans))
            .replace("{free_text}", sanitize_free_text(answers.get("free_text", "")))
            .replace("{cards_json}", cj(notes))
            .replace("{schema_compact}", SCHEMA_COMPACT))
    assert v_sys == v_usr
    return {"system": system, "user": user, "prompt_version": v_sys}


def build_repair(previous: str, errors: list[str], root: Path = PROMPTS) -> str:
    _, t = load_template("repair.txt", root)
    return (t.replace("{errors}", "; ".join(errors)).replace("{schema_compact}", SCHEMA_COMPACT)
            .replace("{previous}", previous.replace("<<<", "").replace(">>>", "")))


def build_extract(transcript: str, root: Path = PROMPTS) -> dict:
    v, system = load_template("extract_system.txt", root)
    _, user_t = load_template("extract_user.txt", root)
    return {"system": system, "user": user_t.replace("{transcript}", transcript.replace("<<<", "").replace(">>>", "")),
            "prompt_version": v}


def build_narrate(metrics: dict, cv_before: list[dict], cv_after: list[dict], timeline_tier: str, baseline_date: str,
                  root: Path = PROMPTS) -> dict:
    v, system = load_template("narrate_system.txt", root)
    _, user_t = load_template("narrate_user.txt", root)
    user = (user_t.replace("{baseline_date}", baseline_date).replace("{metrics_json}", cj(metrics))
            .replace("{cv_before_json}", cj(top3(cv_before))).replace("{cv_after_json}", cj(top3(cv_after)))
            .replace("{timeline_tier}", timeline_tier))
    return {"system": system, "user": user, "prompt_version": v}


def build_translate(source: dict, target_language: str, root: Path = PROMPTS) -> dict:
    v, system = load_template("translate_system.txt", root)
    _, user_t = load_template("translate_user.txt", root)
    return {"system": system.replace("{target_language}", target_language),
            "user": user_t.replace("{source_json}", cj(source)), "prompt_version": v}


def runtime_system_message(system: str) -> str:
    """System message for LiteRT-LM's Python/C API in the same form the app sends (Kotlin `Contents.of(text)` → a one-part
    JSON list; the C API parses JSON when it can). The Gemma 4 template renders a list as "<text> <turn|>" and a plain
    string as "<text><turn|>", so desktop evals must use this to see exactly the phone's prompt."""
    return json.dumps([{"type": "text", "text": system}], ensure_ascii=False)
