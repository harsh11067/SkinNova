"""IntakeValidator (contracts §9) — Python twin of android `ml/IntakeValidator.kt`. Fixture: tests/fixtures/intake_cases.json.

A field survives only if: its value is valid for the field AND its evidence quote matches the transcript
(normalised: Unicode NFC, lowercase, whitespace collapsed; exact substring, else best same-length window with
Levenshtein similarity ≥ 0.8). Failing fields become null — never defaulted, never guessed.
"""
from __future__ import annotations

import json
import re
import unicodedata

from ml.common.schema import AGE_BANDS, BODY_SITES, CHANGING, DURATIONS
from ml.llm.validate import extract_json

FIELDS = {
    "body_site": ("enum", BODY_SITES), "duration": ("enum", DURATIONS), "changing": ("enum", CHANGING),
    "age_band": ("enum", AGE_BANDS), "itch": ("scale", None), "pain": ("scale", None),
    "bleeding_or_crusting": ("bool", None), "fever_or_unwell": ("bool", None),
    "others_affected": ("bool", None), "new_product_or_exposure": ("bool", None),
}
MIN_SIM = 0.8


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s).lower()).strip()


def lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def evidence_ok(evidence: str, transcript: str) -> bool:
    e, t = norm(evidence), norm(transcript)
    if len(e) < 2:
        return False
    if e in t:
        return True
    n = len(e)
    if n > len(t):
        return 1 - lev(e, t) / max(n, len(t)) >= MIN_SIM
    return max(1 - lev(e, t[i:i + n]) / n for i in range(len(t) - n + 1)) >= MIN_SIM


def value_ok(field: str, v) -> bool:
    kind, allowed = FIELDS[field]
    if kind == "enum":
        return isinstance(v, str) and v in allowed
    if kind == "scale":
        return isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 3
    return isinstance(v, bool)


def validate_intake(text: str, transcript: str) -> dict:
    """→ {"ok", "language", "fields": {f: {value, evidence} | None}, "dropped": {f: reason}, "unparsed_notes"}"""
    raw = extract_json(text)
    out = {"ok": False, "language": None, "fields": {f: None for f in FIELDS}, "dropped": {}, "unparsed_notes": ""}
    if raw is None:
        out["dropped"]["_all"] = "no_json_object"; return out
    try:
        d = json.loads(raw)
    except json.JSONDecodeError:
        out["dropped"]["_all"] = "json_parse"; return out
    if not isinstance(d, dict) or not isinstance(d.get("fields"), dict):
        out["dropped"]["_all"] = "bad_shape"; return out
    out["ok"] = True
    out["language"] = d.get("language") if isinstance(d.get("language"), str) else None
    notes = d.get("unparsed_notes")
    out["unparsed_notes"] = notes[:200] if isinstance(notes, str) else ""
    for f, item in d["fields"].items():
        if f not in FIELDS:
            out["dropped"][f] = "unknown_field"; continue
        if item is None:
            continue
        if not isinstance(item, dict) or "value" not in item or not isinstance(item.get("evidence"), str):
            out["dropped"][f] = "bad_item"; continue
        if not value_ok(f, item["value"]):
            out["dropped"][f] = "bad_value"; continue
        if not evidence_ok(item["evidence"], transcript):
            out["dropped"][f] = "evidence_not_in_transcript"; continue
        out["fields"][f] = {"value": item["value"], "evidence": item["evidence"]}
    return out
