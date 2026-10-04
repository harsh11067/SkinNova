"""LLM output validator + content guards — Python mirror of android `ml/OutputParser.kt` + `safety/ContentGuards.kt`.

contracts.md §3 rules 1–7. Shared fixture: tests/fixtures/validator_cases.json.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ml.common.schema import LIKELIHOODS, TIERS, UNCERTAINTY

REPO = Path(__file__).resolve().parents[2]
RX_TERMS_FILE = REPO / "ml" / "llm" / "safety" / "rx_terms.txt"

DOSE_RE = re.compile(r"\b\d+(\.\d+)?\s?(mg|mcg|g|ml|%|IU)(?![A-Za-z])", re.IGNORECASE)
DIAGNOSIS_RE = re.compile(r"\b(you have|you are diagnosed|this is definitely)\b", re.IGNORECASE)
SENTENCE_RE = re.compile(r"[^.!?]+[.!?]+|[^.!?]+$")


def load_rx_terms(path: Path = RX_TERMS_FILE) -> list[str]:
    return [l.strip().lower() for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]


@dataclass
class ValidationResult:
    ok: bool
    data: dict | None = None
    errors: list[str] = field(default_factory=list)


def extract_json(text: str) -> str | None:
    """Rule 1: strip fences / leading text; return first balanced top-level JSON object."""
    t = text.replace("```json", "```")
    start = t.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(t)):
            ch = t[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return t[start:i + 1]
        # Unbalanced from the first "{" (truncated output): never fall back to an inner object.
        return None
    return None


def count_sentences(text: str) -> int:
    return len([s for s in SENTENCE_RE.findall(text.strip()) if s.strip()])


def guard_text(text: str, rx_terms: list[str]) -> list[str]:
    errs = []
    if DOSE_RE.search(text):
        errs.append(f"dose_pattern:{DOSE_RE.search(text).group(0)}")
    low = text.lower()
    for term in rx_terms:
        if re.search(r"\b" + re.escape(term) + r"\b", low):
            errs.append(f"rx_term:{term}")
    if DIAGNOSIS_RE.search(text):
        errs.append(f"diagnosis_phrasing:{DIAGNOSIS_RE.search(text).group(0).lower()}")
    return errs


def validate(text: str, label_keys: list[str], *, cv_top1_p: float | None = None,
             rule_tier: str = "LOW", rx_terms: list[str] | None = None) -> ValidationResult:
    rx_terms = load_rx_terms() if rx_terms is None else rx_terms
    raw = extract_json(text)
    if raw is None:
        return ValidationResult(False, errors=["no_json_object"])
    try:
        d = json.loads(raw)
    except json.JSONDecodeError as e:
        return ValidationResult(False, errors=[f"json_parse:{e.msg}"])
    if not isinstance(d, dict):
        return ValidationResult(False, errors=["not_object"])
    errs: list[str] = []
    # Rule 2 categories
    cats = d.get("possible_categories")
    if not isinstance(cats, list) or not 1 <= len(cats) <= 3:
        errs.append("categories_count")
        cats = cats if isinstance(cats, list) else []
    seen = set()
    for c in cats:
        if not isinstance(c, dict):
            errs.append("category_not_object"); continue
        k = c.get("key")
        if k not in label_keys:
            errs.append(f"unknown_key:{k}")
        if k in seen:
            errs.append(f"duplicate_key:{k}")
        seen.add(k)
        if c.get("likelihood") not in LIKELIHOODS:
            errs.append(f"bad_likelihood:{c.get('likelihood')}")
        why = c.get("why")
        if not isinstance(why, str) or not why.strip():
            errs.append("missing_why")
        elif len(why) > 200:
            errs.append("why_too_long")
    # Rule 3 uncertainty
    unc = d.get("uncertainty")
    if not isinstance(unc, dict) or unc.get("level") not in UNCERTAINTY:
        errs.append("bad_uncertainty")
    elif not isinstance(unc.get("reasons", []), list):
        errs.append("bad_uncertainty_reasons")
    # Rule 4 triage
    tri = d.get("triage")
    if not isinstance(tri, dict) or tri.get("tier") not in TIERS or not isinstance(tri.get("advice"), str):
        errs.append("bad_triage")
    # Rule 6 explanation
    exp = d.get("explanation")
    if not isinstance(exp, str) or not exp.strip():
        errs.append("missing_explanation")
    else:
        n = count_sentences(exp)
        if not 1 <= n <= 6:
            errs.append(f"explanation_sentences:{n}")
    for key in ("what_would_help", "self_care_info"):
        v = d.get(key, [])
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
            errs.append(f"bad_{key}")
    if not isinstance(d.get("disagreement_with_image_model", False), bool):
        errs.append("bad_disagreement")
    # Rule 5 content guards
    guarded = [exp if isinstance(exp, str) else ""]
    guarded += [x for x in d.get("self_care_info", []) if isinstance(x, str)]
    guarded += [c.get("why", "") for c in cats if isinstance(c, dict) and isinstance(c.get("why"), str)]
    for t in guarded:
        errs.extend(guard_text(t, rx_terms))
    if errs:
        return ValidationResult(False, data=d, errors=errs)
    # Normalisations (not errors): rule 3 upgrade, rule 4 tier never below rule tier
    disagree = d.get("disagreement_with_image_model", False)
    if d["uncertainty"]["level"] == "low" and ((cv_top1_p is not None and cv_top1_p < 0.5) or disagree):
        d["uncertainty"]["level"] = "moderate"
    from ml.common.schema import max_tier
    d["triage"]["tier"] = max_tier(d["triage"]["tier"], rule_tier)
    d.setdefault("what_would_help", [])
    d.setdefault("self_care_info", [])
    d.setdefault("disagreement_with_image_model", False)
    return ValidationResult(True, data=d)
