"""Deterministic safety rules — Python mirror of android `safety/RedFlagRules.kt` + `TierResolver.kt`.

contracts.md §4. Both implementations run the same fixtures (tests/fixtures/redflag_cases.json,
tier_cases.json). Change order: contracts.md → fixtures → Kotlin + Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ml.common.schema import TIER_RANK, max_tier

LESION_CHANGES = {"growing", "changing_color", "changing_shape"}
LONG_DURATIONS = {"1_4w", "1_6m", "gt_6m"}
CLASS_FLOOR_P = 0.15
SUSPICIOUS_P = 0.15


@dataclass
class RuleResult:
    tier: str = "LOW"
    fired: list[str] = field(default_factory=list)       # e.g. ["R1", "R5"]
    messages: list[str] = field(default_factory=list)    # strings.xml keys: rf_r1 …
    force_uncertainty_high: bool = False

    def raise_to(self, tier: str, rule: str, msg_key: str | None = None):
        self.tier = max_tier(self.tier, tier)
        if rule not in self.fired:
            self.fired.append(rule)
        if msg_key and msg_key not in self.messages:
            self.messages.append(msg_key)


def _top3(cv: list[dict]) -> list[dict]:
    return sorted(cv, key=lambda s: -s["p"])[:3]


def evaluate(answers: dict, cv: list[dict], labels: dict, *, quality_forced: bool = False,
             timeline: dict | None = None) -> RuleResult:
    """answers: QuestionnaireAnswers dict; cv: [{"key","p"}] calibrated probs (any length)."""
    meta = {c["key"]: c for c in labels["classes"]}
    top3 = _top3(cv)
    r = RuleResult()
    a = answers
    # R1 possible infection
    if a["fever_or_unwell"] and (a["pain"] >= 2 or a["changing"] == "spreading"):
        r.raise_to("URGENT", "R1", "rf_r1")
    # R2 changing lesion
    if a["changing"] in LESION_CHANGES and any(meta[s["key"]]["lesion_type"] for s in top3 if s["key"] in meta):
        r.raise_to("HIGH", "R2", "rf_r2")
    # R3 bleeding/crusting that persists
    if a["bleeding_or_crusting"] and a["duration"] in LONG_DURATIONS:
        r.raise_to("HIGH", "R3", "rf_r3")
    # R4 suspicious lesion probability
    sus = next((s["p"] for s in cv if s["key"] == "suspicious_lesion"), 0.0)
    if sus >= SUSPICIOUS_P:
        r.raise_to("HIGH", "R4", "rf_r4")
    # R5 child
    if a["age_band"] == "lt_12":
        r.raise_to("MODERATE", "R5", "rf_r5")
    # R6 painful face/groin
    if a["body_site"] in {"groin", "face"} and a["pain"] >= 2:
        r.raise_to("MODERATE", "R6", "rf_r6")
    # R7 household spread
    if a["others_affected"] and a["itch"] >= 2:
        r.raise_to("MODERATE", "R7", "rf_r7")
    # R8 forced past quality gate
    if quality_forced:
        r.force_uncertainty_high = True
        if "R8" not in r.fired:
            r.fired.append("R8")
            r.messages.append("rf_r8")
    if timeline:
        apply_timeline(r, timeline)
    return r


def apply_timeline(r: RuleResult, t: dict) -> RuleResult:
    """T1–T3. confidence=low disables T1/T2 (contracts §8); T3 still allowed."""
    if not t.get("align_ok"):
        return r
    conf_ok = t.get("confidence") == "ok"
    lesion = bool(t.get("lesion_type"))
    area = t.get("area_ratio")
    contrast = t.get("contrast_delta")
    if conf_ok and lesion and t.get("coin_in_both") and area is not None and area >= 1.25:
        r.raise_to("HIGH", "T1", "rf_t1")
    if conf_ok and lesion and contrast is not None and contrast >= max(5.0, 2 * t.get("noise_contrast", 0.0)):
        r.raise_to("HIGH", "T2", "rf_t2")
    if (not lesion) and area is not None and area >= 1.5:
        r.raise_to("MODERATE", "T3", "rf_t3")
    return r


def class_floor(cv: list[dict], labels: dict) -> str:
    """tier_floor of any class in top-3 with calibrated p ≥ 0.15."""
    meta = {c["key"]: c for c in labels["classes"]}
    floor = "LOW"
    for s in _top3(cv):
        if s["p"] >= CLASS_FLOOR_P and s["key"] in meta:
            floor = max_tier(floor, meta[s["key"]]["tier_floor"])
    return floor


def resolve_tier(llm_tier: str | None, rule_tier: str, floor: str, timeline_tier: str | None = None) -> str:
    """TierResolver: monotone max. The LLM can raise but never lower."""
    tiers = [rule_tier, floor] + [t for t in (llm_tier, timeline_tier) if t in TIER_RANK]
    return max_tier(*tiers)
