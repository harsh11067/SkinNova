"""S1/S2/S3: shared fixtures — the same JSON files run in JUnit (android/app/src/test)."""
import json
import random

import pytest

from ml.common.paths import FIXTURES
from ml.common.schema import TIERS, TIER_RANK, load_labels
from ml.eval import redflags as rf
from ml.llm.validate import validate

LABELS = load_labels()
KEYS = [c["key"] for c in LABELS["classes"]]
RF = json.loads((FIXTURES / "redflag_cases.json").read_text())
TC = json.loads((FIXTURES / "tier_cases.json").read_text())
VC = json.loads((FIXTURES / "validator_cases.json").read_text())


def test_fixture_size():
    assert len(RF) >= 60


@pytest.mark.parametrize("case", RF, ids=[c["id"] for c in RF])
def test_redflag(case):
    r = rf.evaluate(case["answers"], case["cv"], LABELS, quality_forced=case["quality_forced"], timeline=case["timeline"])
    e = case["expected"]
    assert r.tier == e["tier"]
    assert sorted(r.fired) == sorted(e["fired"])
    assert sorted(r.messages) == sorted(e["messages"])
    assert r.force_uncertainty_high == e["force_uncertainty_high"]


@pytest.mark.parametrize("case", TC, ids=[c["id"] for c in TC])
def test_tier_resolver(case):
    floor = rf.class_floor(case["cv"], LABELS)
    assert rf.resolve_tier(case["llm"], case["rule"], floor, case["timeline"]) == case["expected"]


def test_tier_monotone_property():
    """S2: 10,000 random inputs → final ≥ max(rule, floor) and ≥ every rule-fired tier."""
    rnd = random.Random(3407)
    from ml.common import schema as s
    for _ in range(10_000):
        a = dict(body_site=rnd.choice(s.BODY_SITES), duration=rnd.choice(s.DURATIONS), itch=rnd.randint(0, 3),
                 pain=rnd.randint(0, 3), changing=rnd.choice(s.CHANGING), bleeding_or_crusting=rnd.random() < .3,
                 fever_or_unwell=rnd.random() < .2, others_affected=rnd.random() < .2,
                 new_product_or_exposure=rnd.random() < .3, age_band=rnd.choice(s.AGE_BANDS))
        w = [rnd.random() for _ in KEYS]; tot = sum(w)
        cv = [{"key": k, "p": x / tot} for k, x in zip(KEYS, w)]
        r = rf.evaluate(a, cv, LABELS)
        floor = rf.class_floor(cv, LABELS)
        llm = rnd.choice(TIERS + [None])
        final = rf.resolve_tier(llm, r.tier, floor)
        assert TIER_RANK[final] >= max(TIER_RANK[r.tier], TIER_RANK[floor])
        if llm:
            assert TIER_RANK[final] >= TIER_RANK[llm]


@pytest.mark.parametrize("case", VC, ids=[c["id"] for c in VC])
def test_validator(case):
    res = validate(case["text"], KEYS, cv_top1_p=case.get("cv_top1_p"), rule_tier=case.get("rule_tier", "LOW"))
    assert res.ok == case["expect_ok"], res.errors
    for pre in case["expect_error_prefixes"]:
        assert any(e.startswith(pre) for e in res.errors), (pre, res.errors)
    if res.ok and "expect_uncertainty" in case:
        assert res.data["uncertainty"]["level"] == case["expect_uncertainty"]
    if res.ok and "expect_tier" in case:
        assert res.data["triage"]["tier"] == case["expect_tier"]
