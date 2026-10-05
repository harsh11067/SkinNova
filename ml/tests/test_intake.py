"""USP-2 IntakeValidator (contracts §9) — shared fixture with JUnit IntakeValidatorTest."""
import json

import pytest

from ml.common.paths import FIXTURES
from ml.voice.intake import validate_intake

CASES = json.loads((FIXTURES / "intake_cases.json").read_text())


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_intake(case):
    r = validate_intake(case["text"], case["transcript"])
    assert r["ok"] == case["expect_ok"]
    kept = {k: v["value"] for k, v in r["fields"].items() if v is not None}
    assert kept == case["expect_fields"]
    assert r["dropped"] == case["expect_dropped"]
    if "expect_notes_len" in case:
        assert len(r["unparsed_notes"]) == case["expect_notes_len"]


def test_never_fills_unstated():
    """V4 property: no field is ever non-null unless its evidence is in the transcript."""
    for c in CASES:
        r = validate_intake(c["text"], c["transcript"])
        for f, v in r["fields"].items():
            if v is not None:
                from ml.voice.intake import evidence_ok
                assert evidence_ok(v["evidence"], c["transcript"])
