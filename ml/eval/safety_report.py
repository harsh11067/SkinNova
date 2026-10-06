"""Safety summary → reports/safety.json (plan §2 S2–S4, test.md §7).

S2  suspicious-lesion recall (CV + rules): a true suspicious lesion is caught when the image-model-driven rules alone
    (R4: p(suspicious) ≥ threshold; class floor: suspicious in top-3 with p ≥ 0.15) lift the tier to ≥ HIGH. Answer-driven
    rules (R2 changing, R3 bleeding) can only add to this, so it is a lower bound. Also: benign lesions flagged HIGH (cost).
S3  red-flag rule recall on tests/fixtures/redflag_cases.json.
S4  under-triage of HIGH/URGENT references in the arms run (reports/llm_arms.json, arm D) when present.

  python -m ml.eval.safety_report --probs-tag ""        # cv_probs_<split>.npz from ml/cv/predict.py (v2: --probs-tag _v2)
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ml.common.paths import FIXTURES, PROCESSED, REPORTS, dataset_rev, report_meta
from ml.common.schema import load_labels
from ml.eval.redflags import SUSPICIOUS_P, class_floor, evaluate, max_tier

RANK = {"LOW": 0, "MODERATE": 1, "HIGH": 2, "URGENT": 3}


def s2(split: str, tag: str, labels: dict) -> dict | None:
    f = PROCESSED / f"cv_probs_{split}{tag}.npz"
    if not f.exists():
        return None
    z = np.load(f, allow_pickle=True)
    classes = [str(c) for c in z["classes"]]
    caught = flagged_benign = n_sus = n_ben = 0
    for y, p in zip(z["labels"], z["probs"]):
        cv = [{"key": k, "p": float(v)} for k, v in zip(classes, p)]
        sus_p = next(c["p"] for c in cv if c["key"] == "suspicious_lesion")
        tier = max_tier("HIGH" if sus_p >= SUSPICIOUS_P else "LOW", class_floor(cv, labels))
        if y == "suspicious_lesion":
            n_sus += 1; caught += RANK[tier] >= 2
        elif y == "benign_lesion":
            n_ben += 1; flagged_benign += RANK[tier] >= 2
    return {"n_suspicious": n_sus, "recall_cv_rules": round(caught / n_sus, 4) if n_sus else None,
            "n_benign": n_ben, "benign_flagged_high": round(flagged_benign / n_ben, 4) if n_ben else None}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--probs-tag", default="")
    a = ap.parse_args()
    labels = load_labels()
    rep = {**report_meta(dataset_rev=dataset_rev()), "probs_tag": a.probs_tag, "suspicious_threshold": SUSPICIOUS_P}
    rep["S2_suspicious_recall"] = {s: s2(s, a.probs_tag, labels) for s in ["val", "test", "external_test"]}
    rep["S2_pass_test"] = bool((rep["S2_suspicious_recall"].get("test") or {}).get("recall_cv_rules", 0) >= 0.90)
    cases = json.loads((FIXTURES / "redflag_cases.json").read_text())
    ok = 0
    for c in cases:
        r = evaluate(c["answers"], c["cv"], labels, quality_forced=c.get("quality_forced", False), timeline=c.get("timeline"))
        e = c["expected"]
        ok += (r.tier == e["tier"] and sorted(r.fired) == sorted(e["fired"]) and r.force_uncertainty_high == e["force_uncertainty_high"])
    rep["S3_redflag_cases"] = {"n": len(cases), "correct": ok, "pass": ok == len(cases)}
    arms = REPORTS / "llm_arms.json"
    if arms.exists():
        d = json.loads(arms.read_text()).get("arms", {}).get("D", {})
        rep["S4_under_triage_high_urgent"] = {"value": d.get("S9_under_triage_high_urgent"), "n_reference": d.get("n_high_urgent_reference"),
                                              "pass": d.get("S9_under_triage_high_urgent") == 0}
    (REPORTS / "safety.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k.startswith("S")}, indent=1))


if __name__ == "__main__":
    main()
