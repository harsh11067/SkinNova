"""Re-render the frozen llm_test records with the SHIPPED image model's scores, for the shipping-pipeline arm (D).

The frozen arms set (data/llm_eval) carries the scores of the image model current when it was built (CV v1); the app ships
CV v2. Everything else stays identical — image, answers, free text, label — the deterministic rules are re-run on the new
scores and the prompt is rebuilt with prompt_builder.build_analysis (the SFT builder's and the app's code path).
Self-check first: rebuilding every record from its OWN stored scores must reproduce its stored prompt byte for byte.

  python -m ml.eval.rescore_cv --suffix _v2      # → data/llm_eval_cv_v2/llm_test.jsonl (+ images → ../llm_eval/images)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np

from ml.common.paths import LLM_DATA, PROCESSED
from ml.common.schema import load_labels
from ml.eval import redflags as rf
from ml.llm import prompt_builder as pb

LABELS = load_labels()
KEYS = [c["key"] for c in LABELS["classes"]]


def render(rec: dict, cv: list[dict]) -> dict:
    a = rec["meta"]["answers"]
    rule = rf.evaluate(a, cv, LABELS, quality_forced="R8" in rec["meta"]["rules"])
    p = pb.build_analysis(a, cv, rule.tier, rule.messages)
    r = copy.deepcopy(rec)
    r["messages"][1]["content"] = [c if c["type"] != "text" else {"type": "text", "text": p["user"]} for c in r["messages"][1]["content"]]
    t1 = max(cv, key=lambda s: s["p"])
    r["meta"].update({"rule_tier": rule.tier, "rules": rule.fired, "cv_top1": t1["key"], "cv_top1_p": round(t1["p"], 4), "cv": cv})
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--suffix", default="_v2")
    a = ap.parse_args()
    src = LLM_DATA.parent / "llm_eval"; dst = LLM_DATA.parent / f"llm_eval_cv{a.suffix}"
    z = np.load(PROCESSED / f"cv_probs_test{a.suffix}.npz", allow_pickle=True)
    classes = list(z["classes"]); by_name = {Path(str(i)).name: p for i, p in zip(z["img"], z["probs"])}
    recs = [json.loads(l) for l in open(src / "llm_test.jsonl")]
    out, same, n_an = [], 0, 0
    for r in recs:
        if r["task"] not in {"T1", "T9"}:
            out.append(r); continue
        n_an += 1
        own = render(r, r["meta"]["cv"])                       # self-check: the renderer reproduces the frozen prompt
        assert own["messages"][1] == r["messages"][1] and own["meta"]["rule_tier"] == r["meta"]["rule_tier"], r["id"]
        m = dict(zip(classes, by_name[Path(r["image"]).name].tolist()))
        cv = [{"key": k, "p": float(m.get(k, 0.0))} for k in KEYS]
        nr = render(r, cv); nr["meta"]["cv_model"] = f"cv{a.suffix}"
        same += nr["meta"]["cv_top1"] == r["meta"]["cv_top1"]
        out.append(nr)
    dst.mkdir(exist_ok=True)
    with open(dst / "llm_test.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    if not (dst / "images").exists():
        os.symlink(os.path.relpath(src / "images", dst), dst / "images")
    tiers = sum(o["meta"]["rule_tier"] != r["meta"]["rule_tier"] for o, r in zip(out, recs) if r["task"] in {"T1", "T9"})
    print(f"{n_an} analysis records re-rendered (self-check passed on all); image-model top-1 unchanged {same}/{n_an}, "
          f"rule tier changed {tiers}/{n_an} → {dst}")


if __name__ == "__main__":
    main()
