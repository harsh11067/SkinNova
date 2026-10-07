"""CV v3 adoption check — rule pre-registered in docs/decisions.md (2026-10-07) before v3 was trained. VAL only.

v3 = v2 recipe + Shades-of-Gray colour constancy. ADOPT iff (1) SCIN-val top-3 ≥ 0.826, (2) full-val macro-F1 ≥ 0.679,
(3) suspicious-lesion recall at p ≥ 0.15 ≥ 0.907 — calibrated probabilities (cv_probs_val_v2/_v3.npz from ml.cv.predict).
v2's numbers are recomputed here with the same code for reference. Writes reports/cv_v3_adoption.json.

  python -m ml.cv.adopt_v3
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ml.common.paths import PROCESSED, REPORTS, SPLITS, report_meta

RULE = {"scin_val_top3_min": 0.826, "val_macro_f1_min": 0.679, "suspicious_recall_p15_min": 0.907}


def metrics(tag: str) -> dict:
    z = np.load(PROCESSED / f"cv_probs_val{tag}.npz", allow_pickle=True)
    cls = list(z["classes"]); P = z["probs"]; y = np.array([cls.index(l) for l in z["labels"]])
    src = pd.read_csv(SPLITS / "val.csv").set_index("img").source.reindex([str(i) for i in z["img"]]).values
    pred = P.argmax(1); top3 = (np.argsort(-P, 1)[:, :3] == y[:, None]).any(1)
    s = src == "scin"; sus = cls.index("suspicious_lesion")
    return {"n": int(len(y)), "n_scin": int(s.sum()), "val_top1": float(np.mean(pred == y)), "val_top3": float(top3.mean()),
            "val_macro_f1": float(f1_score(y, pred, labels=sorted(set(y.tolist())), average="macro", zero_division=0)),
            "scin_val_top1": float(np.mean(pred[s] == y[s])), "scin_val_top3": float(top3[s].mean()),
            "suspicious_recall_p15": float((P[y == sus, sus] >= 0.15).mean())}


def main():
    v2, v3 = metrics("_v2"), metrics("_v3")
    checks = {"scin_val_top3": v3["scin_val_top3"] >= RULE["scin_val_top3_min"],
              "val_macro_f1": v3["val_macro_f1"] >= RULE["val_macro_f1_min"],
              "suspicious_recall_p15": v3["suspicious_recall_p15"] >= RULE["suspicious_recall_p15_min"]}
    rep = {**report_meta(), "rule": RULE, "v2": v2, "v3": v3, "checks": checks, "decision": "ADOPT_V3" if all(checks.values()) else "KEEP_V2",
           "note": "val only; test/external are evaluated once and only if adopted"}
    (REPORTS / "cv_v3_adoption.json").write_text(json.dumps(rep, indent=1))
    for k in ("val_top1", "val_top3", "val_macro_f1", "scin_val_top1", "scin_val_top3", "suspicious_recall_p15"):
        print(f"{k:24} v2 {v2[k]:.3f}  v3 {v3[k]:.3f}")
    print(checks); print(rep["decision"])


if __name__ == "__main__":
    main()
