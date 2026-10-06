"""CV v2 adoption check (rule fixed in docs/decisions.md before v2 results existed). Uses VAL only.

  (a) on the full v2 val set, macro-F1(v2) ≥ macro-F1(v1)        (both models scored on the same images)
  (b) on the frozen v1-val images, macro-F1(v2) ≥ 0.723          (v1's 95 % CI lower bound on those images)
Needs cv_probs_val_v1on2.npz (v1 checkpoint on the v2 val split) and cv_probs_val_v2.npz (ml/cv/predict.py).
Writes reports/cv_v2_adoption.json and prints ADOPT or KEEP_V1.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ml.common.paths import PROCESSED, REPORTS, SPLITS, dataset_rev, report_meta

V1_CI_LOW = 0.723


def f1(z, mask=None) -> float:
    classes = [str(c) for c in z["classes"]]
    y = np.array([classes.index(str(l)) for l in z["labels"]]); p = z["probs"].argmax(1)
    if mask is not None:
        y, p = y[mask], p[mask]
    return float(f1_score(y, p, labels=sorted(set(y.tolist())), average="macro", zero_division=0))


def main():
    z1, z2 = (np.load(PROCESSED / f"cv_probs_val_{t}.npz", allow_pickle=True) for t in ("v1on2", "v2"))
    assert list(z1["img"]) == list(z2["img"]), "both models must be scored on the same val images"
    frozen = pd.read_csv(SPLITS / "frozen_v1.csv"); v1val = set(frozen[frozen.split == "val"].img)
    m = np.array([i in v1val for i in z2["img"]])
    rep = {**report_meta(dataset_rev=dataset_rev()), "n_val": int(len(m)), "n_v1_val": int(m.sum()),
           "full_val_f1": {"v1": round(f1(z1), 4), "v2": round(f1(z2), 4)},
           "v1_val_images_f1": {"v1": round(f1(z1, m), 4), "v2": round(f1(z2, m), 4)},
           "scin_val_images_f1": {"v1": round(f1(z1, ~m), 4), "v2": round(f1(z2, ~m), 4)}}
    rep["rule_a"] = rep["full_val_f1"]["v2"] >= rep["full_val_f1"]["v1"]
    rep["rule_b"] = rep["v1_val_images_f1"]["v2"] >= V1_CI_LOW
    rep["decision"] = "ADOPT" if rep["rule_a"] and rep["rule_b"] else "KEEP_V1"
    (REPORTS / "cv_v2_adoption.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    print(rep["decision"])


if __name__ == "__main__":
    main()
