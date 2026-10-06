"""TL3 fixtures (test.md §8): Kotlin Timeline ≡ ml/timeline/metrics.py on the same images.

A few TL1 synthetic pairs per kind (eval_timeline.synth_pair, same val photos and seeds as the TL1 run) plus different-spot
pairs are saved as lossless PNG; the expected metrics are computed by Python on the decoded PNG bytes, i.e. exactly what
the phone decodes. The androidTest TimelineParityTest runs Timeline.compute on them.

  python -m ml.timeline.make_tl3_fixtures      # → tests/fixtures/metrics_cases.json (the oracle) and
                                               #   android/app/src/androidTest/assets/tl3/ (PNGs + a copy of the json)
Tolerance (test.md TL3): area ratio 2 % relative, contrast ΔE 0.5; align_ok and coin_in_both must agree.
"""
from __future__ import annotations

import argparse
import json
import shutil

import cv2
import pandas as pd

from ml.common.paths import REPO, SPLITS, report_meta
from ml.timeline import metrics as M
from ml.timeline.eval_timeline import COIN_MM, load, synth_pair

FIX = REPO / "tests/fixtures"
ASSETS = REPO / "android/app/src/androidTest/assets/tl3"
KEYS = ["align_ok", "align_inliers", "align_score", "coin_in_both", "coin_scale_err", "seg_ok", "area_ratio", "contrast_delta",
        "border_irregularity_delta", "confidence"]


def read_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per-kind", type=int, default=3); ap.add_argument("--diff", type=int, default=2)
    a = ap.parse_args()
    va = pd.read_csv(SPLITS / "val.csv")
    les = va[va.label.isin(["benign_lesion", "suspicious_lesion"])].sample(frac=1, random_state=7).img.tolist()   # = TL1
    kinds = ["area", "area_coin", "colour", "light"]
    img_dir = ASSETS; shutil.rmtree(img_dir, ignore_errors=True); img_dir.mkdir(parents=True)
    cases, have, i = [], {k: 0 for k in kinds}, 0
    while min(have.values()) < a.per_kind:
        kind = kinds[i % len(kinds)]
        p = synth_pair((i, les[i % len(les)], 1000 + i, kind)) if have[kind] < a.per_kind else None
        i += 1
        if p is None:
            continue
        base, new, seed_xy, coin, truth = p
        cid = f"{kind}_{i - 1:03d}"
        cv2.imwrite(str(img_dir / f"{cid}_base.png"), cv2.cvtColor(base, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img_dir / f"{cid}_new.png"), cv2.cvtColor(new, cv2.COLOR_RGB2BGR))
        cases.append({"id": cid, "kind": kind, "base": f"{cid}_base.png", "new": f"{cid}_new.png", "seed_xy": list(seed_xy),
                      "coin_mm": COIN_MM if coin else None, "truth": truth})
        have[kind] += 1
    for j in range(a.diff):                       # different spots: alignment must fail on both sides
        b, n = load(les[j]), load(les[-1 - j])
        n = cv2.resize(n, (b.shape[1], b.shape[0]))
        cid = f"diff_{j:03d}"
        cv2.imwrite(str(img_dir / f"{cid}_base.png"), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(img_dir / f"{cid}_new.png"), cv2.cvtColor(n, cv2.COLOR_RGB2BGR))
        cases.append({"id": cid, "kind": "different_spot", "base": f"{cid}_base.png", "new": f"{cid}_new.png", "seed_xy": [0.5, 0.5],
                      "coin_mm": None, "truth": {}})
    for c in cases:                               # expected = Python on the decoded PNGs (what the phone sees)
        met = M.change_metrics(read_rgb(img_dir / c["base"]), read_rgb(img_dir / c["new"]), tuple(c["seed_xy"]),
                               coin_mm=c["coin_mm"], noise={"n": 3}, lesion_type=True)
        c["expected"] = {k: met.get(k) for k in KEYS}
        print(c["id"], {k: c["expected"][k] for k in ("align_ok", "area_ratio", "contrast_delta", "coin_in_both", "confidence")})
    out = {**report_meta(), "opencv_python": cv2.__version__, "tolerance": {"area_ratio_rel": 0.02, "contrast_delta_abs": 0.5},
           "noise_floor_n": 3, "cases": cases}
    (FIX / "metrics_cases.json").write_text(json.dumps(out, indent=1))
    shutil.copyfile(FIX / "metrics_cases.json", ASSETS / "metrics_cases.json")
    print(len(cases), "cases →", FIX / "metrics_cases.json", "and", ASSETS)


if __name__ == "__main__":
    main()
