"""Quality-gate thresholds (test.md C6) — Python twin of android ml/QualityGate.kt, tuned on val photos.

Same maths as Kotlin: box-average to ≤ 256 px, BT.601 integer luma, 4-neighbour Laplacian variance, mean luma,
YCbCr skin box (Cb 77–127, Cr 133–173) on a strided sample. Variants per image: clean, blur σ=2, blur σ=3,
dark (×0.25), bright (gamma 0.25). Writes reports/quality_gate.json with the chosen thresholds:
  BLUR_MIN  = highest value with ≤ 5 % false rejects on clean photos, provided σ=3 rejection ≥ 90 %
  LUMA_MIN / LUMA_MAX from clean-photo 1st / 99th percentiles (with margin)
  python -m ml.eval.quality_gate --n 500
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

from ml.common.paths import REPO, REPORTS, SPLITS, report_meta


def gray_down(rgb: np.ndarray, max_side: int = 256) -> np.ndarray:
    h, w = rgb.shape[:2]
    f = max(1, (max(w, h) + max_side - 1) // max_side)
    H, W = h // f, w // f
    c = rgb.astype(np.int64)
    y = (299 * c[..., 0] + 587 * c[..., 1] + 114 * c[..., 2]) // 1000
    return y[:H * f, :W * f].reshape(H, f, W, f).sum((1, 3)) // (f * f)


def measures(rgb: np.ndarray) -> dict:
    g = gray_down(rgb).astype(np.float64)
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4 * g[1:-1, 1:-1]
    flat = rgb.reshape(-1, 3).astype(np.float64)
    step = max(1, len(flat) // 40000)
    s = flat[::step]
    cb = 128 - 0.168736 * s[:, 0] - 0.331264 * s[:, 1] + 0.5 * s[:, 2]
    cr = 128 + 0.5 * s[:, 0] - 0.418688 * s[:, 1] - 0.081312 * s[:, 2]
    skin = float(((cb >= 77) & (cb <= 127) & (cr >= 133) & (cr <= 173)).mean())
    return {"blur": float(lap.var()), "luma": float(g.mean()), "skin": skin}


def variants(path):
    im = Image.open(REPO / path).convert("RGB")
    out = {"clean": im, "blur2": im.filter(ImageFilter.GaussianBlur(2)), "blur3": im.filter(ImageFilter.GaussianBlur(3)),
           "dark": Image.eval(im, lambda v: int(v * 0.25)), "bright": Image.eval(im, lambda v: int(255 * (v / 255) ** 0.25))}
    return {k: measures(np.asarray(v)) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=500); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    va = pd.read_csv(SPLITS / "val.csv").sample(a.n, random_state=3407)
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(variants, va.img, chunksize=8))
    M = {k: pd.DataFrame([r[k] for r in rows]) for k in rows[0]}
    clean = M["clean"]
    best = None
    for t in np.unique(np.concatenate([np.linspace(5, 400, 200)])):
        fr = float((clean.blur < t).mean()); rej3 = float((M["blur3"].blur < t).mean())
        if fr <= 0.05 and rej3 >= 0.90:
            best = {"BLUR_MIN": float(t), "clean_false_reject": fr, "blur3_reject": rej3, "blur2_reject": float((M["blur2"].blur < t).mean())}
    if best is None:   # cannot satisfy both: report the trade-off honestly at 90 % σ=3 rejection
        t = float(np.quantile(M["blur3"].blur, 0.90))
        best = {"BLUR_MIN": t, "clean_false_reject": float((clean.blur < t).mean()), "blur3_reject": 0.90,
                "blur2_reject": float((M["blur2"].blur < t).mean()), "note": "cannot reach ≤5 % clean false rejects at 90 % σ=3 rejection"}
    luma_min = float(max(20.0, np.quantile(clean.luma, 0.01) - 10)); luma_max = float(min(240.0, np.quantile(clean.luma, 0.99) + 10))
    skin_min = float(max(0.02, np.quantile(clean.skin, 0.03)))
    rep = {**report_meta(), "n": int(a.n), **best, "LUMA_MIN": luma_min, "LUMA_MAX": luma_max, "SKIN_MIN": skin_min,
           "dark_reject": float((M["dark"].luma < luma_min).mean()), "bright_reject": float((M["bright"].luma > luma_max).mean()),
           "clean_skin_reject": float((clean.skin < skin_min).mean()),
           "quantiles": {k: {c: np.quantile(v[c], [.05, .5, .95]).round(2).tolist() for c in v} for k, v in M.items()}}
    rep["C6_pass"] = bool(rep["blur3_reject"] >= 0.90)
    (REPORTS / "quality_gate.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ["BLUR_MIN", "clean_false_reject", "blur3_reject", "blur2_reject", "LUMA_MIN", "LUMA_MAX",
                                           "SKIN_MIN", "dark_reject", "bright_reject", "clean_skin_reject", "C6_pass"]}, indent=1))


if __name__ == "__main__":
    main()
