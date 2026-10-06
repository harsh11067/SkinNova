"""TL1 synthetic ground truth for SkinTimeline (test.md §8) → reports/timeline_eval.json

Base = val-split lesion photos (benign/suspicious; PAD-UFES + ISIC-style). For each pair we know the truth:
  area   lesion inpainted out, then re-pasted scaled about its centroid by √A (A ∈ [0.8, 1.6]); true ratio = mask areas
  colour lesion L* lowered inside the mask (target ΔE 3–15); true contrast delta measured on the ground-truth masks
  light  global gamma 0.7–1.4 + white-balance gains (lesion-vs-own-skin contrast should barely move)
  camera similarity (rot ±20°, scale ±15 %) + small perspective applied to the whole "after" photo
  coin   synthetic 20 mm coin pasted in both photos in ~half the pairs (scales with the camera)
Different-spot pairs (base vs another image's "after") measure false alignment.

  python -m ml.timeline.eval_timeline --n 300
"""
from __future__ import annotations

import argparse
import json
import math
import random
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd

from ml.common.paths import REPO, REPORTS, SPLITS, report_meta
from ml.timeline import metrics as M
from ml.timeline.colour import srgb_to_lab

COIN_MM = 20.0


def load(path) -> np.ndarray:
    im = cv2.cvtColor(cv2.imread(str(REPO / path)), cv2.COLOR_BGR2RGB)
    s = 512 / max(im.shape[:2])
    return cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else im


def draw_coin(img: np.ndarray, c, r) -> None:
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    d = np.hypot(xx - c[0], yy - c[1])
    inside = d <= r
    shade = (175 + 40 * (1 - d / r)).clip(0, 255)
    for k, tint in enumerate((1.0, 0.97, 0.9)):
        img[..., k][inside] = (shade * tint)[inside]
    rim = (d > r - max(2, r * 0.08)) & inside
    img[rim] = (110, 105, 95)


def make_pair(args):
    p = synth_pair(args)
    if p is None:
        return None
    base_c, after_cam, seed_xy, coin, truth = p
    met = M.change_metrics(base_c, after_cam, seed_xy, coin_mm=COIN_MM if coin else None, noise={"n": 3}, lesion_type=True)
    return {"i": args[0], "kind": args[3], **truth, **met}


def synth_pair(args):
    """→ (base photo, after photo, seed_xy, coin pasted?, ground truth) or None if the spot is out of range."""
    i, path, seed, kind = args
    rng = random.Random(seed)
    base = load(path)
    h, w = base.shape[:2]
    m0 = M.segment(base, (0.5, 0.5), init="disc")    # generator's lesion region: pinned (TL1 ≤ v6), independent of SEG_INIT
    frac = m0.sum() / (h * w)
    if not 0.01 <= frac <= 0.25:   # a tracked spot fills a modest part of a 15–20 cm photo
        return None
    ys, xs = np.nonzero(m0); cx, cy = xs.mean(), ys.mean()
    bg = cv2.inpaint(base, cv2.dilate(m0, np.ones((9, 9), np.uint8)) * 255, 7, cv2.INPAINT_TELEA)
    A = rng.uniform(0.8, 1.6) if kind in {"area", "area_coin"} else 1.0
    s = math.sqrt(A)
    S = np.array([[s, 0, cx * (1 - s)], [0, s, cy * (1 - s)]], np.float32)
    lesion = cv2.warpAffine(base, S, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m1 = cv2.warpAffine(m0, S, (w, h), flags=cv2.INTER_NEAREST)
    alpha = cv2.GaussianBlur(m1.astype(np.float32), (7, 7), 0)[..., None]
    after = (bg * (1 - alpha) + lesion * alpha).astype(np.uint8)
    true_area = m1.sum() / m0.sum()
    if kind == "colour":
        lab = cv2.cvtColor(after, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 0][m1.astype(bool)] -= rng.uniform(3, 15) * 255 / 100
        after = cv2.cvtColor(lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
    if kind == "light":
        g = rng.uniform(0.7, 1.4); gains = np.array([rng.uniform(0.9, 1.1) for _ in range(3)])
        after = (255 * ((after / 255.0) ** g) * gains).clip(0, 255).astype(np.uint8)
    true_cd = M.lesion_contrast(after, m1) - M.lesion_contrast(base, m0)   # on ground-truth masks, before the camera move
    coin = kind == "area_coin"
    base_c = base.copy()
    if coin:
        # beside the spot, as the app tells users (strings tl_coin_tip): just outside the (possibly grown) lesion, towards the
        # image centre, kept inside the frame. v3 drew it in a corner, where the ±20° / ±15 % camera move cut it off in half of
        # the pairs (coin found in both photos: 47 %).
        r0 = 0.07 * min(h, w); rl = math.sqrt(m0.sum() / math.pi) * math.sqrt(A)
        vx, vy = (w / 2 - cx), (h / 2 - cy); nv = math.hypot(vx, vy)
        if nv < 0.05 * min(h, w):                   # centred spot (typical): put the coin below-right of it instead
            vx, vy, nv = 1.0, 1.0, math.sqrt(2.0)
        d = rl + 1.6 * r0
        cc = (float(np.clip(cx + vx / nv * d, 1.8 * r0, w - 1.8 * r0)), float(np.clip(cy + vy / nv * d, 1.8 * r0, h - 1.8 * r0)))
        draw_coin(base_c, cc, r0); draw_coin(after, cc, r0)
    rot, sc = rng.uniform(-20, 20), rng.uniform(0.85, 1.15)
    R = cv2.getRotationMatrix2D((w / 2, h / 2), rot, sc)
    Hc = np.vstack([R, [rng.uniform(-2e-4, 2e-4), rng.uniform(-2e-4, 2e-4), 1]])
    after_cam = cv2.warpPerspective(after, Hc, (w, h), borderMode=cv2.BORDER_REFLECT)
    truth = {"true_area_ratio": float(true_area), "true_contrast_delta": float(true_cd), "rot": rot, "scale": sc,
             "coin_xyr": [cc[0], cc[1], r0] if coin else None}       # in base-photo pixels
    return base_c, after_cam, (cx / w, cy / h), coin, truth


def diff_pair(args):
    i, p1, p2 = args
    a, b = load(p1), load(p2)
    b = cv2.resize(b, (a.shape[1], a.shape[0]))
    H, ratio, n = M.align(a, b)
    return {"i": i, "align_ok": bool(H is not None and ratio >= M.ALIGN_MIN_RATIO and n >= M.ALIGN_MIN_INLIERS), "ratio": ratio, "n": n}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--kinds", nargs="+", default=["area", "area_coin", "colour", "light"])
    ap.add_argument("--out", default="timeline_eval.json")
    a = ap.parse_args()
    va = pd.read_csv(SPLITS / "val.csv")
    les = va[va.label.isin(["benign_lesion", "suspicious_lesion"])].sample(frac=1, random_state=7).img.tolist()
    kinds = a.kinds
    jobs = [(i, les[i % len(les)], 1000 + i, kinds[i % len(kinds)]) for i in range(int(a.n * 1.4))]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(make_pair, jobs, chunksize=4) if r is not None][:a.n]
        diffs = list(ex.map(diff_pair, [(i, les[i], les[-1 - i]) for i in range(100)], chunksize=4))
    df = pd.DataFrame(rows)
    ok = df[df.align_ok]
    rel = lambda d: (d.area_ratio - d.true_area_ratio).abs() / d.true_area_ratio
    coin = ok[(ok.kind == "area_coin") & (ok.confidence == "ok")]        # what the app would report as confidence OK
    nocoin_scaled = df[(df.kind == "area") & ((df.scale - 1).abs() > 0.05)]
    light = ok[ok.kind == "light"]; col = ok[ok.kind == "colour"]
    rep = {**report_meta(), "n_pairs": int(len(df)), "kinds": df.kind.value_counts().to_dict(),
           "align_ok_same_spot": float(df.align_ok.mean()),
           "align_ok_different_spot": float(np.mean([d["align_ok"] for d in diffs])),
           "coin_detected_both_rate": float(df[df.kind == "area_coin"].coin_in_both.mean()),
           "area_rel_err_coin": {"median": float(rel(coin).median()) if len(coin) else None, "p90": float(rel(coin).quantile(.9)) if len(coin) else None, "n": int(len(coin))},
           "area_rel_err_no_coin": {"median": float(rel(ok[ok.kind == "area"]).median()), "p90": float(rel(ok[ok.kind == "area"]).quantile(.9))},
           "area_rel_err_seg_stable": {"median": float(rel(ok[ok.kind.str.startswith("area") & ok.seg_ok]).median()),
                                       "n": int((ok.kind.str.startswith("area") & ok.seg_ok).sum())},
           "seg_stable_rate": float(ok.seg_ok.mean()), "confidence_ok_rate_coin_pairs": float((df[df.kind == "area_coin"].confidence == "ok").mean()),
           "no_coin_scaled_flagged_low": float((nocoin_scaled.confidence == "low").mean()),
           "contrast_abs_err_light_only": float((light.contrast_delta - light.true_contrast_delta).abs().median()),
           "contrast_abs_delta_light_only": float(light.contrast_delta.abs().median()),
           "contrast_abs_err_colour": float((col.contrast_delta - col.true_contrast_delta).abs().median()),
           }
    rep["gates"] = {
        "area_coin_median_le_10pct": rep["area_rel_err_coin"]["median"] is not None and rep["area_rel_err_coin"]["median"] <= 0.10,
        "area_coin_p90_le_20pct": rep["area_rel_err_coin"]["p90"] is not None and rep["area_rel_err_coin"]["p90"] <= 0.20,
        "no_coin_low_conf_ge_95pct": rep["no_coin_scaled_flagged_low"] >= 0.95,
        "light_only_median_le_1_5": rep["contrast_abs_delta_light_only"] <= 1.5,
        "colour_err_median_le_2": rep["contrast_abs_err_colour"] <= 2.0,
        "align_same_ge_95pct": rep["align_ok_same_spot"] >= 0.95,
        "align_diff_le_5pct": rep["align_ok_different_spot"] <= 0.05,
    }
    # contrast normalisation variants (ml/timeline/metrics.py SKIN_NORM): same alignment/masks, three re-lighting choices
    if "contrast_variants" in df:
        rep["contrast_variants"] = {}
        for mode in ("none", "mean", "meanstd"):
            lv = light.contrast_variants.map(lambda d: d[mode]); cv_ = col.contrast_variants.map(lambda d: d[mode])
            rep["contrast_variants"][mode] = {"light_abs_delta_median": float(lv.abs().median()) if len(lv) else None,
                                              "colour_abs_err_median": float((cv_ - col.true_contrast_delta).abs().median()) if len(cv_) else None}
    (REPORTS / a.out).write_text(json.dumps(rep, indent=1, default=float))
    df.to_csv(REPORTS / "timeline_eval_pairs.csv", index=False)
    print(json.dumps({k: v for k, v in rep.items() if k not in ("git_sha", "created_at")}, indent=1, default=float))


if __name__ == "__main__":
    main()
