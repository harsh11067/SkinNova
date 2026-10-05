"""SkinTimeline change metrics — Python twin of android timeline/{Aligner,LesionSegmenter,CoinDetector,ChangeMetrics}.kt.

contracts §8. Same OpenCV calls and parameters on both sides:
  align      ORB(1000) → BF Hamming kNN ratio 0.75 → findHomography RANSAC 4 px; align_ok = inlier_ratio ≥ 0.25 and ≥ 30 inliers
  segment    GrabCut (5 iters) seeded by the tap point: definite-FG disc r/3, probable-FG disc 1.6 r, definite-BG frame;
             largest component containing the seed; r = 0.12 × min side unless given
  coin       HoughCircles on a blurred grey image; radius 2–15 % of min side; must not overlap the lesion
  metrics    area_ratio (coin-scaled mm² if coin in both, else baseline-frame px after warp),
             contrast_delta = ΔE2000(lesion, surrounding ring) new − base, border irregularity P²/(4πA) delta,
             cv_shift = Jensen–Shannon divergence of CV probabilities
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from ml.timeline.colour import ciede2000, srgb_to_lab

ALIGN_MIN_RATIO, ALIGN_MIN_INLIERS = 0.25, 30


def align(base_rgb: np.ndarray, new_rgb: np.ndarray, seed: int = 0):
    """→ (H mapping new→base or None, inlier_ratio, n_inliers)."""
    cv2.setRNGSeed(seed)
    g1 = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY); g2 = cv2.cvtColor(new_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=1000)
    k1, d1 = orb.detectAndCompute(g1, None); k2, d2 = orb.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, 0.0, 0
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    if len(good) < 8:
        return None, 0.0, 0
    src = np.float32([k2[g.queryIdx].pt for g in good]).reshape(-1, 1, 2)
    dst = np.float32([k1[g.trainIdx].pt for g in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
    if H is None:
        return None, 0.0, 0
    n = int(mask.sum())
    return H, n / len(good), n


def segment(rgb: np.ndarray, seed_xy: tuple[float, float], r: float | None = None, iters: int = 5, rng_seed: int = 0) -> np.ndarray:
    """Binary lesion mask (uint8 0/1). seed_xy normalised (0..1)."""
    cv2.setRNGSeed(rng_seed)
    h, w = rgb.shape[:2]
    sx, sy = int(seed_xy[0] * w), int(seed_xy[1] * h)
    r = r or 0.12 * min(w, h)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    cv2.circle(mask, (sx, sy), int(1.6 * r), cv2.GC_PR_FGD, -1)
    cv2.circle(mask, (sx, sy), max(2, int(r / 3)), cv2.GC_FGD, -1)
    b = max(2, int(0.02 * min(w, h)))
    mask[:b, :] = mask[-b:, :] = cv2.GC_BGD; mask[:, :b] = mask[:, -b:] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), mask, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    fg = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg = cv2.morphologyEx(cv2.morphologyEx(fg, cv2.MORPH_OPEN, k), cv2.MORPH_CLOSE, k)
    n, lab = cv2.connectedComponents(fg)
    keep = lab[min(h - 1, sy), min(w - 1, sx)]
    if keep == 0:                                          # seed fell outside: take the largest component
        if n <= 1:
            return fg
        keep = 1 + int(np.argmax([(lab == i).sum() for i in range(1, n)]))
    return (lab == keep).astype(np.uint8)


def detect_coin(rgb: np.ndarray, lesion_mask: np.ndarray | None = None):
    """→ (cx, cy, r_px) or None."""
    g = cv2.medianBlur(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 5)
    m = min(g.shape)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=m * 0.2, param1=120, param2=40,
                         minRadius=int(0.02 * m), maxRadius=int(0.15 * m))
    if c is None:
        return None
    for x, y, r in c[0]:
        if lesion_mask is not None:
            disc = np.zeros_like(lesion_mask); cv2.circle(disc, (int(x), int(y)), int(r), 1, -1)
            if (disc & lesion_mask).sum() > 0.1 * disc.sum():
                continue
        return float(x), float(y), float(r)
    return None


def ring_mask(mask: np.ndarray) -> np.ndarray:
    a = max(1.0, math.sqrt(mask.sum() / math.pi))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.6 * a) + 1,) * 2)
    return (cv2.dilate(mask, k) & (1 - mask)).astype(np.uint8)


def lesion_contrast(rgb: np.ndarray, mask: np.ndarray) -> float:
    lab = srgb_to_lab(rgb.reshape(-1, 3)).reshape(rgb.shape)
    ring = ring_mask(mask)
    if mask.sum() == 0 or ring.sum() == 0:
        return 0.0
    return ciede2000(lab[mask.astype(bool)].mean(0), lab[ring.astype(bool)].mean(0))


def irregularity(mask: np.ndarray) -> float:
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea)
    A = cv2.contourArea(c)
    return float(cv2.arcLength(c, True) ** 2 / (4 * math.pi * A)) if A > 0 else 0.0


def js_divergence(p, q) -> float:
    p = np.asarray(p, float) + 1e-12; q = np.asarray(q, float) + 1e-12
    p /= p.sum(); q /= q.sum(); m = (p + q) / 2
    return float(0.5 * (p * np.log2(p / m)).sum() + 0.5 * (q * np.log2(q / m)).sum())


def change_metrics(base_rgb, new_rgb, seed_xy, coin_mm: float | None = None, cv_base=None, cv_new=None,
                   noise: dict | None = None, lesion_type: bool = True) -> dict:
    H, ratio, n = align(base_rgb, new_rgb)
    align_ok = H is not None and ratio >= ALIGN_MIN_RATIO and n >= ALIGN_MIN_INLIERS
    out = {"align_score": round(ratio, 4), "align_inliers": n, "align_ok": bool(align_ok), "lesion_type": lesion_type}
    if not align_ok:
        return {**out, "confidence": "low", "coin_in_both": False, "area_ratio": None, "contrast_delta": None,
                "border_irregularity_delta": None, "cv_shift_js": None}
    h, w = base_rgb.shape[:2]
    warped = cv2.warpPerspective(new_rgb, H, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    m0 = segment(base_rgb, seed_xy); m1 = segment(warped, seed_xy)
    a0, a1 = int(m0.sum()), int(m1.sum())
    c0, c1 = detect_coin(base_rgb, m0), detect_coin(new_rgb)
    coin_both = c0 is not None and c1 is not None and coin_mm
    if coin_both:
        # areas in mm² in each photo's own frame: new-frame mask = base-frame mask mapped back through H⁻¹
        m1_own = cv2.warpPerspective(m1, np.linalg.inv(H), (new_rgb.shape[1], new_rgb.shape[0]), flags=cv2.INTER_NEAREST)
        mm0 = a0 * (coin_mm / (2 * c0[2])) ** 2; mm1 = m1_own.sum() * (coin_mm / (2 * c1[2])) ** 2
        area_ratio = float(mm1 / mm0) if mm0 else None
    else:
        area_ratio = float(a1 / a0) if a0 else None
    cd = lesion_contrast(warped, m1) - lesion_contrast(base_rgb, m0)
    nf_ok = bool(noise and noise.get("n", 0) >= 3)
    conf = "ok" if (coin_both and align_ok and nf_ok) else "low"
    return {**out, "coin_in_both": bool(coin_both), "area_ratio": None if area_ratio is None else round(area_ratio, 4),
            "contrast_delta": round(float(cd), 3), "border_irregularity_delta": round(irregularity(m1) - irregularity(m0), 4),
            "cv_shift_js": None if cv_base is None else round(js_divergence(cv_base, cv_new), 4),
            "noise_floor": noise, "confidence": conf, "area_px": [a0, a1]}
