"""SkinTimeline change metrics — Python twin of android timeline/{Aligner,LesionSegmenter,CoinDetector,ChangeMetrics}.kt.

contracts §8. Same OpenCV calls and parameters on both sides:
  align      CLAHE(2.0, 8×8) grey → SIFT(2000) → BF L2 kNN ratio 0.75 → findHomography RANSAC 4 px;
             align_ok = inlier_ratio ≥ 0.25 and ≥ 30 inliers   (v1 ORB aligned only 65 % of same-spot pairs: smooth skin)
  segment    GrabCut (5 iters) seeded by the tap point: definite-FG disc r/3, probable-FG disc 1.6 r, definite-BG frame;
             largest component containing the seed. r adapts: baseline segmented with r = 0.12 × min side, then both
             photos are re-segmented with r = max(that, 1.25 × equivalent radius of the baseline mask) so growth isn't clipped
  coin       HOUGH_GRADIENT_ALT candidates (radius 2–15 % of min side), verified as metal: mean HSV saturation inside
             the disc ≤ 0.35 and ≥ 50 % of the rim on a Canny(30,90) edge; must not overlap the lesion
  stability  each photo is segmented with priors r and 1.4 r; IoU < 0.75 → segmentation unstable → confidence low
             (faint / hair-covered / textured lesions: v2 eval showed these produce the worst area errors)
  metrics    area_ratio from the homography-aligned masks (scale-free); the coin is a SCALE CHECK: coin px ratio between
             photos must agree with the homography scale within 10 % for confidence "ok" (single noisy coin detections
             made coin-scaled areas worse: median error 29.5 % in v2),
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
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g1 = clahe.apply(cv2.cvtColor(base_rgb, cv2.COLOR_RGB2GRAY)); g2 = clahe.apply(cv2.cvtColor(new_rgb, cv2.COLOR_RGB2GRAY))
    sift = cv2.SIFT_create(nfeatures=2000)
    k1, d1 = sift.detectAndCompute(g1, None); k2, d2 = sift.detectAndCompute(g2, None)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return None, 0.0, 0
    m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(d2, d1, k=2)
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


def segment_stable(rgb: np.ndarray, seed_xy, r: float) -> tuple[np.ndarray, float]:
    """Mask with prior r, and IoU against the mask with prior 1.4 r (stability)."""
    a = segment(rgb, seed_xy, r=r); b = segment(rgb, seed_xy, r=1.4 * r)
    inter = float((a & b).sum()); uni = float((a | b).sum())
    return a, (inter / uni if uni else 0.0)


def detect_coin(rgb: np.ndarray, lesion_mask: np.ndarray | None = None):
    """→ (cx, cy, r_px) or None."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(grey, 5)
    m = min(g.shape)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=m * 0.2, param1=300, param2=0.8,
                         minRadius=int(0.02 * m), maxRadius=int(0.15 * m))
    if c is None:
        return None
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    edges = cv2.Canny(g, 30, 90)
    for x, y, r in c[0]:
        disc = np.zeros(g.shape, np.uint8); cv2.circle(disc, (int(x), int(y)), int(r * 0.85), 1, -1)
        if disc.sum() == 0 or float(sat[disc.astype(bool)].mean()) > 0.35:
            continue                                        # coloured blob (lesion, bubble), not metal
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        support = 0
        for t in ang:                                       # rim support: an edge pixel within ±2 px of the circle
            ok = False
            for dr in (-2, -1, 0, 1, 2):
                px, py = int(round(x + (r + dr) * np.cos(t))), int(round(y + (r + dr) * np.sin(t)))
                if 0 <= px < g.shape[1] and 0 <= py < g.shape[0] and edges[py, px]:
                    ok = True; break
            support += ok
        if support / len(ang) < 0.5:
            continue
        if lesion_mask is not None:
            full = np.zeros_like(lesion_mask); cv2.circle(full, (int(x), int(y)), int(r), 1, -1)
            if (full & lesion_mask).sum() > 0.1 * full.sum():
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
    m_init = segment(base_rgb, seed_xy)
    r = max(0.12 * min(h, w), 1.25 * math.sqrt(max(1, m_init.sum()) / math.pi))
    m0, iou0 = segment_stable(base_rgb, seed_xy, r)
    m1, iou1 = segment_stable(warped, seed_xy, r)
    seg_ok = min(iou0, iou1) >= 0.75
    a0, a1 = int(m0.sum()), int(m1.sum())
    area_ratio = float(a1 / a0) if a0 else None
    c0, c1 = detect_coin(base_rgb, m0), detect_coin(new_rgb)
    coin_both = bool(c0 is not None and c1 is not None and coin_mm)
    hom_scale = math.sqrt(abs(np.linalg.det(H[:2, :2])))          # new→base linear scale
    coin_scale_err = abs((c0[2] / c1[2]) / hom_scale - 1) if coin_both else None
    coin_ok = coin_both and coin_scale_err <= 0.10
    cd = lesion_contrast(warped, m1) - lesion_contrast(base_rgb, m0)
    nf_ok = bool(noise and noise.get("n", 0) >= 3)
    conf = "ok" if (coin_ok and align_ok and nf_ok and seg_ok) else "low"
    return {**out, "coin_in_both": bool(coin_both), "coin_scale_err": None if coin_scale_err is None else round(coin_scale_err, 4),
            "seg_iou": [round(iou0, 3), round(iou1, 3)], "seg_ok": bool(seg_ok),
            "area_mm2_base": round(a0 * (coin_mm / (2 * c0[2])) ** 2, 1) if coin_both else None,
            "area_ratio": None if area_ratio is None else round(area_ratio, 4),
            "contrast_delta": round(float(cd), 3), "border_irregularity_delta": round(irregularity(m1) - irregularity(m0), 4),
            "cv_shift_js": None if cv_base is None else round(js_divergence(cv_base, cv_new), 4),
            "noise_floor": noise, "confidence": conf, "area_px": [a0, a1]}
