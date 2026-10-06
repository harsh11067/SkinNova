"""SkinTimeline change metrics — Python twin of android timeline/{Aligner,LesionSegmenter,CoinDetector,ChangeMetrics}.kt.

contracts §8. Same OpenCV calls and parameters on both sides:
  align      CLAHE(2.0, 8×8) grey → SIFT(2000) → BF L2 kNN ratio 0.75 → findHomography RANSAC 4 px;
             align_ok = inlier_ratio ≥ 0.25 and ≥ 30 inliers   (v1 ORB aligned only 65 % of same-spot pairs: smooth skin)
  segment    GrabCut (5 iters) seeded by the tap point: definite-FG disc r/3, probable-FG disc 1.6 r, definite-BG frame;
             inside the disc, pixels closer in Lab to the surrounding skin than Otsu's threshold start as probable-BG;
             largest component containing the seed. r adapts: baseline segmented with r = 0.12 × min side, then both
             photos are re-segmented with r = max(that, 1.25 × equivalent radius of the baseline mask) so growth isn't clipped
  coin       median 3×3 → HOUGH_GRADIENT_ALT candidates (radius 4–15 % of min side, min distance 5 %), verified as metal:
             mean HSV saturation inside the disc ≤ 0.35 and ≥ 50 % of the rim on a Canny(30,90) edge; a candidate that
             contains the tapped spot is the lesion, never the coin. The detected coin's
             disc is forced to background in GrabCut (a coin beside the spot, as the app asks, was swallowed by the lesion
             mask in 56 % of TL1 pairs). Chosen on TRAIN-split photos (decisions.md): both-photo recall 40 → 92 %,
             false coins on coin-free photos 18 → 7 % (mostly the Dermnet watermark "D"; the 4 % radius floor follows the
             15–20 cm capture distance: a ₹1 coin spans 5–7.5 % of the short side)
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


SEG_INIT = "otsu"     # "otsu": inside the probable-lesion disc, split by colour distance from the skin | "disc": whole disc
                      # (v2). Chosen on TRAIN pairs (decisions.md): coin-pair area error median 14.8 → 9.0 %, p90 35 → 28 %,
                      # colour-change error 2.0 → 0.8 ΔE; a definite-background ring beyond 2.2 r made everything worse


def _otsu_init(rgb: np.ndarray, mask: np.ndarray, sx: int, sy: int, r: float) -> None:
    """Inside the probable-lesion disc: probable FG where the Lab distance from the surrounding skin (mean of the annulus
    1.6–2.2 r) is above Otsu's threshold, probable BG elsewhere — GrabCut starts from a real boundary and a skin-only
    background model instead of keeping the whole disc. OpenCV 8-bit Lab, float64, half-even rounding (Kotlin twin)."""
    h, w = mask.shape
    disc = np.zeros((h, w), np.uint8); cv2.circle(disc, (sx, sy), int(1.6 * r), 1, -1)
    ann = np.zeros((h, w), np.uint8); cv2.circle(ann, (sx, sy), int(2.2 * r), 1, -1)
    ann[(disc == 1) | (mask == cv2.GC_BGD)] = 0                    # not the disc, not the frame / coin
    if ann.sum() < 50:
        return
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float64)
    mu = lab[ann == 1].mean(0)
    d = lab - mu
    d8 = np.clip(np.rint(2.0 * np.sqrt(d[..., 0] * d[..., 0] + d[..., 1] * d[..., 1] + d[..., 2] * d[..., 2])), 0, 255).astype(np.uint8)
    pr = (disc == 1) & (mask != cv2.GC_BGD) & (mask != cv2.GC_FGD)
    if pr.sum() < 50:
        return
    t, _ = cv2.threshold(d8[pr].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask[pr & (d8 > t)] = cv2.GC_PR_FGD
    mask[pr & (d8 <= t)] = cv2.GC_PR_BGD


def segment(rgb: np.ndarray, seed_xy: tuple[float, float], r: float | None = None, iters: int = 5, rng_seed: int = 0,
            exclude=None, init: str | None = None) -> np.ndarray:
    """Binary lesion mask (uint8 0/1). seed_xy normalised (0..1). exclude: (cx, cy, r) of a detected coin → background.
    init: override SEG_INIT (the TL1 generator pins "disc" so test pairs don't depend on the segmenter under test)."""
    cv2.setRNGSeed(rng_seed)
    h, w = rgb.shape[:2]
    sx, sy = int(seed_xy[0] * w), int(seed_xy[1] * h)
    r = r or 0.12 * min(w, h)
    mask = np.full((h, w), cv2.GC_PR_BGD, np.uint8)
    cv2.circle(mask, (sx, sy), int(1.6 * r), cv2.GC_PR_FGD, -1)
    cv2.circle(mask, (sx, sy), max(2, int(r / 3)), cv2.GC_FGD, -1)
    b = max(2, int(0.02 * min(w, h)))
    mask[:b, :] = mask[-b:, :] = cv2.GC_BGD; mask[:, :b] = mask[:, -b:] = cv2.GC_BGD
    if exclude is not None:
        cv2.circle(mask, (int(exclude[0]), int(exclude[1])), int(COIN_EXCLUDE * exclude[2]), cv2.GC_BGD, -1)
    if (init or SEG_INIT) == "otsu":
        _otsu_init(rgb, mask, sx, sy, r)
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


def segment_stable(rgb: np.ndarray, seed_xy, r: float, exclude=None) -> tuple[np.ndarray, float]:
    """Mask with prior r, and IoU against the mask with prior 1.4 r (stability)."""
    a = segment(rgb, seed_xy, r=r, exclude=exclude); b = segment(rgb, seed_xy, r=1.4 * r, exclude=exclude)
    inter = float((a & b).sum()); uni = float((a | b).sum())
    return a, (inter / uni if uni else 0.0)


COIN_EXCLUDE = 1.15   # coin disc × this (rim + contact shadow) is background for the lesion segmentation


def detect_coin(rgb: np.ndarray, seed_px: tuple[float, float] | None = None):
    """→ (cx, cy, r_px) or None. seed_px: the tapped spot in this photo's pixels (a circle containing it is the lesion)."""
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    g = cv2.medianBlur(grey, 3)
    m = min(g.shape)
    c = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=m * 0.05, param1=300, param2=0.8,
                         minRadius=int(0.04 * m), maxRadius=int(0.15 * m))
    if c is None:
        return None
    sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float32) / 255.0
    edges = cv2.Canny(g, 30, 90)
    for x, y, r in c[0]:
        if seed_px is not None and math.hypot(x - seed_px[0], y - seed_px[1]) <= r:
            continue                                        # contains the tapped spot: that is the lesion
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
        return float(x), float(y), float(r)
    return None


def ring_mask(mask: np.ndarray) -> np.ndarray:
    a = max(1.0, math.sqrt(mask.sum() / math.pi))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.6 * a) + 1,) * 2)
    return (cv2.dilate(mask, k) & (1 - mask)).astype(np.uint8)


def to_lab(rgb: np.ndarray) -> np.ndarray:
    return srgb_to_lab(rgb.reshape(-1, 3)).reshape(rgb.shape)


def lesion_contrast(rgb: np.ndarray, mask: np.ndarray, lab: np.ndarray | None = None) -> float:
    lab = to_lab(rgb) if lab is None else lab
    ring = ring_mask(mask)
    if mask.sum() == 0 or ring.sum() == 0:
        return 0.0
    return ciede2000(lab[mask.astype(bool)].mean(0), lab[ring.astype(bool)].mean(0))


SKIN_SD_CLAMP = (0.5, 2.0)


SKIN_NORM = "mean"   # "none" | "mean" | "meanstd" — chosen on TL1 variants (decisions.md): mean-only is best on both gates


def skin_normalised_lab(lab_new: np.ndarray, ring_new: np.ndarray, lab_base: np.ndarray, ring_base: np.ndarray,
                        mode: str | None = None) -> np.ndarray:
    """Re-light the new photo onto the base photo using the SURROUNDING SKIN as the reference: per-channel Lab mean/spread
    of the skin ring matched to the base's (spread ratio clamped). Exposure / white-balance shifts cancel; a lesion that
    really darkens or reddens relative to its own skin keeps that change (TL1 light-only gate, v3: 1.65 ΔE drift)."""
    mode = SKIN_NORM if mode is None else mode
    rb, rn = ring_base.astype(bool), ring_new.astype(bool)
    if mode == "none" or rb.sum() < 20 or rn.sum() < 20:
        return lab_new
    mu_b, sd_b = lab_base[rb].mean(0), lab_base[rb].std(0) + 1e-6
    mu_n, sd_n = lab_new[rn].mean(0), lab_new[rn].std(0) + 1e-6
    k = np.clip(sd_b / sd_n, *SKIN_SD_CLAMP) if mode == "meanstd" else 1.0
    return (lab_new - mu_n) * k + mu_b


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
    # coins first (each in its own photo; the tap mapped into the new photo by H⁻¹), so their discs are background
    seed_b = (seed_xy[0] * w, seed_xy[1] * h)
    sn = np.linalg.inv(H) @ np.array([seed_b[0], seed_b[1], 1.0])
    c0, c1 = detect_coin(base_rgb, seed_px=seed_b), detect_coin(new_rgb, seed_px=(sn[0] / sn[2], sn[1] / sn[2]))
    hom_scale = math.sqrt(abs(np.linalg.det(H[:2, :2])))          # new→base linear scale
    c1w = None
    if c1 is not None:                                             # the new photo's coin in the warped (base) frame
        q = H @ np.array([c1[0], c1[1], 1.0]); c1w = (q[0] / q[2], q[1] / q[2], c1[2] * hom_scale)
    m_init = segment(base_rgb, seed_xy, exclude=c0)
    r = max(0.12 * min(h, w), 1.25 * math.sqrt(max(1, m_init.sum()) / math.pi))
    m0, iou0 = segment_stable(base_rgb, seed_xy, r, exclude=c0)
    m1, iou1 = segment_stable(warped, seed_xy, r, exclude=c1w)
    seg_ok = min(iou0, iou1) >= 0.75
    a0, a1 = int(m0.sum()), int(m1.sum())
    area_ratio = float(a1 / a0) if a0 else None
    coin_both = bool(c0 is not None and c1 is not None and coin_mm)
    coin_scale_err = abs((c0[2] / c1[2]) / hom_scale - 1) if coin_both else None
    coin_ok = coin_both and coin_scale_err <= 0.10
    lab0, lab1 = to_lab(base_rgb), to_lab(warped)
    r0, r1 = ring_mask(m0), ring_mask(m1)
    c_base = lesion_contrast(base_rgb, m0, lab0)
    variants = {mode: round(float(lesion_contrast(warped, m1, skin_normalised_lab(lab1, r1, lab0, r0, mode)) - c_base), 3)
                for mode in ("none", "mean", "meanstd")}
    cd = variants[SKIN_NORM]
    nf_ok = bool(noise and noise.get("n", 0) >= 3)
    conf = "ok" if (coin_ok and align_ok and nf_ok and seg_ok) else "low"
    return {**out, "coin_in_both": bool(coin_both), "coin_scale_err": None if coin_scale_err is None else round(coin_scale_err, 4),
            "seg_iou": [round(iou0, 3), round(iou1, 3)], "seg_ok": bool(seg_ok),
            "area_mm2_base": round(a0 * (coin_mm / (2 * c0[2])) ** 2, 1) if coin_both else None,
            "area_ratio": None if area_ratio is None else round(area_ratio, 4),
            "contrast_delta": round(float(cd), 3), "contrast_variants": variants, "border_irregularity_delta": round(irregularity(m1) - irregularity(m0), 4),
            "cv_shift_js": None if cv_base is None else round(js_divergence(cv_base, cv_new), 4),
            "noise_floor": noise, "confidence": conf, "area_px": [a0, a1]}
