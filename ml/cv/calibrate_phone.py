"""v2.1 items 6 + 7: calibrate the image model on PHONE photos, and a conformal "most likely one of these" short list.

Pre-registered (decisions 2026-10-09). SCIN-val (consented phone photos, the closest to real users) is split 50/50 by
case id (seed 3407) into scin_calib (fit) / scin_holdout (judge); the split is frozen in data/splits/scin_calib.csv.
Everything uses the shipped inference: TTA mean over 4 views (id, h, v, r180) of softmax(logits / T).

Item 6: temperature T refit on scin_calib (NLL); uncertainty cut-offs from scin_calib ("low" where top-1 acc ≥ 80 %,
        "high" where < 50 %). Adopt iff holdout NLL and ECE both lower, full-val argmax / top-1 unchanged, suspicious
        recall (p ≥ 0.15) ≥ current − 0.5 pt, holdout "low" bucket accuracy ≥ 80 %.
Item 7: split-conformal APS (non-randomised), target 90 %, on scin_calib with the adopted T; suspicious_lesion forced in
        at p ≥ 0.15. Adopt iff holdout coverage 88–92 % and forced inclusion 100 %. Groups < 80 % are disclosed.

  python -m ml.cv.calibrate_phone     → reports/cv_calibration_phone.json, android assets cv/calibration.json,
                                         tests/fixtures/conformal_cases.json
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from ml.common.paths import MODELS, REPORTS, SPLITS, report_meta
from ml.cv.eval_cv import load_model
from ml.cv.tta_eval import view_logits
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VIEWS = ["id", "h", "v", "r180"]
ALPHA = 0.10
FORCE = {"suspicious_lesion": 0.15}


def probs(L: dict, T: float) -> np.ndarray:
    def sm(z):
        z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
    return np.mean([sm(L[v]) for v in VIEWS], 0)


def nll(P, Y):
    return float(-np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1))))


def ece(P, Y, bins=15):
    conf = P.max(1); acc = (P.argmax(1) == Y).astype(float); e = 0.0; rel = []
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean()); rel.append({"bin": [float(lo), float(hi)], "n": int(m.sum()),
                                                                          "conf": float(conf[m].mean()), "acc": float(acc[m].mean())})
    return float(e), rel


def fit_T(L, Y):
    grid = np.exp(np.linspace(np.log(0.4), np.log(6.0), 400))
    vals = [nll(probs(L, t), Y) for t in grid]
    return float(grid[int(np.argmin(vals))])


def cutoffs(P, Y):
    """t_low: smallest p with acc(top1 p ≥ t) ≥ 0.80; t_high: largest p with acc(top1 p < t) < 0.50."""
    conf = P.max(1); ok = P.argmax(1) == Y
    grid = np.round(np.arange(0.20, 0.99, 0.01), 2)
    t_low = next((float(t) for t in grid if (conf >= t).sum() >= 10 and ok[conf >= t].mean() >= 0.80), 0.99)
    t_high = max((float(t) for t in grid if (conf < t).sum() >= 10 and ok[conf < t].mean() < 0.50), default=0.20)
    return t_low, min(t_high, t_low)


def bucket(p, t_low, t_high):
    return "low" if p >= t_low else ("high" if p < t_high else "moderate")


def aps_scores(P, Y):
    order = np.argsort(-P, 1); s = []
    for i in range(len(Y)):
        cum = np.cumsum(P[i, order[i]]); s.append(float(cum[list(order[i]).index(Y[i])]))
    return np.array(s)


def aps_set(p: np.ndarray, qhat: float, classes: list[str]) -> list[str]:
    """Classes in descending probability until the cumulative mass reaches qhat (the crossing class included); forced
    classes added when above their floor. Kotlin twin: CvClassifier.shortList (shared fixture conformal_cases.json)."""
    order = np.argsort(-p, kind="stable"); out, cum = [], 0.0
    for j in order:
        out.append(classes[j]); cum += p[j]
        if cum >= qhat - 1e-12:
            break
    for k, floor in FORCE.items():
        if k in classes and k not in out and p[classes.index(k)] >= floor:
            out.append(k)
    return out


def set_word(s: list[str]) -> str:
    return "high" if len(s) >= 4 or "other" in s else ("low" if len(s) == 1 else "moderate")


def main():
    model, ck = load_model(MODELS / "cv" / "ckpt" / "best.pt"); classes = ck["classes"]; T0 = float(ck["temperature"])
    val = pd.read_csv(SPLITS / "val.csv"); val = val[val.label.isin(classes)].reset_index(drop=True)
    cache = MODELS / "cv" / "val_tta_logits.npz"
    if cache.exists():
        z = np.load(cache); L = {v: z[v] for v in VIEWS}; Y = z["y"]
    else:
        Lall, Y = view_logits(model, val, classes, ck.get("color_constancy")); L = {v: Lall[v] for v in VIEWS}
        np.savez(cache, y=Y, **L)
    scin = (val.source == "scin").values
    cases = sorted(val[scin].group.unique()); rng = np.random.RandomState(3407); rng.shuffle(cases)
    calib_cases = set(cases[: len(cases) // 2])
    is_cal = scin & val.group.isin(calib_cases).values; is_hold = scin & ~val.group.isin(calib_cases).values
    val[is_cal][["img", "label", "group"]].to_csv(SPLITS / "scin_calib.csv", index=False)
    val[is_hold][["img", "label", "group"]].to_csv(SPLITS / "scin_holdout.csv", index=False)
    sub = lambda m: {v: L[v][m] for v in VIEWS}
    Lc, Yc, Lh, Yh = sub(is_cal), Y[is_cal], sub(is_hold), Y[is_hold]

    # ---- item 6
    T1 = fit_T(Lc, Yc)
    P0h, P1h = probs(Lh, T0), probs(Lh, T1); P0v, P1v = probs(L, T0), probs(L, T1)
    e0, rel0 = ece(P0h, Yh); e1, rel1 = ece(P1h, Yh)
    sus = classes.index("suspicious_lesion"); sm = Y == sus
    rec0 = float((P0v[sm, sus] >= 0.15).mean()); rec1 = float((P1v[sm, sus] >= 0.15).mean())
    t_low, t_high = cutoffs(probs(Lc, T1), Yc)
    def buckets(P, Yb, tl, th):
        conf = P.max(1); ok = P.argmax(1) == Yb; out = {}
        for b in ("low", "moderate", "high"):
            m = np.array([bucket(c, tl, th) == b for c in conf])
            out[b] = {"n": int(m.sum()), "share": float(m.mean()), "acc": float(ok[m].mean()) if m.any() else None}
        return out
    item6 = {"T_before": T0, "T_after": T1, "t_low": t_low, "t_high": t_high,
             "holdout": {"n": int(len(Yh)), "nll_before": nll(P0h, Yh), "nll_after": nll(P1h, Yh), "ece_before": e0, "ece_after": e1,
                         "reliability_before": rel0, "reliability_after": rel1,
                         "confident_but_wrong_before": float(((P0h.max(1) >= 0.6) & (P0h.argmax(1) != Yh)).sum() / max(1, (P0h.max(1) >= 0.6).sum())),
                         "buckets_old_cutoffs_before": buckets(P0h, Yh, 0.7, 0.4), "buckets_new_cutoffs_after": buckets(P1h, Yh, t_low, t_high)},
             "full_val": {"argmax_changed": float((P0v.argmax(1) != P1v.argmax(1)).mean()), "top1_before": float((P0v.argmax(1) == Y).mean()),
                          "top1_after": float((P1v.argmax(1) == Y).mean()), "suspicious_recall_p015_before": rec0, "suspicious_recall_p015_after": rec1}}
    h = item6["holdout"]; fv = item6["full_val"]
    item6["rule"] = {"nll_lower": h["nll_after"] < h["nll_before"], "ece_lower": h["ece_after"] < h["ece_before"],
                     "argmax_unchanged": fv["argmax_changed"] == 0.0, "top1_unchanged": abs(fv["top1_after"] - fv["top1_before"]) < 1e-9,
                     "sus_recall_ok": fv["suspicious_recall_p015_after"] >= fv["suspicious_recall_p015_before"] - 0.005,
                     "low_bucket_acc_ge_80": (h["buckets_new_cutoffs_after"]["low"]["acc"] or 0) >= 0.80}
    item6["adopt"] = all(item6["rule"].values())
    T = T1 if item6["adopt"] else T0

    # ---- item 7 (with the adopted temperature)
    Pc, Ph = probs(Lc, T), probs(Lh, T)
    sc = aps_scores(Pc, Yc); n = len(sc); qhat = float(np.quantile(sc, min(1.0, math.ceil((n + 1) * (1 - ALPHA)) / n), method="higher"))
    sets = [aps_set(p, qhat, classes) for p in Ph]
    cov = float(np.mean([classes[y] in s for s, y in zip(sets, Yh)]))
    forced_ok = all(("suspicious_lesion" in s) for s, p in zip(sets, Ph) if p[sus] >= 0.15)
    hold_df = val[is_hold].reset_index(drop=True)
    per_tone = {}
    for t, g in hold_df.groupby("skin_tone"):
        idx = g.index.values; per_tone[t] = {"n": int(len(idx)), "coverage": float(np.mean([classes[Yh[i]] in sets[i] for i in idx]))}
    words = pd.Series([set_word(s) for s in sets]).value_counts(normalize=True).to_dict()
    word_acc = {w: float(np.mean([Ph[i].argmax() == Yh[i] for i in range(len(sets)) if set_word(sets[i]) == w])) for w in words}
    # external brown-skin phone photos: reported only (its labels are a subset of the classes)
    ext = pd.read_csv(SPLITS / "external_test.csv"); ext = ext[ext.label.isin(classes)].reset_index(drop=True)
    from ml.cv.tta_eval import view_logits as vl
    Le, Ye = vl(model, ext, classes, ck.get("color_constancy")); Pe = probs({v: Le[v] for v in VIEWS}, T)
    se = [aps_set(p, qhat, classes) for p in Pe]
    item7 = {"alpha": ALPHA, "qhat": qhat, "n_calib": n, "holdout": {"n": int(len(Yh)), "coverage": cov, "mean_size": float(np.mean([len(s) for s in sets])),
             "size_hist": pd.Series([len(s) for s in sets]).value_counts().sort_index().to_dict(), "forced_suspicious_ok": forced_ok,
             "per_skin_tone": per_tone, "uncertainty_word_share": words, "top1_acc_by_word": word_acc},
             "external_brown_skin_reported_only": {"n": int(len(Ye)), "coverage": float(np.mean([classes[y] in s for s, y in zip(se, Ye)])),
                                                   "mean_size": float(np.mean([len(s) for s in se]))}}
    item7["rule"] = {"coverage_88_92": 0.88 <= cov <= 0.92, "forced_inclusion_100": forced_ok}
    item7["adopt"] = all(item7["rule"].values())
    item7["disclose_groups_below_80"] = {k: v for k, v in per_tone.items() if v["coverage"] < 0.80 and v["n"] >= 10}

    rep = {**report_meta(model_sha=ck.get("arch", "")), "views": VIEWS, "split": {"calib_images": int(is_cal.sum()), "holdout_images": int(is_hold.sum()),
           "calib_cases": len(calib_cases), "seed": 3407}, "item6_calibration": item6, "item7_conformal": item7}
    (REPORTS / "cv_calibration_phone.json").write_text(json.dumps(rep, indent=1, default=float))
    # app + fixture artefacts (written whatever the decision; the app reads "adopted" flags)
    calib = {"temperature": T, "temperature_adopted": item6["adopt"], "t_low": t_low if item6["adopt"] else 0.7,
             "t_high": t_high if item6["adopt"] else 0.4, "conformal": {"adopted": item7["adopt"], "qhat": qhat, "alpha": ALPHA,
             "force": FORCE, "coverage_holdout": cov, "report": "reports/cv_calibration_phone.json"}}
    (REPO / "android/app/src/main/assets/cv/calibration.json").write_text(json.dumps(calib, indent=1))
    rs = np.random.RandomState(1); cases_fx = []
    for i in rs.choice(len(Ph), 12, replace=False):
        cases_fx.append({"probs": [round(float(x), 6) for x in Ph[i]], "qhat": qhat, "expected": aps_set(np.round(Ph[i], 6), qhat, classes)})
    cases_fx.append({"probs": [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.4, 0.16, 0.09], "qhat": 0.5, "expected": aps_set(np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.4, 0.16, 0.09]), 0.5, classes)})
    (REPO / "tests/fixtures/conformal_cases.json").write_text(json.dumps({"classes": classes, "force": FORCE, "cases": cases_fx}, indent=1))
    print(json.dumps({"T": [T0, T1], "item6_rule": item6["rule"], "item6_adopt": item6["adopt"], "cutoffs": [t_low, t_high],
                      "holdout": {k: h[k] for k in ("nll_before", "nll_after", "ece_before", "ece_after", "confident_but_wrong_before")},
                      "new_buckets": h["buckets_new_cutoffs_after"], "full_val": fv,
                      "item7": {"qhat": qhat, **{k: item7["holdout"][k] for k in ("coverage", "mean_size", "size_hist", "per_skin_tone", "top1_acc_by_word")},
                                "rule": item7["rule"], "adopt": item7["adopt"], "external": item7["external_brown_skin_reported_only"]}}, indent=1, default=float))


if __name__ == "__main__":
    main()
