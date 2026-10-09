"""v2.1 item 10: let the user's answers re-rank the image model (symptom fusion). Pre-registered (decisions 2026-10-09).

Leakage-safe design (the image model was trained on the train photos, so its train-set scores are in-sample):
  1. ANSWERS model: multinomial logistic regression on the app's questionnaire features only (body site, duration, itch,
     pain, bleeding, growth, colour change, fever, age band + missing flags), trained on TRAIN-split cases with real
     self-reported metadata (SCIN, PAD-UFES-20, SkinDisNet). It never sees an image score.
  2. FUSION: p ∝ p_image · (p_answers / prior)^w — one weight w fitted on scin_calib (NLL), with the shipped TTA image
     probabilities.
  3. JUDGED on scin_holdout (+ PAD-UFES val for lesions), image-only vs fused; also with 30 % of answers missing.
ADOPT iff on scin_holdout: top-1 ≥ image + 3 pts OR NLL ≤ image − 0.10; top-3 not lower; suspicious recall (p ≥ 0.15)
≥ image − 0.5 pt. Red-flag rules keep using the image model's suspicious probability (max of the two), so fusion can
never dilute a red flag.

  python -m ml.cv.fusion     → reports/fusion.json (+ android assets cv/fusion.json and a parity fixture if adopted)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ml.common.paths import MODELS, REPORTS, SPLITS, report_meta
from ml.cv.calibrate_phone import VIEWS, probs

REPO = Path(__file__).resolve().parents[2]
SITES = ["head", "arm", "hand", "trunk_front", "back", "groin", "leg", "foot", "other"]
DURS = ["lt_1w", "1_4w", "1_6m", "gt_6m"]
AGES = ["lt_12", "12_17", "18_39", "40_59", "60_plus"]
FLAGS = ["itch", "pain", "bleed", "grow", "color", "fever"]
FEATURES = ([f"site_{s}" for s in SITES] + ["site_missing"] + [f"dur_{d}" for d in DURS] + ["dur_missing"] +
            [f"{f}_yes" for f in FLAGS] + [f"{f}_missing" for f in FLAGS] + [f"age_{a}" for a in AGES] + ["age_missing"])

# ---------------- app answers → features (Kotlin twin: ml/Fusion.kt; shared fixture fusion_cases.json) ----------------
APP_SITE = {"face": "head", "scalp": "head", "neck": "head", "arm": "arm", "hand": "hand", "chest": "trunk_front", "abdomen": "trunk_front",
            "back": "back", "groin": "groin", "leg": "leg", "foot": "foot", "nails": "other", "other": "other"}


def app_features(a: dict) -> np.ndarray:
    x = dict.fromkeys(FEATURES, 0.0)
    s = APP_SITE.get(a.get("body_site")); x[f"site_{s}" if s else "site_missing"] = 1.0
    x[f"dur_{a['duration']}" if a.get("duration") in DURS else "dur_missing"] = 1.0
    vals = {"itch": (a.get("itch") or 0) >= 1, "pain": (a.get("pain") or 0) >= 1, "bleed": bool(a.get("bleeding_or_crusting")),
            "grow": a.get("changing") in ("growing", "spreading", "changing_shape"), "color": a.get("changing") == "changing_color",
            "fever": bool(a.get("fever_or_unwell"))}
    for f, v in vals.items():
        x[f"{f}_yes"] = 1.0 if v else 0.0
    x[f"age_{a['age_band']}" if a.get("age_band") in AGES else "age_missing"] = 1.0
    return np.array([x[k] for k in FEATURES])


# ---------------- dataset metadata → the same features ----------------
SCIN_SITE = {"head_or_neck": "head", "arm": "arm", "palm": "hand", "back_of_hand": "hand", "torso_front": "trunk_front", "torso_back": "back",
             "genitalia_or_groin": "groin", "buttocks": "groin", "leg": "leg", "foot_top_or_side": "foot", "foot_sole": "foot", "other": "other"}
SCIN_DUR = {"ONE_DAY": "lt_1w", "LESS_THAN_ONE_WEEK": "lt_1w", "ONE_TO_FOUR_WEEKS": "1_4w", "ONE_TO_THREE_MONTHS": "1_6m",
            "THREE_TO_TWELVE_MONTHS": "1_6m", "MORE_THAN_ONE_YEAR": "gt_6m", "MORE_THAN_FIVE_YEARS": "gt_6m"}
SCIN_AGE = {"AGE_0_TO_2": "lt_12", "AGE_3_TO_11": "lt_12", "AGE_12_TO_17": "12_17", "AGE_18_TO_29": "18_39", "AGE_30_TO_39": "18_39",
            "AGE_40_TO_49": "40_59", "AGE_50_TO_59": "40_59", "AGE_60_TO_69": "60_plus", "AGE_70_TO_79": "60_plus", "AGE_80_OR_ABOVE": "60_plus"}
PAD_SITE = {"FACE": "head", "NOSE": "head", "EAR": "head", "LIP": "head", "SCALP": "head", "NECK": "head", "FOREARM": "arm", "ARM": "arm",
            "HAND": "hand", "CHEST": "trunk_front", "ABDOMEN": "trunk_front", "BACK": "back", "THIGH": "leg", "FOOT": "foot"}
SDN_SITE = {"leg": "leg", "thigh": "leg", "knee": "leg", "hand": "hand", "finger": "hand", "palm": "hand", "foot": "foot", "toe": "foot",
            "forearm": "arm", "arm": "arm", "elbow": "arm", "belly": "trunk_front", "chest": "trunk_front", "abdomen": "trunk_front",
            "back": "back", "face": "head", "head": "head", "neck": "head", "scalp": "head", "groin": "groin", "buttock": "groin"}


def age_band(v) -> str | None:
    try:
        a = float(v)
    except (TypeError, ValueError):
        return None
    return "lt_12" if a < 12 else "12_17" if a < 18 else "18_39" if a < 40 else "40_59" if a < 60 else "60_plus"


def _tf(v):
    s = str(v).strip().upper()
    return True if s == "TRUE" else False if s == "FALSE" else None


def row_features(r) -> np.ndarray | None:
    x = dict.fromkeys(FEATURES, 0.0); src = r.source
    if src == "scin":
        parts = [] if pd.isna(r.scin_parts) else [SCIN_SITE.get(p) for p in str(r.scin_parts).split(";")]
        parts = [p for p in parts if p]
        for p in parts:
            x[f"site_{p}"] = 1.0
        if not parts: x["site_missing"] = 1.0
        d = SCIN_DUR.get(str(r.scin_duration)); x[f"dur_{d}" if d else "dur_missing"] = 1.0
        if pd.isna(r.scin_symptoms):
            for f in FLAGS[:5]: x[f"{f}_missing"] = 1.0
        else:
            sy = set(str(r.scin_symptoms).split(";"))
            x["itch_yes"] = float("itching" in sy); x["pain_yes"] = float(bool({"pain", "burning"} & sy)); x["bleed_yes"] = float("bleeding" in sy)
            x["grow_yes"] = float("increasing_size" in sy); x["color_yes"] = float("darkening" in sy)
        x["fever_yes"] = float(str(r.scin_fever) == "True")
        a = SCIN_AGE.get(str(r.scin_age)); x[f"age_{a}" if a else "age_missing"] = 1.0
    elif src == "pad_ufes20":
        s = PAD_SITE.get(str(r.pad_region)); x[f"site_{s}" if s else "site_missing"] = 1.0
        x["dur_missing"] = 1.0
        for f, col in (("itch", "pad_itch"), ("pain", "pad_hurt"), ("bleed", "pad_bleed"), ("grow", "pad_grew"), ("color", "pad_changed")):
            v = _tf(getattr(r, col)); x[f"{f}_yes" if v else f"{f}_missing" if v is None else f"{f}_yes"] = 1.0 if v or v is None else 0.0
        x["fever_missing"] = 1.0
        a = age_band(r.pad_age); x[f"age_{a}" if a else "age_missing"] = 1.0
    elif src == "skindisnet":
        loc = str(r.sdn_location).strip().lower(); s = SDN_SITE.get(loc); x[f"site_{s}" if s else "site_missing"] = 1.0
        x["dur_missing"] = 1.0
        for f in FLAGS: x[f"{f}_missing"] = 1.0
        a = age_band(r.sdn_age); x[f"age_{a}" if a else "age_missing"] = 1.0
    else:
        return None
    return np.array([x[k] for k in FEATURES])


def featurise(df):
    rows, keep = [], []
    for i, r in enumerate(df.itertuples(index=False)):
        f = row_features(r)
        if f is not None:
            rows.append(f); keep.append(i)
    return np.array(rows), np.array(keep)


def scores(P, Y, classes):
    top = np.argsort(-P, 1); sus = classes.index("suspicious_lesion"); s = Y == sus
    return {"n": int(len(Y)), "top1": float(np.mean(top[:, 0] == Y)), "top3": float(np.mean((top[:, :3] == Y[:, None]).any(1))),
            "nll": float(-np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)))),
            "suspicious_recall_p015": float(np.mean(P[s, sus] >= 0.15)) if s.any() else None}


def fuse(Pimg, Pans, prior, w):
    lp = np.log(np.clip(Pimg, 1e-12, 1)) + w * (np.log(np.clip(Pans, 1e-12, 1)) - np.log(prior))
    lp -= lp.max(1, keepdims=True); e = np.exp(lp); return e / e.sum(1, keepdims=True)


def main():
    from ml.cv.eval_cv import load_model
    _, ck = load_model(MODELS / "cv" / "ckpt" / "best.pt"); classes = ck["classes"]; T = float(ck["temperature"])
    tr = pd.read_csv(SPLITS / "train.csv"); tr = tr[tr.label.isin(classes)].reset_index(drop=True)
    Xtr, ktr = featurise(tr); ytr = np.array([classes.index(l) for l in tr.label.iloc[ktr]])
    # answers model, one row per CASE (several photos of a case carry the same answers)
    grp = tr.group.iloc[ktr].values; _, first = np.unique(grp, return_index=True)
    Xc, yc = Xtr[first], ytr[first]
    best = None
    from sklearn.model_selection import GroupKFold, cross_val_score
    for C in (0.03, 0.1, 0.3, 1.0):
        m = LogisticRegression(C=C, max_iter=5000)
        s = cross_val_score(m, Xc, yc, cv=5, scoring="neg_log_loss").mean()
        if best is None or s > best[1]: best = (C, s)
    ans = LogisticRegression(C=best[0], max_iter=5000).fit(Xc, yc)
    full_coef = np.zeros((len(classes), Xc.shape[1])); full_b = np.full(len(classes), -30.0)
    full_coef[ans.classes_] = ans.coef_; full_b[ans.classes_] = ans.intercept_
    prior = np.full(len(classes), 1e-6); cnt = np.bincount(yc, minlength=len(classes)); prior = np.maximum(cnt / cnt.sum(), 1e-6)

    def p_ans(X):
        z = X @ full_coef.T + full_b; z -= z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)

    z = np.load(MODELS / "cv" / "val_tta_logits.npz"); L = {v: z[v] for v in VIEWS}; Y = z["y"]
    val = pd.read_csv(SPLITS / "val.csv"); val = val[val.label.isin(classes)].reset_index(drop=True)
    Pimg = probs(L, T)
    Xv = np.zeros((len(val), len(FEATURES))); has = np.zeros(len(val), bool)
    for i, r in enumerate(val.itertuples(index=False)):
        f = row_features(r)
        if f is not None: Xv[i] = f; has[i] = True
    Pa = p_ans(Xv)
    cal = val.img.isin(pd.read_csv(SPLITS / "scin_calib.csv").img).values
    hold = val.img.isin(pd.read_csv(SPLITS / "scin_holdout.csv").img).values
    ws = np.linspace(0, 1.5, 61)
    nlls = [scores(fuse(Pimg[cal], Pa[cal], prior, w), Y[cal], classes)["nll"] for w in ws]
    w = float(ws[int(np.argmin(nlls))])
    Pf = fuse(Pimg, Pa, prior, w)
    res = {"holdout_image": scores(Pimg[hold], Y[hold], classes), "holdout_fused": scores(Pf[hold], Y[hold], classes)}
    pad = (val.source == "pad_ufes20").values
    res["pad_val_image"] = scores(Pimg[pad], Y[pad], classes); res["pad_val_fused"] = scores(Pf[pad], Y[pad], classes)
    # robustness: 30 % of answers missing (random fields set to missing) on the holdout
    rng = np.random.RandomState(3407); Xm = Xv.copy()
    groups = [(list(range(0, 9)), 9), (list(range(10, 14)), 14)] + [([15 + k], 21 + k) for k in range(6)] + [(list(range(27, 32)), 32)]
    for i in np.where(hold)[0]:
        for cols, miss in groups:
            if rng.rand() < 0.3:
                Xm[i, cols] = 0.0; Xm[i, miss] = 1.0
    res["holdout_fused_30pct_missing"] = scores(fuse(Pimg[hold], p_ans(Xm[hold]), prior, w), Y[hold], classes)
    hi, hf = res["holdout_image"], res["holdout_fused"]
    rule = {"top1_plus3_or_nll_minus_0.10": (hf["top1"] >= hi["top1"] + 0.03) or (hf["nll"] <= hi["nll"] - 0.10),
            "top3_not_lower": hf["top3"] >= hi["top3"],
            "suspicious_recall_ok": (hf["suspicious_recall_p015"] or 0) >= (hi["suspicious_recall_p015"] or 0) - 0.005}
    adopt = all(rule.values())
    # per confusion pair gain (eczema / tinea / scabies / contact cluster) on the holdout
    names = ["eczema_atopic", "contact_dermatitis", "tinea", "scabies", "psoriasis", "other"]
    per_class = {c: {"image_recall": float(np.mean(Pimg[hold][Y[hold] == classes.index(c)].argmax(1) == classes.index(c))) if (Y[hold] == classes.index(c)).any() else None,
                     "fused_recall": float(np.mean(Pf[hold][Y[hold] == classes.index(c)].argmax(1) == classes.index(c))) if (Y[hold] == classes.index(c)).any() else None,
                     "n": int((Y[hold] == classes.index(c)).sum())} for c in names}
    rep = {**report_meta(), "answers_model": {"train_cases": int(len(yc)), "C": best[0], "cv_neg_log_loss": best[1], "features": FEATURES},
           "w": w, "temperature": T, "results": res, "per_class_holdout": per_class, "rule": rule, "adopt": adopt}
    (REPORTS / "fusion.json").write_text(json.dumps(rep, indent=1, default=float))
    if adopt:
        (REPO / "android/app/src/main/assets/cv/fusion.json").write_text(json.dumps({
            "classes": classes, "features": FEATURES, "w": w, "coef": full_coef.round(8).tolist(), "intercept": full_b.round(8).tolist(),
            "prior": prior.round(10).tolist(), "report": "reports/fusion.json"}, indent=1))
        rs = np.random.RandomState(5); cases = []
        demo = [{"body_site": "arm", "duration": "1_4w", "itch": 2, "pain": 0, "changing": "no", "bleeding_or_crusting": False, "fever_or_unwell": False, "age_band": "18_39"},
                {"body_site": "back", "duration": "gt_6m", "itch": 0, "pain": 0, "changing": "growing", "bleeding_or_crusting": True, "fever_or_unwell": False, "age_band": "60_plus"},
                {"body_site": "foot", "duration": "1_6m", "itch": 3, "pain": 1, "changing": "spreading", "bleeding_or_crusting": False, "fever_or_unwell": False, "age_band": "40_59"}]
        for a in demo:
            pi = rs.dirichlet(np.ones(len(classes)))
            pf = fuse(pi[None], p_ans(app_features(a)[None]), prior, w)[0]
            cases.append({"answers": a, "image_probs": pi.round(8).tolist(), "fused": pf.round(8).tolist()})
        (REPO / "tests/fixtures/fusion_cases.json").write_text(json.dumps({"cases": cases}, indent=1))
    print(json.dumps({"w": w, "C": best[0], "train_cases": int(len(yc)), "results": res, "rule": rule, "adopt": adopt, "per_class": per_class}, indent=1, default=float))


if __name__ == "__main__":
    main()
