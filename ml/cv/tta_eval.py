"""Test-time augmentation (TTA) for the shipped image model — pre-registered rule in docs/decisions.md (2026-10-08).

Views are computed on the app's exact eval preprocessing (centre square, 384 px), then averaged as calibrated
probabilities. VAL only: test / external are never touched here.

Also reports (no selection uses it) the "add a second photo" effect: SCIN val cases with ≥ 2 photos, scored with one
photo vs the mean of two — what the result screen's "Add another photo" does on the phone.

  python -m ml.cv.tta_eval            → reports/cv_tta.json
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F

from ml.common.paths import MODELS, REPORTS, report_meta
from ml.cv.eval_cv import load_model
from ml.cv.dataset import SkinDS, eval_transform
from torch.utils.data import DataLoader
import pandas as pd
from ml.common.paths import SPLITS

VIEWS = {
    "id": lambda x: x,
    "h": lambda x: x.flip(-1),
    "v": lambda x: x.flip(-2),
    "r180": lambda x: x.flip(-1).flip(-2),
    "r90": lambda x: x.rot90(1, (-2, -1)),
    "r270": lambda x: x.rot90(3, (-2, -1)),
    "t": lambda x: x.transpose(-2, -1),
    "at": lambda x: x.rot90(1, (-2, -1)).flip(-1),
    "z80": lambda x: F.interpolate(x[..., int(.1 * x.shape[-2]):int(.9 * x.shape[-2]), int(.1 * x.shape[-1]):int(.9 * x.shape[-1])],
                                   size=x.shape[-2:], mode="bilinear", align_corners=False),
}
VARIANTS = {"base": ["id"], "V2": ["id", "h"], "Z2": ["id", "z80"], "V4": ["id", "h", "v", "r180"],
            "V8": ["id", "h", "v", "r180", "r90", "r270", "t", "at"]}


@torch.no_grad()
def view_logits(model, df, classes, cc):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    out = {k: [] for k in VIEWS}; Y = []
    for x, y in DataLoader(SkinDS(df, classes, eval_transform(cc)), 48, num_workers=3, multiprocessing_context="forkserver"):
        x = x.to(dev)
        for k, f in VIEWS.items():
            out[k].append(model(f(x)).float().cpu())
        Y.append(y)
    return {k: torch.cat(v).numpy() for k, v in out.items()}, torch.cat(Y).numpy()


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def scores(P, Y, classes, mask=None):
    if mask is not None:
        P, Y = P[mask], Y[mask]
    top = np.argsort(-P, 1)
    sus = classes.index("suspicious_lesion")
    s = Y == sus
    conf = P.max(1)
    return {"n": int(len(Y)), "top1": float(np.mean(top[:, 0] == Y)), "top3": float(np.mean((top[:, :3] == Y[:, None]).any(1))),
            "nll": float(-np.mean(np.log(np.clip(P[np.arange(len(Y)), Y], 1e-9, 1)))),
            "suspicious_recall_p015": float(np.mean(P[s, sus] >= 0.15)) if s.any() else None,
            "confident_share_p060": float(np.mean(conf >= 0.6)),
            "acc_when_confident_p060": float(np.mean(top[conf >= 0.6, 0] == Y[conf >= 0.6])) if (conf >= 0.6).any() else None,
            "top1_other_share": float(np.mean(top[:, 0] == classes.index("other")))}


def main():
    model, ck = load_model(MODELS / "cv" / "ckpt" / "best.pt")
    classes, T, cc = ck["classes"], float(ck["temperature"]), ck.get("color_constancy")
    df = pd.read_csv(SPLITS / "val.csv"); df = df[df.label.isin(classes)].reset_index(drop=True)
    L, Y = view_logits(model, df, classes, cc)
    probs = {k: softmax(v / T) for k, v in L.items()}
    scin = (df.source == "scin").values
    rep = {**report_meta(model_sha=ck.get("arch", "")), "temperature": T, "variants": {}}
    for name, views in VARIANTS.items():
        P = np.mean([probs[v] for v in views], 0)
        rep["variants"][name] = {"views": views, "full_val": scores(P, Y, classes), "scin_val": scores(P, Y, classes, scin)}
    b = rep["variants"]["base"]
    passing = []
    for name, v in rep["variants"].items():
        if name == "base":
            continue
        ok = {"top1": v["full_val"]["top1"] >= b["full_val"]["top1"] + 0.005,
              "scin_top3": v["scin_val"]["top3"] >= b["scin_val"]["top3"],
              "sus_recall": v["full_val"]["suspicious_recall_p015"] >= b["full_val"]["suspicious_recall_p015"] - 0.005,
              "nll": v["full_val"]["nll"] <= b["full_val"]["nll"]}
        v["rule"] = ok
        if all(ok.values()):
            passing.append(name)
    passing.sort(key=lambda n: (len(VARIANTS[n]), -rep["variants"][n]["full_val"]["top1"]))
    rep["decision"] = {"adopt": passing[0] if passing else "base", "passing": passing}

    # report only: second photo of the same SCIN case (what "Add another photo" does)
    sdf = df[scin].reset_index(); sP = probs["id"][scin]; sY = Y[scin]
    one, two, n = [], [], 0
    for _, g in sdf.groupby("group"):
        if len(g) < 2 or g.label.nunique() != 1:
            continue
        i, j = g.index[0], g.index[1]
        y = sY[i]; p1 = sP[i]; p2 = (sP[i] + sP[j]) / 2
        one.append((np.argmax(p1) == y, y in np.argsort(-p1)[:3], p1.max())); two.append((np.argmax(p2) == y, y in np.argsort(-p2)[:3], p2.max()))
        n += 1
    one, two = np.array(one, float), np.array(two, float)
    rep["second_photo_scin_val"] = {"cases": n, "one_photo": {"top1": one[:, 0].mean(), "top3": one[:, 1].mean(), "mean_top_p": one[:, 2].mean()},
                                    "two_photos": {"top1": two[:, 0].mean(), "top3": two[:, 1].mean(), "mean_top_p": two[:, 2].mean()}}
    (REPORTS / "cv_tta.json").write_text(json.dumps(rep, indent=1, default=float))
    print(json.dumps({k: {"full": {m: round(v["full_val"][m], 4) for m in ("top1", "top3", "nll", "confident_share_p060", "top1_other_share")},
                          "scin": {m: round(v["scin_val"][m], 4) for m in ("top1", "top3")}, "rule": v.get("rule")}
                      for k, v in rep["variants"].items()}, indent=1))
    print(rep["decision"]); print(json.dumps(rep["second_photo_scin_val"], indent=1, default=float))


if __name__ == "__main__":
    main()
