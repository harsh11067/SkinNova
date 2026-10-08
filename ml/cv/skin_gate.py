"""Skin-photo gate: is this a photo of skin at all? (non-skin photos must not get skin-condition results)

A logistic head on the shipped image model's pooled features (EfficientNet-B0, 1280-d, the app's exact preprocessing).
  positives: TRAIN-split skin photos (all sources; PacificRM Unknown_Normal excluded — it mixes objects with faces)
  negatives: Imagenette train (ImageNet subset: animals, objects, buildings, vehicles — non-commercial research licence)
Threshold chosen on VAL only: wrongly reject ≤ 1 % of val skin photos overall and ≤ 2 % of SCIN phone photos, catch as
many Imagenette-val photos as possible. Reported, never tuned on: external_test (brown-skin phone photos) acceptance and
the mixed Unknown_Normal val folder. Writes models/cv/skin_gate.npz (w, b, threshold) + reports/skin_gate.json.

  python -m ml.cv.skin_gate                  # v1 (Imagenette negatives)
  python -m ml.cv.skin_gate --version v2     # v2: + COCO val2017 photos without people + DTD textures (decisions 2026-10-08)

v2 negatives are split 70/30 train/val by a hash of the file name. DTD "freckled" and "wrinkled" are left out (most are
close-ups of human faces = skin). The adoption rule compares v1 and v2 on the pooled v2 val negatives at their own thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader, Dataset

from ml.common.paths import MODELS, REPORTS, SPLITS, report_meta
from ml.cv.dataset import eval_transform
from ml.cv.eval_cv import load_model

IMAGENETTE = Path.home() / "skinnova-data/data/raw/imagenette/x/imagenette2-320"
COCO = Path.home() / "skinnova-data/data/raw/coco"
DTD = Path.home() / "skinnova-data/data/raw/dtd/dtd/images"
DTD_SKIN_LIKE = {"freckled", "wrinkled"}


def _is_val(path: str) -> bool:
    return int(hashlib.sha1(Path(path).name.encode()).hexdigest()[:8], 16) % 10 < 3


def coco_no_person() -> list[str]:
    ann = json.loads((COCO / "annotations" / "instances_val2017.json").read_text())
    person = {c["id"] for c in ann["categories"] if c["name"] == "person"}
    with_person = {a["image_id"] for a in ann["annotations"] if a["category_id"] in person}
    return sorted(str(COCO / "val2017" / im["file_name"]) for im in ann["images"] if im["id"] not in with_person)


def dtd_images() -> list[str]:
    return sorted(str(p) for p in DTD.rglob("*.jpg") if p.parent.name not in DTD_SKIN_LIKE)
MAX_FALSE_REJECT, MAX_FALSE_REJECT_SCIN = 0.01, 0.02


class Paths(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = list(paths), tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.open(self.paths[i]).convert("RGB"))


@torch.no_grad()
def features(model, paths, cc=None) -> np.ndarray:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(dev)
    out = []
    # 1 worker: shares the 7.8 GB box with the LLM evaluations
    for x in DataLoader(Paths(paths, eval_transform(cc)), 64, num_workers=1, multiprocessing_context="forkserver"):
        f = model.forward_head(model.forward_features(x.to(dev)), pre_logits=True)
        out.append(f.float().cpu().numpy())
    return np.concatenate(out)


def skin_rows(split: str, rng_seed: int, n: int | None) -> pd.DataFrame:
    df = pd.read_csv(SPLITS / f"{split}.csv")
    df = df[df.source_label != "Unknown_Normal"]
    if n and len(df) > n:
        df = df.groupby("source", group_keys=False).apply(lambda g: g.sample(min(len(g), max(1, round(n * len(g) / len(df)))), random_state=rng_seed))
    return df


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--version", default="v1", choices=["v1", "v2"]); args = ap.parse_args()
    model, ck = load_model(MODELS / "cv" / "ckpt" / "best.pt"); cc = ck.get("color_constancy")
    repo = Path(__file__).resolve().parents[2]
    abs_ = lambda s: [str(repo / p) for p in s]
    tr_skin = skin_rows("train", 3407, 6000); va_skin = skin_rows("val", 3407, None)
    va_all = pd.read_csv(SPLITS / "val.csv"); va_unk = va_all[va_all.source_label == "Unknown_Normal"]
    ext = pd.read_csv(SPLITS / "external_test.csv")
    neg_tr = sorted(str(p) for p in (IMAGENETTE / "train").rglob("*.JPEG")); neg_va = sorted(str(p) for p in (IMAGENETTE / "val").rglob("*.JPEG"))
    neg_src = {"imagenette_val": list(neg_va)}   # copy: neg_va is extended below
    if args.version == "v2":
        for name, paths in (("coco", coco_no_person()), ("dtd", dtd_images())):
            tr = [p for p in paths if not _is_val(p)]; va = [p for p in paths if _is_val(p)]
            neg_tr += tr; neg_va += va; neg_src[f"{name}_val"] = va
            print(name, "train", len(tr), "val", len(va), flush=True)
    print("train skin", len(tr_skin), "| imagenette train", len(neg_tr), "| val skin", len(va_skin), "| imagenette val", len(neg_va), flush=True)
    F = {k: features(model, v, cc) for k, v in [("tr_skin", abs_(tr_skin.img)), ("tr_neg", neg_tr), ("va_skin", abs_(va_skin.img)), ("va_neg", neg_va),
                                                 ("va_unk", abs_(va_unk.img)), ("ext", abs_(ext.img))]}
    X = np.concatenate([F["tr_skin"], F["tr_neg"]]); y = np.r_[np.ones(len(F["tr_skin"])), np.zeros(len(F["tr_neg"]))]
    mu, sd = X.mean(0), X.std(0) + 1e-6                                  # standardise (folded into w, b below)
    clf = LogisticRegression(C=0.5, max_iter=4000, class_weight="balanced").fit((X - mu) / sd, y)
    w = clf.coef_[0] / sd; b = float(clf.intercept_[0] - (clf.coef_[0] * mu / sd).sum())
    p = {k: 1 / (1 + np.exp(-(v @ w + b))) for k, v in F.items()}       # p(skin photo)
    scin = (va_skin.source == "scin").values
    best = None
    for t in np.linspace(0.01, 0.99, 197):
        fr, fr_scin, catch = np.mean(p["va_skin"] < t), np.mean(p["va_skin"][scin] < t), np.mean(p["va_neg"] < t)
        if fr <= MAX_FALSE_REJECT and fr_scin <= MAX_FALSE_REJECT_SCIN and (best is None or catch > best[3]):
            best = (float(t), fr, fr_scin, catch)
    t = best[0]
    # per-source catch on the val negatives, and (v2) the pre-registered v1-vs-v2 comparison on the pooled v2 val negatives
    idx, per_src = 0, {}
    for k, v in neg_src.items():
        per_src[k] = float(np.mean(p["va_neg"][idx:idx + len(v)] < t)); idx += len(v)
    cmp = None
    if args.version == "v2":
        g1 = np.load(MODELS / "cv" / "skin_gate_v1.npz"); p1 = 1 / (1 + np.exp(-(F["va_neg"] @ g1["w"] + float(g1["b"]))))
        v1c = float(np.mean(p1 < float(g1["threshold"]))); v2c = float(np.mean(p["va_neg"] < t))
        cmp = {"pooled_val_negatives": len(F["va_neg"]), "v1_threshold": float(g1["threshold"]), "v1_caught": v1c, "v2_caught": v2c,
               "rule": "adopt v2 iff v2_caught >= v1_caught + 0.02", "adopt_v2": v2c >= v1c + 0.02,
               "v1_skin_false_reject_val": float(np.mean(1 / (1 + np.exp(-(F["va_skin"] @ g1["w"] + float(g1["b"])))) < float(g1["threshold"])))}
    rep = {**report_meta(model_sha=ck.get("arch", "")), "version": args.version, "per_source_val_caught": per_src, "v1_vs_v2": cmp,
           "rule": {"val_false_reject_max": MAX_FALSE_REJECT, "scin_false_reject_max": MAX_FALSE_REJECT_SCIN},
           "threshold": t, "n": {k: int(len(v)) for k, v in F.items()},
           "val": {"skin_false_reject": best[1], "scin_false_reject": best[2], "nonskin_caught": best[3]},
           "train": {"skin_false_reject": float(np.mean(p["tr_skin"] < t)), "nonskin_caught": float(np.mean(p["tr_neg"] < t))},
           "reported_only": {"external_brown_skin_phone_accepted": float(np.mean(p["ext"] >= t)),
                             "unknown_normal_val_flagged": float(np.mean(p["va_unk"] < t))},
           "per_source_val_false_reject": {s: float(np.mean(p["va_skin"][(va_skin.source == s).values] < t)) for s in sorted(va_skin.source.unique())}}
    (MODELS / "cv").mkdir(parents=True, exist_ok=True)
    tag = "" if args.version == "v1" else "_v2"
    np.savez(MODELS / "cv" / f"skin_gate{tag}.npz", w=w.astype(np.float32), b=np.float32(b), threshold=np.float32(t))
    (REPORTS / f"skin_gate{tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ("threshold", "val", "train", "reported_only", "per_source_val_false_reject", "per_source_val_caught", "v1_vs_v2")}, indent=1))


if __name__ == "__main__":
    main()
