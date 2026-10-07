"""C6 robustness (test.md §4): top-1 / top-3 on val images under phone-like corruptions, as drops from clean.

Corruptions: Gaussian blur σ=2, JPEG q=40, brightness ×0.7 and ×1.3, rotation 15° (corners filled with the image's
median border colour, as a tilted phone shot would show skin rather than black). The quality-gate half of C6 lives in
ml/eval/quality_gate.py. Stratified 100 val images per class (fixed seed) → reports/cv_robustness.json.

  python -m ml.cv.robustness
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

from ml.common.paths import MODELS, REPO, REPORTS, SPLITS, dataset_rev, report_meta
from ml.cv.dataset import eval_transform
from ml.cv.eval_cv import load_model


def _border_median(im: Image.Image) -> tuple:
    a = np.asarray(im)
    edge = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    return tuple(int(v) for v in np.median(edge, 0))


def _jpeg(im, q):
    b = io.BytesIO(); im.save(b, "JPEG", quality=q); b.seek(0)
    return Image.open(b).convert("RGB")


def _clean(im):
    return im


def _blur2(im):
    return im.filter(ImageFilter.GaussianBlur(2))


def _jpeg40(im):
    return _jpeg(im, 40)


def _dark(im):
    return ImageEnhance.Brightness(im).enhance(0.7)


def _bright(im):
    return ImageEnhance.Brightness(im).enhance(1.3)


def _rot15(im):
    return im.rotate(15, resample=Image.BICUBIC, fillcolor=_border_median(im))


# named functions (not lambdas): DataLoader workers start via forkserver and receive the corruption by name
CORRUPTIONS = {"clean": _clean, "blur_sigma2": _blur2, "jpeg_q40": _jpeg40, "brightness_0.7": _dark,
               "brightness_1.3": _bright, "rotate_15": _rot15}


class _DS(Dataset):
    def __init__(self, paths, y, name, cc=None):
        self.paths, self.y, self.name, self.tf = paths, y, name, eval_transform(cc)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(CORRUPTIONS[self.name](Image.open(self.paths[i]).convert("RGB"))), self.y[i]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS / "cv" / "ckpt" / "best.pt"))
    ap.add_argument("--per-class", type=int, default=100)
    ap.add_argument("--tag", default="", help="report suffix, e.g. _v2")
    a = ap.parse_args()
    model, ck = load_model(a.ckpt); classes = ck["classes"]; T = ck.get("temperature", 1.0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model = model.to(dev)
    df = pd.read_csv(SPLITS / "val.csv"); df = df[df.label.isin(classes)]
    df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), a.per_class), random_state=3407)).reset_index(drop=True)
    idx = {k: i for i, k in enumerate(classes)}
    paths = [str(REPO / p) for p in df.img]; y = np.array([idx[k] for k in df.label])
    out = {}
    for name in CORRUPTIONS:
        P = []
        for x, _ in DataLoader(_DS(paths, y.tolist(), name, ck.get("color_constancy")), 64, num_workers=4, multiprocessing_context="forkserver"):
            P.append((model(x.to(dev)).float() / T).softmax(1).cpu())
        P = torch.cat(P).numpy()
        top1 = float((P.argmax(1) == y).mean()); top3 = float((np.argsort(-P, 1)[:, :3] == y[:, None]).any(1).mean())
        out[name] = {"top1": round(top1, 4), "top3": round(top3, 4)}
        print(name, out[name], flush=True)
    for name in out:
        out[name]["top1_drop_pts"] = round(100 * (out["clean"]["top1"] - out[name]["top1"]), 2)
        out[name]["top3_drop_pts"] = round(100 * (out["clean"]["top3"] - out[name]["top3"]), 2)
    sha = hashlib.sha256(open(a.ckpt, "rb").read()).hexdigest()[:16]
    rep = {**report_meta(dataset_rev=dataset_rev(), model_sha=sha), "n": int(len(y)), "per_class": a.per_class, "temperature": T,
           "corruptions": out, "quality_gate": "reports/quality_gate.json (C6 gate half)"}
    (REPORTS / f"cv_robustness{a.tag}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: (v["top3"], v["top3_drop_pts"]) for k, v in out.items()}))


if __name__ == "__main__":
    main()
