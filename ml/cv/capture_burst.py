"""v2.1 item 9 (offline): what does best-of-burst capture buy under hand-shake blur?

Each photo gets several "frames" with a random Gaussian blur σ ~ U(0, 3) (phone hand-shake / focus misses); the app keeps
the frame with the highest variance of Laplacian (QualityGate's own blur measure, 256-px grey). Compared on a fixed
stratified val subset with the shipped model (single view): clean, single frame at σ 2, single random frame, best of 3
(what the app takes), best of 5.

  python -m ml.cv.capture_burst      → reports/capture_burst.json
"""
from __future__ import annotations

import json

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter

from ml.common.paths import MODELS, REPORTS, SPLITS, report_meta
from ml.cv.dataset import eval_transform
from ml.cv.eval_cv import load_model
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def lap_var(im: Image.Image) -> float:
    g = np.asarray(im.convert("L").resize((256, 256)), np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def main():
    model, ck = load_model(MODELS / "cv" / "ckpt" / "best.pt"); classes = ck["classes"]; T = float(ck["temperature"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model = model.to(dev).eval(); tf = eval_transform(ck.get("color_constancy"))
    val = pd.read_csv(SPLITS / "val.csv"); val = val[val.label.isin(classes)]
    sub = val.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), 120), random_state=3407)).reset_index(drop=True)
    rng = np.random.RandomState(3407)
    conds = {"clean": [], "single_sigma2": [], "single_random": [], "best_of_3": [], "best_of_5": []}
    Y = []
    for r in sub.itertuples():
        im = Image.open(REPO / r.img).convert("RGB"); Y.append(classes.index(r.label))
        frames = [im.filter(ImageFilter.GaussianBlur(float(s))) for s in rng.uniform(0, 3, 5)]
        scores = [lap_var(f) for f in frames]
        conds["clean"].append(im); conds["single_sigma2"].append(im.filter(ImageFilter.GaussianBlur(2.0)))
        conds["single_random"].append(frames[0])
        conds["best_of_3"].append(frames[int(np.argmax(scores[:3]))]); conds["best_of_5"].append(frames[int(np.argmax(scores))])
    Y = np.array(Y); res = {}
    with torch.no_grad():
        for k, ims in conds.items():
            P = []
            for i in range(0, len(ims), 48):
                x = torch.stack([tf(m) for m in ims[i:i + 48]]).to(dev)
                P.append(torch.softmax(model(x).float() / T, 1).cpu().numpy())
            P = np.concatenate(P); top = np.argsort(-P, 1)
            res[k] = {"top1": float((top[:, 0] == Y).mean()), "top3": float((top[:, :3] == Y[:, None]).any(1).mean())}
    rep = {**report_meta(), "n": int(len(Y)), "blur": "Gaussian σ ~ U(0, 3) per frame; sharpest by variance of Laplacian (256-px grey)",
           "results": res, "top3_drop_vs_clean": {k: res["clean"]["top3"] - v["top3"] for k, v in res.items()}}
    (REPORTS / "capture_burst.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep["results"], indent=1)); print(json.dumps(rep["top3_drop_vs_clean"], indent=1))


if __name__ == "__main__":
    main()
