"""Dump calibrated CV probabilities per split → data/processed/cv_probs_<split>.npz (img, labels, probs, classes).

Consumers: ml/llm/build_sft_dataset.py (CV scores shown to the LLM), ml/eval (combined arms).
  python -m ml.cv.predict --splits val test external_test
"""
from __future__ import annotations

import argparse

import numpy as np

from ml.common.paths import MODELS, PROCESSED
from ml.cv.eval_cv import load_model, logits_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(MODELS / "cv" / "ckpt" / "best.pt"))
    ap.add_argument("--splits", nargs="+", default=["val", "test", "external_test"])
    ap.add_argument("--per-class", type=int, default=0, help="class-balanced subset (smoke-test data while the GPU trains)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    model, ck = load_model(a.ckpt)
    T = ck.get("temperature", 1.0)
    for s in a.splits:
        L, Y, df = logits_for(model, s, ck["classes"], per_class=a.per_class, cpu=a.cpu, cc=ck.get("color_constancy"))
        s = s + a.suffix
        P = (L / T).softmax(1).numpy().astype(np.float32)
        np.savez(PROCESSED / f"cv_probs_{s}.npz", img=df.img.values.astype(str), labels=df.label.values.astype(str),
                 probs=P, classes=np.array(ck["classes"]))
        # accuracy printed for val only: test/external numbers come from eval_cv once per final candidate (no peeking)
        print(s, P.shape, *(["val top1 acc", float((P.argmax(1) == Y.numpy()).mean())] if s == "val" else []))


if __name__ == "__main__":
    main()
