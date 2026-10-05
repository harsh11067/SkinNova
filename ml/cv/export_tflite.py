"""Export the CV classifier to LiteRT .tflite + parity (test.md C3) + Kotlin preprocessing fixture (C4).

Runs in the isolated export venv (.venv-export: CPU torch 2.11, litert-torch 0.9.4, ai-edge-litert 2.2.0 — the
same LiteRT version as the app's com.google.ai.edge.litert:litert:2.2.0):

  .venv-export/bin/python -m ml.cv.export_tflite

Writes
  models/tflite/skin_cls.tflite                       (fp32; fp16/int8 only if their parity also passes — C3)
  android/app/src/main/assets/cv/{skin_cls.tflite, preprocess.json}, assets/model_manifest.json "cv" block
  reports/cv_parity.json                              (C3: max |Δprob| ≤ 0.01, top-1 agreement ≥ 99.5 % on 200 val)
  tests/fixtures/cv_preproc/*                         (C4: decoded RGB + exact Python input tensor for JVM tests)
  android/app/src/androidTest/assets/cv_fixtures/*    (C5: 20 images + expected probabilities for the phone)
"""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import torch
from PIL import Image

from ml.common.paths import ANDROID_ASSETS, FIXTURES, MODELS, REPO, REPORTS, SPLITS, report_meta
from ml.cv.dataset import MEAN, SIZE, STD, center_square
from ml.cv.eval_cv import load_model


class NHWC(torch.nn.Module):
    """App feeds [1,384,384,3] float32 normalised RGB (preprocess.json); model wants NCHW."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x.permute(0, 3, 1, 2))


def preprocess(path) -> np.ndarray:
    """Exactly dataset.eval_transform, as NHWC float32 (normalize long side ≤ 512 already holds for processed images)."""
    im = Image.open(REPO / path).convert("RGB")
    im = center_square(im).resize((SIZE, SIZE), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    return ((x - np.array(MEAN, np.float32)) / np.array(STD, np.float32))[None]


def sha(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    import litert_torch
    from ai_edge_litert.interpreter import Interpreter
    ck_path = MODELS / "cv" / "ckpt" / "best.pt"
    model, ck = load_model(ck_path)
    model = model.float().eval()
    T = float(ck.get("temperature", 1.0))
    wrapped = NHWC(model).eval()
    sample = (torch.from_numpy(preprocess(pd.read_csv(SPLITS / "val.csv").img.iloc[0])),)
    out_dir = MODELS / "tflite"; out_dir.mkdir(parents=True, exist_ok=True)
    tfl = out_dir / "skin_cls.tflite"
    litert_torch.convert(wrapped, sample).export(str(tfl))

    # ---- C3 parity on 200 val images
    va = pd.read_csv(SPLITS / "val.csv")
    va = va[va.label.isin(ck["classes"])].sample(200, random_state=3407)
    interp = Interpreter(model_path=str(tfl)); interp.allocate_tensors()
    inp, outp = interp.get_input_details()[0], interp.get_output_details()[0]
    maxd, agree, rows = 0.0, 0, []
    with torch.no_grad():
        for p in va.img:
            x = preprocess(p)
            pt = torch.softmax(wrapped(torch.from_numpy(x)) / T, 1).numpy()[0]
            interp.set_tensor(inp["index"], x); interp.invoke()
            tl = torch.softmax(torch.from_numpy(interp.get_tensor(outp["index"])) / T, 1).numpy()[0]
            d = float(np.abs(pt - tl).max()); maxd = max(maxd, d); agree += int(pt.argmax() == tl.argmax())
            rows.append({"img": p, "max_abs_dprob": d})
    tfl_sha = sha(tfl)
    rep = {**report_meta(model_sha=tfl_sha[:16]), "ckpt_sha": sha(ck_path)[:16], "tflite_sha256": tfl_sha, "tflite_bytes": tfl.stat().st_size,
           "precision": "fp32", "n": len(va), "max_abs_dprob": maxd, "top1_agreement": agree / len(va),
           "input": inp["shape"].tolist(), "input_dtype": str(inp["dtype"]), "output": outp["shape"].tolist()}
    rep["C3_pass"] = bool(maxd <= 0.01 and rep["top1_agreement"] >= 0.995)
    (REPORTS / "cv_parity.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ["max_abs_dprob", "top1_agreement", "C3_pass", "tflite_bytes"]}))
    assert rep["C3_pass"], "C3 parity failed — do not ship this .tflite"

    # ---- ship to the app
    cv_assets = ANDROID_ASSETS / "cv"; cv_assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(tfl, cv_assets / "skin_cls.tflite")
    from ml.data.make_splits import dataset_rev
    pre = {"size": SIZE, "mean": list(MEAN), "std": list(STD), "temperature": T, "classes": ck["classes"], "layout": "NHWC",
           "normalize_long_side": 512, "model_sha256": tfl_sha, "dataset_rev": dataset_rev(), "arch": ck["arch"]}
    (cv_assets / "preprocess.json").write_text(json.dumps(pre, indent=1))
    mf = ANDROID_ASSETS / "model_manifest.json"; m = json.loads(mf.read_text())
    m["cv"].update({"file": "cv/skin_cls.tflite", "input": [1, SIZE, SIZE, 3], "temperature": T, "sha256": tfl_sha, "classes": ck["classes"]})
    mf.write_text(json.dumps(m, indent=2) + "\n")

    # ---- C4 fixture (JVM): decoded pixels + exact Python tensor, 3 images
    fx = FIXTURES / "cv_preproc"; shutil.rmtree(fx, ignore_errors=True); fx.mkdir(parents=True)
    meta = []
    for i, p in enumerate(va.img.iloc[:3]):
        im = Image.open(REPO / p).convert("RGB")
        (fx / f"img{i}.rgb").write_bytes(np.asarray(im, np.uint8).tobytes())
        (fx / f"img{i}.f32").write_bytes(preprocess(p).astype("<f4").tobytes())
        meta.append({"name": f"img{i}", "w": im.width, "h": im.height})
    (fx / "index.json").write_text(json.dumps({"size": SIZE, "mean": list(MEAN), "std": list(STD), "images": meta}, indent=1))

    # ---- C5 fixture (phone): 20 JPEGs + expected calibrated probabilities from the .tflite
    c5 = REPO / "android/app/src/androidTest/assets/cv_fixtures"; shutil.rmtree(c5, ignore_errors=True); c5.mkdir(parents=True)
    exp = []
    for i, p in enumerate(va.img.iloc[:20]):
        shutil.copyfile(REPO / p, c5 / f"{i:02d}.jpg")
        x = preprocess(p); interp.set_tensor(inp["index"], x); interp.invoke()
        pr = torch.softmax(torch.from_numpy(interp.get_tensor(outp["index"])) / T, 1).numpy()[0]
        exp.append({"file": f"{i:02d}.jpg", "probs": dict(zip(ck["classes"], map(float, pr)))})
    (c5 / "expected.json").write_text(json.dumps(exp, indent=1))
    print("shipped", cv_assets / "skin_cls.tflite", tfl.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
