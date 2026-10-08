"""Skin Library example photos — real, openly licensed (CC BY 4.0) images, several skin tones per condition.

Sources (redistribution allowed with attribution): SCIN (Google Research + Stanford, consented phone photos with
dermatologist labels) and PAD-UFES-20 (clinical smartphone photos of skin lesions). Non-commercial sources (DermNet
mirror, SkinDisNet) and mixed-provenance sets are NOT bundled.

  python -m ml.data.library_examples sheets            # candidates ranked by the image model → contact sheets for review
  python -m ml.data.library_examples pick eczema_atopic=3,7,12 tinea=0,4,9 …   # export the reviewed choices

Candidates: SCIN cases whose dermatologist-weighted top label is the condition (weight ≥ 0.5), no genital/buttock
photos; PAD-UFES-20 for moles / suspicious spots. Ranked by the shipped image model's probability for the category
(typical-looking examples first), round-robin over Fitzpatrick groups so light, medium and dark skin all appear.
Every pick is checked by eye on the contact sheet (no faces, eyes, tattoos, jewellery, text).
"""
from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

from ml.common.paths import MODELS
from ml.cv.dataset import eval_transform
from ml.cv.eval_cv import load_model

REPO = Path(__file__).resolve().parents[2]
SCIN = Path.home() / "skinnova-data/data/raw/scin"
PAD = Path.home() / "skinnova-data/data/raw/pad_ufes20/x_pad"
OUT_DIR = REPO / "reports" / "library"
RES = REPO / "android/app/src/main/res/drawable-nodpi"
CREDITS = REPO / "android/app/src/main/assets/library/credits.json"
SCIN_LABELS = {
    "eczema_atopic": ["Eczema"], "contact_dermatitis": ["Allergic Contact Dermatitis", "Irritant Contact Dermatitis", "CD - Contact dermatitis"],
    "seborrheic_dermatitis": ["Seborrheic Dermatitis"], "tinea": ["Tinea"], "scabies": ["Scabies"], "acne": ["Acne"],
    "psoriasis": ["Psoriasis"], "vitiligo": ["Vitiligo"],
}
PAD_LABELS = {"benign_lesion": ["NEV", "SEK"], "suspicious_lesion": ["MEL", "BCC", "SCC"]}
CV_ALIAS = {"seborrheic_dermatitis": "other"}   # not a CV class: rank by "other", review by eye


def fst_group(v) -> str:
    s = str(v)
    for g, ks in (("fitz_1_2", ("FST1", "FST2")), ("fitz_3_4", ("FST3", "FST4")), ("fitz_5_6", ("FST5", "FST6"))):
        if any(k in s for k in ks):
            return g
    return "unknown"


def _readable(p) -> bool:
    try:
        with Image.open(p) as im:
            im.verify()
        return True
    except Exception:   # a few SCIN downloads are truncated
        return False


def candidates() -> pd.DataFrame:
    cases = pd.read_csv(SCIN / "scin_cases.csv", dtype={"case_id": str}); labels = pd.read_csv(SCIN / "scin_labels.csv", dtype={"case_id": str})
    df = cases.merge(labels, on="case_id")
    rows = []
    for _, r in df.iterrows():
        try:
            w = ast.literal_eval(r.weighted_skin_condition_label) if isinstance(r.weighted_skin_condition_label, str) else {}
        except (ValueError, SyntaxError):
            continue
        if not w:
            continue
        top, wt = max(w.items(), key=lambda kv: kv[1])
        key = next((k for k, names in SCIN_LABELS.items() if top in names), None)
        if key is None or wt < 0.5 or r.body_parts_genitalia_or_groin == "YES" or r.body_parts_buttocks == "YES":
            continue
        for i in (1, 2, 3):
            p = r.get(f"image_{i}_path")
            p = SCIN / "images" / Path(p).name if isinstance(p, str) else None   # paths are "dataset/images/<id>.png"
            if p is not None and p.exists() and _readable(p):
                rows.append({"key": key, "path": str(p), "source": "SCIN", "id": f"{r.case_id}/{i}", "label": top,
                             "tone": fst_group(r.dermatologist_fitzpatrick_skin_type_label_1), "shot": r.get(f"image_{i}_shot_type")})
    meta = pd.read_csv(PAD / "metadata.csv")
    files = {f.name: f for f in PAD.rglob("*.png")}
    for _, r in meta.iterrows():
        key = next((k for k, d in PAD_LABELS.items() if r.diagnostic in d), None)
        hits = [files[r.img_id]] if r.img_id in files else []
        if r.region in ("FACE", "NOSE", "LIP", "EAR"):   # no face close-ups in the app
            continue
        if key and hits:
            ft = r.get("fitspatrick")
            tone = {1: "fitz_1_2", 2: "fitz_1_2", 3: "fitz_3_4", 4: "fitz_3_4", 5: "fitz_5_6", 6: "fitz_5_6"}.get(int(ft) if pd.notna(ft) else 0, "unknown")
            rows.append({"key": key, "path": str(hits[0]), "source": "PAD-UFES-20", "id": r.img_id, "label": r.diagnostic, "tone": tone, "shot": "CLOSE_UP"})
    return pd.DataFrame(rows)


@torch.no_grad()
def score(df: pd.DataFrame) -> pd.DataFrame:
    model, ck = load_model(MODELS / "cv" / "ckpt" / "best.pt"); classes = ck["classes"]; T = float(ck["temperature"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"; model = model.to(dev); tf = eval_transform(ck.get("color_constancy"))
    ps = []
    for i in range(0, len(df), 32):
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p in df.path.iloc[i:i + 32]]).to(dev)
        ps.append(torch.softmax(model(x).float() / T, 1).cpu().numpy())
    P = np.concatenate(ps)
    df = df.copy(); df["p"] = [P[j, classes.index(CV_ALIAS.get(k, k))] for j, k in enumerate(df.key)]
    return df


def ranked(g: pd.DataFrame, n: int = 18) -> pd.DataFrame:
    """Round-robin over skin-tone groups, each group by descending p (one image per case first)."""
    g = g.sort_values("p", ascending=False).drop_duplicates(subset=["id"]).copy()
    g["case"] = g.id.str.split("/").str[0]; g = g.drop_duplicates("case")
    buckets = [b for _, b in g.groupby("tone")]
    out, i = [], 0
    while len(out) < n and any(i < len(b) for b in buckets):
        for b in buckets:
            if i < len(b) and len(out) < n:
                out.append(b.iloc[i])
        i += 1
    return pd.DataFrame(out).reset_index(drop=True)


def sheets():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = score(candidates())
    allc = {}
    for key, g in df.groupby("key"):
        r = ranked(g); allc[key] = r.to_dict("records")
        W = 200; sheet = Image.new("RGB", (W * 6, (W + 18) * ((len(r) + 5) // 6)), "white"); d = ImageDraw.Draw(sheet)
        for j, row in r.iterrows():
            im = Image.open(row.path).convert("RGB"); s = min(im.size)
            im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2)).resize((W, W))
            x, y = (j % 6) * W, (j // 6) * (W + 18); sheet.paste(im, (x, y))
            d.text((x + 3, y + W + 2), f"{j} {row.tone} p{row.p:.2f}", fill="black")
        sheet.save(OUT_DIR / f"sheet_{key}.jpg", quality=80)
        print(key, len(g), "→ sheet", len(r))
    (OUT_DIR / "candidates.json").write_text(json.dumps(allc, indent=1, default=str))


def pick(args: list[str]):
    allc = json.loads((OUT_DIR / "candidates.json").read_text())
    credits = {}
    for a in args:
        key, idx = a.split("="); credits[key] = []
        for n, j in enumerate(int(x) for x in idx.split(",")):
            row = allc[key][j]
            im = Image.open(row["path"]).convert("RGB"); s = min(im.size)
            im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2)).resize((480, 480), Image.LANCZOS)
            name = f"lib_{key}_{n}"; im.save(RES / f"{name}.webp", "WEBP", quality=82)
            credits[key].append({"res": name, "source": row["source"], "id": row["id"], "label": row["label"], "tone": row["tone"],
                                 "license": "CC BY 4.0"})
    CREDITS.parent.mkdir(parents=True, exist_ok=True)
    CREDITS.write_text(json.dumps({"note": "Example photos: SCIN (Google Research & Stanford Medicine) and PAD-UFES-20 (Pacheco et al., 2020), "
                                           "both CC BY 4.0. Centre-cropped and resized.", "photos": credits}, indent=1, ensure_ascii=False))
    print(json.dumps({k: len(v) for k, v in credits.items()}))


if __name__ == "__main__":
    if sys.argv[1] == "sheets":
        sheets()
    else:
        pick(sys.argv[2:])
