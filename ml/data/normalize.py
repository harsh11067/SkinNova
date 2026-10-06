"""normalize: EXIF-orientation fix → RGB → long side 512 → JPEG q95 (metadata stripped) → hashes.

Input : data/processed/sources.csv  (path, source, source_label, group_id, + optional metadata columns)
Output: data/processed/images/<sha1>.jpg and data/processed/manifest.csv (+ sha1, phash, dhash, phash_dih, w, h)

phash_dih = min pHash over the 8 dihedral transforms (flip/90° rotations) so mirrored/rotated augmented
copies collide in dedupe (several Kaggle sets ship pre-augmented copies).
"""
from __future__ import annotations

import argparse
import hashlib
import io
from concurrent.futures import ProcessPoolExecutor

import imagehash
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from ml.common.paths import IMAGES, PROCESSED, REPO

LONG_SIDE = 512


def dihedral(im: Image.Image):
    for flip in (False, True):
        base = ImageOps.mirror(im) if flip else im
        for rot in (0, 90, 180, 270):
            yield base.rotate(rot, expand=True) if rot else base


def process(row: dict) -> dict | None:
    try:
        with Image.open(row["path"]) as im0:
            im = ImageOps.exif_transpose(im0).convert("RGB")
    except Exception as e:  # unreadable / truncated file
        return {**row, "error": f"{type(e).__name__}: {e}"[:200]}
    w, h = im.size
    if min(w, h) < 64:
        return {**row, "error": f"too_small {w}x{h}"}
    s = LONG_SIDE / max(w, h)
    if s < 1:
        im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=95)  # new file: no EXIF / GPS carried over
    data = buf.getvalue()
    sha1 = hashlib.sha1(data).hexdigest()
    out = IMAGES / f"{sha1}.jpg"
    if not out.exists():
        out.write_bytes(data)
    small = im.copy(); small.thumbnail((256, 256))
    ph = imagehash.phash(small)
    dih = min(int(str(imagehash.phash(t)), 16) for t in dihedral(small))
    return {**row, "sha1": sha1, "img": str(out.relative_to(REPO)) if out.is_relative_to(REPO) else str(out),
            "phash": str(ph), "dhash": str(imagehash.dhash(small)), "phash_dih": f"{dih:016x}",
            "w": w, "h": h, "error": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(PROCESSED / "labeled.csv")  # only labeled images are normalized
    rows = src.to_dict("records")
    # incremental: reuse hashes for paths already normalized (output file still present); label columns come fresh
    prev = {}
    if (PROCESSED / "manifest.csv").exists():
        old = pd.read_csv(PROCESSED / "manifest.csv")
        keep_cols = ["sha1", "img", "phash", "dhash", "phash_dih", "w", "h"]
        prev = {r["path"]: {k: r[k] for k in keep_cols} for r in old.to_dict("records") if (REPO / r["img"]).exists()}
    todo = [r for r in rows if r["path"] not in prev]
    print(f"normalize: {len(rows) - len(todo)} reused, {len(todo)} to process", flush=True)
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(tqdm(ex.map(process, todo, chunksize=32), total=len(todo), desc="normalize"))
    res += [{**r, **prev[r["path"]], "error": ""} for r in rows if r["path"] in prev]
    df = pd.DataFrame([r for r in res if r])
    bad = df[df.error.fillna("") != ""]
    df = df[df.error.fillna("") == ""].drop(columns=["error"])
    df.to_csv(PROCESSED / "manifest.csv", index=False)
    bad.to_csv(PROCESSED / "normalize_errors.csv", index=False)
    print(f"ok={len(df)} errors={len(bad)} unique_sha1={df.sha1.nunique()}")


if __name__ == "__main__":
    main()
