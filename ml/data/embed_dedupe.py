"""Copy-detection pass (complements pHash in dedupe.py): embedding candidates → ORB + RANSAC verification.

pHash misses crop/zoom augmentations that the Kaggle mirrors ship as separate files (leakage if one copy lands in
train and another in test). Global embeddings alone can't separate copies from look-alikes (dermoscopy images are
globally similar: measured random same-label cosine p99 ≈ 0.83 vs planted-copy p5 ≈ 0.62). So:
  1. DINOv2-small CLS embeddings → k nearest neighbours with cosine ≥ MIN_COS as candidates (cheap recall),
  2. each candidate pair verified geometrically: ORB keypoints + ratio-test matches + RANSAC similarity transform;
     a copy has many consistent inliers, a look-alike doesn't (precision).

  python -m ml.data.embed_dedupe embed       # → data/processed/emb_dinov2s.npy (row-aligned with manifest.csv)
  python -m ml.data.embed_dedupe calibrate   # planted copies vs hard negatives → reports/dedupe_calibration.json
  python -m ml.data.embed_dedupe pairs       # verified duplicate pairs → data/processed/copy_pairs.npy (read by dedupe.py)

Restart-safe and incremental (WSL may restart mid-run): embeddings are cached per image sha1 (emb_cache.npz),
ORB features per sha1 (orb/<sha1>.npz), and every verified candidate pair is appended to pair_cache.tsv after each
chunk. Re-running only computes what is missing, so adding a dataset costs only its own images.
"""
from __future__ import annotations

import json
import random
import sys

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from ml.common.paths import PROCESSED, REPO, REPORTS, report_meta

EMB = PROCESSED / "emb_dinov2s.npy"
CAL = REPORTS / "dedupe_calibration.json"
ARCH = "vit_small_patch14_dinov2.lvd142m"
TF = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC), T.CenterCrop(224), T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


def model():
    import timm
    m = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=224).eval()
    return m.cuda() if torch.cuda.is_available() else m


class _DS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        im = it if isinstance(it, Image.Image) else Image.open(REPO / it if not str(it).startswith("/") else it).convert("RGB")
        return TF(im)


@torch.no_grad()
def embed_items(m, items, bs=128, workers=8) -> np.ndarray:
    dev = next(m.parameters()).device
    out = []
    for x in DataLoader(_DS(items), bs, num_workers=workers if not isinstance(items[0], Image.Image) else 0):
        with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
            f = m(x.to(dev))
        out.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float16)


def cos_pairs(E: np.ndarray, thr: float, chunk: int = 2048):
    """Yield (i, j>i) with cosine ≥ thr. GPU chunked matmul."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(E.astype(np.float32)).to(dev)
    for s in range(0, len(X), chunk):
        S = X[s:s + chunk] @ X.T
        ii, jj = torch.nonzero(S >= thr, as_tuple=True)
        ii = ii + s
        keep = jj > ii
        yield from zip(ii[keep].tolist(), jj[keep].tolist())


def plant(im: Image.Image, rng: random.Random, kind: str | None = None) -> tuple[Image.Image, str]:
    """test.md D2: one of resize (0.5–1×), JPEG q50, crop ≤ 5% per side, brightness ±10%."""
    import io
    kind = kind or rng.choice(["resize", "jpeg", "crop", "brightness"])
    w, h = im.size
    if kind == "resize":
        f = rng.uniform(0.5, 1.0); im = im.resize((max(32, int(w * f)), max(32, int(h * f))))
    elif kind == "crop":
        c = [rng.uniform(0.0, 0.05) for _ in range(4)]
        im = im.crop((int(w * c[0]), int(h * c[1]), int(w * (1 - c[2])), int(h * (1 - c[3]))))
    elif kind == "brightness":
        im = ImageEnhance.Brightness(im).enhance(rng.choice([rng.uniform(0.9, 0.97), rng.uniform(1.03, 1.1)]))
    b = io.BytesIO(); im.save(b, "JPEG", quality=50 if kind == "jpeg" else 92); b.seek(0)
    return Image.open(b).convert("RGB"), kind


def _phashes(im: Image.Image) -> tuple[int, int]:
    import imagehash
    from ml.data.normalize import dihedral
    small = im.copy(); small.thumbnail((256, 256))
    return int(str(imagehash.phash(small)), 16), min(int(str(imagehash.phash(t)), 16) for t in dihedral(small))


K_NN = 32
MIN_COS = 0.50
ORB_SIDE = 320


def _gray(im) -> np.ndarray:
    import cv2
    if not isinstance(im, Image.Image):
        im = Image.open(REPO / im if not str(im).startswith("/") else im)
    im = im.convert("L"); im.thumbnail((ORB_SIDE, ORB_SIDE))
    return np.asarray(im)


def orb_feats(im, with_gray: bool = False):
    """ORB keypoint coordinates (N×2 float32) + descriptors (N×32 uint8) [+ the 320 px grey image they live in]."""
    import cv2
    g = _gray(im)
    orb = cv2.ORB_create(nfeatures=600, fastThreshold=10)
    k, d = orb.detectAndCompute(g, None)
    pts, desc = (np.zeros((0, 2), np.float32), np.zeros((0, 32), np.uint8)) if d is None else (np.float32([p.pt for p in k]), d)
    return (pts, desc, g) if with_gray else (pts, desc)


def _match(fa, fb):
    import cv2
    pa, da = fa[0], fa[1]; pb, db = fb[0], fb[1]
    if len(da) < 8 or len(db) < 8:
        return 0, None
    m = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in m if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]
    if len(good) < 6:
        return 0, None
    A = pa[[g.queryIdx for g in good]]; B = pb[[g.trainIdx for g in good]]
    M, inl = cv2.estimateAffinePartial2D(A, B, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return (int(inl.sum()), M) if inl is not None else (0, None)


def match_feats(fa, fb) -> int:
    """Ratio-test matches + RANSAC partial-affine inliers. Copies: dozens–hundreds; look-alikes: ~0–8."""
    return _match(fa, fb)[0]


def photometric(ga: np.ndarray, gb: np.ndarray, M) -> tuple[float, float]:
    """Warp B onto A with the RANSAC transform (M maps A→B) and correlate the overlap, ignoring near-black pixels
    (dermatoscope vignette, letterbox) → (NCC, overlap fraction). Copies stay ≥ ~0.9 under crop/resize/JPEG/brightness
    (NCC is gain/offset invariant); different lesions sharing a vignette or watermark do not."""
    import cv2
    if M is None:
        return 0.0, 0.0
    h, w = ga.shape
    bw = cv2.warpAffine(gb, M, (w, h), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderValue=0)
    valid = cv2.warpAffine(np.full_like(gb, 255), M, (w, h), flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0) > 0
    valid &= (ga > 25) & (bw > 25)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    frac = float(valid.mean())
    if valid.sum() < 500:
        return 0.0, frac
    x = ga[valid].astype(np.float64); y = bw[valid].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return (float((x * y).sum() / d) if d > 0 else 0.0), frac


def verify_full(fa, fb) -> tuple[int, float, float]:
    """(inliers, ncc, overlap) — fa/fb = (pts, desc, gray)."""
    n, M = _match(fa, fb)
    if n < 10:
        return n, 0.0, 0.0
    ncc, ov = photometric(fa[2], fb[2], M)
    return n, ncc, ov


def orb_inliers(a, b) -> int:
    """Between two images (paths or PIL)."""
    return match_feats(orb_feats(a), orb_feats(b))


ORB_MIN_NCC, ORB_MIN_OVERLAP = 0.6, 0.25   # photometric gate on ORB/RANSAC links (cross-label links: median NCC 0.0)
PHASH_MIN_NCC = 0.93                       # pHash only PROPOSES (dermoscopy collides); dihedral NCC decides
ORB_DIR = PROCESSED / "orb2"          # v2: + 320 px grey thumbnail for photometric verification
PAIR_CACHE = PROCESSED / "pair_cache_v2.tsv"     # a, b, inliers, ncc, overlap
PHASH_CACHE = PROCESSED / "phash_cache.tsv"      # a, b, dihedral ncc
EMB_CACHE = PROCESSED / "emb_cache.npz"


def _orb_job(args):
    sha, img = args
    out = ORB_DIR / f"{sha}.npz"
    if out.exists():
        return
    pts, desc, g = orb_feats(img, with_gray=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, pts=pts, desc=desc, gray=g)
    tmp.rename(out)


_FC: dict = {}


def _load_feat(sha):
    f = _FC.get(sha)
    if f is None:
        z = np.load(ORB_DIR / f"{sha}.npz")
        f = (z["pts"], z["desc"], z["gray"])
        if len(_FC) > 4000:
            _FC.clear()
        _FC[sha] = f
    return f


def _verify_sha(pair):
    a, b = pair
    return a, b, match_feats(_load_feat(a), _load_feat(b))


def dihedral_ncc(ga: np.ndarray, gb: np.ndarray) -> float:
    """Best NCC over the 8 flips/rotations at 128 px, ignoring near-black pixels (for pHash-proposed copies)."""
    import cv2
    A = cv2.resize(ga, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float64)
    best = -1.0
    for flip in (False, True):
        B0 = cv2.resize(gb, (128, 128), interpolation=cv2.INTER_AREA)
        B0 = cv2.flip(B0, 1) if flip else B0
        for k in range(4):
            B = np.rot90(B0, k).astype(np.float64); m = (A > 25) & (B > 25)
            if m.sum() < 500:
                continue
            x = A[m] - A[m].mean(); y = B[m] - B[m].mean(); d = np.sqrt((x * x).sum() * (y * y).sum())
            if d > 0:
                best = max(best, float((x * y).sum() / d))
    return best


def _dih_sha(pair):
    a, b = pair
    return a, b, dihedral_ncc(_load_feat(a)[2], _load_feat(b)[2])


def _photo_sha(pair):
    a, b = pair
    n, ncc, ov = verify_full(_load_feat(a), _load_feat(b))
    return a, b, n, ncc, ov


def knn_candidates(E: np.ndarray, k: int = K_NN, min_cos: float = MIN_COS) -> list[tuple[int, int]]:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(E.astype(np.float32)).to(dev)
    out = set()
    for s in range(0, len(X), 2048):
        S = X[s:s + 2048] @ X.T
        S[torch.arange(S.shape[0]), torch.arange(s, s + S.shape[0])] = -1
        v, idx = S.topk(k, dim=1)
        for r, (vr, ir) in enumerate(zip(v.cpu().numpy(), idx.cpu().numpy())):
            for c, j in zip(vr, ir):
                if c >= min_cos:
                    a, b = s + r, int(j)
                    out.add((min(a, b), max(a, b)))
    return sorted(out)


def verify_pairs(pairs, items, workers=12):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(i, j, items[i], items[j]) for i, j in pairs]
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_verify, jobs, chunksize=64))


def calibrate(n: int = 400, seed: int = 3407):
    """Positives: planted copies (crop ≤5%/side, resize 0.5–1, JPEG q50, brightness ±10%) vs their source, found
    through the same kNN stage. Negatives: each image's nearest *different-source-label-group* neighbours are not
    known to be distinct, so we use cross-source pairs between PAD-UFES patients (distinct by construction when
    present) plus same-label nearest neighbours whose inlier count we report as an upper bound on false merges."""
    df = pd.read_csv(PROCESSED / "manifest.csv")
    E = np.load(EMB).astype(np.float32)
    rng = random.Random(seed)
    idx = rng.sample(range(len(df)), n)
    m = model()
    planted = [plant(Image.open(REPO / df.img[i]).convert("RGB"), rng) for i in idx]
    planted_imgs = [p for p, _ in planted]; kinds = np.array([k for _, k in planted])
    ph_hit = []
    for t, i in enumerate(idx):
        a, ad = _phashes(planted_imgs[t])
        ph_hit.append(bin(a ^ int(df.phash[i], 16)).count("1") <= 6 or bin(ad ^ int(df.phash_dih[i], 16)).count("1") <= 4)
    ph_hit = np.array(ph_hit)
    P = embed_items(m, planted_imgs).astype(np.float32)
    cos_pos = (P * E[idx]).sum(1)
    # Linked the way the pipeline links: planted copy → its kNN candidates (cos ≥ MIN_COS) → ORB-verified;
    # success if a verified neighbour IS the original, or is itself an ORB-verified / pHash copy of the original.
    S = (torch.from_numpy(P) @ torch.from_numpy(E).T)
    top_v, top_i = S.topk(K_NN, dim=1)
    ph = df.phash.map(lambda h: int(h, 16)).values
    def linked(t, i, thr):
        for c, j in zip(top_v[t].tolist(), top_i[t].tolist()):
            if c < MIN_COS:
                continue
            if orb_inliers(planted_imgs[t], df.img[j]) < thr:
                continue
            if j == i or bin(int(ph[i]) ^ int(ph[j])).count("1") <= 6 or orb_inliers(df.img[j], df.img[i]) >= thr:
                return True
        return False
    inl_pos = np.array([orb_inliers(planted_imgs[t], df.img[i]) for t, i in enumerate(idx)])
    cand_ok = np.array([any(j == i for j in top_i[t].tolist()) for t, i in enumerate(idx)])
    # hard negatives: nearest neighbour of random images (most similar *other* image), same label, different sha1
    hard = []
    for i in rng.sample(range(len(df)), 600):
        s = E @ E[i]; s[i] = -1
        j = int(s.argmax())
        same_ph = bin(int(df.phash[i], 16) ^ int(df.phash[j], 16)).count("1") <= 6
        if df.label[i] == df.label[j] and df.sha1[i] != df.sha1[j] and not same_ph:
            hard.append((i, j, float(s[j])))
    inl_neg = np.array([orb_inliers(df.img[i], df.img[j]) for i, j, _ in hard])
    sweep = []
    for t in [10, 12, 15, 20, 25]:
        hit = np.array([bool(ph_hit[k]) or linked(k, i, t) for k, i in enumerate(idx)])
        sweep.append({"min_inliers": t, "planted_recall": float(hit.mean()),
                      "recall_by_transform": {k: float(hit[kinds == k].mean()) for k in sorted(set(kinds))},
                      "phash_only_recall": float(ph_hit.mean()), "nn_same_label_merge_rate": float((inl_neg >= t).mean())})
    # nn negatives are contaminated by real copies (bimodal: ~4 vs 200+ inliers), so pick the highest threshold
    # that still links ≥ 95% of planted copies (most conservative against false merges).
    ok = [w for w in sweep if w["planted_recall"] >= 0.95]
    best = max(ok, key=lambda w: w["min_inliers"]) if ok else max(sweep, key=lambda w: w["planted_recall"])
    rep = {**report_meta(), "arch": ARCH, "k_nn": K_NN, "min_cos": MIN_COS, "n_planted": n, "n_hard_neg": len(hard),
           "original_in_topk": float(cand_ok.mean()), "chosen": best, "sweep": sweep,
           "cos_pos_quantiles": np.quantile(cos_pos, [0.01, 0.05, 0.5]).tolist(),
           "inliers_pos_quantiles": np.quantile(inl_pos, [0.05, 0.5]).tolist(),
           "inliers_neg_quantiles": np.quantile(inl_neg, [0.5, 0.95, 0.99]).tolist(),
           "note": "nn_same_label_merge_rate is an upper bound: some nearest neighbours are genuine copies the pHash stage missed."}
    CAL.write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ["original_in_topk", "chosen", "sweep", "inliers_pos_quantiles", "inliers_neg_quantiles"]}, indent=1))


def features(workers: int = 12):
    """ORB features for every manifest image not yet cached (restart-safe: one file per image, atomic rename)."""
    from concurrent.futures import ProcessPoolExecutor
    df = pd.read_csv(PROCESSED / "manifest.csv").drop_duplicates("sha1")
    ORB_DIR.mkdir(parents=True, exist_ok=True)
    todo = [(r.sha1, r.img) for r in df.itertuples() if not (ORB_DIR / f"{r.sha1}.npz").exists()]
    print(f"orb features: {len(df) - len(todo)} cached, {len(todo)} to compute", flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for k, _ in enumerate(ex.map(_orb_job, todo, chunksize=64)):
            if k % 5000 == 0:
                print(f"  {k}/{len(todo)}", flush=True)


def _read_cache(path, ncols):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split("\t")
            if len(p) == ncols:
                out[(p[0], p[1])] = tuple(float(x) for x in p[2:])
    return out


def _verify_chunks(todo, fn, path, workers, chunk, fmt):
    from concurrent.futures import ProcessPoolExecutor
    import os
    with ProcessPoolExecutor(workers) as ex:
        for s0 in range(0, len(todo), chunk):
            res = list(ex.map(fn, todo[s0:s0 + chunk], chunksize=256))
            with open(path, "a") as f:
                f.write("".join(fmt(r) for r in res)); f.flush(); os.fsync(f.fileno())
            print(f"  verified {min(s0 + chunk, len(todo))}/{len(todo)} → {path.name}", flush=True)


def pairs(workers: int = 12, chunk: int = 20000):
    """Candidates (DINOv2 kNN ∪ pHash near pairs) → geometric + photometric verification → copy edges.
    Writes copy_pairs.npy (row-index pairs, ORB-verified) and phash_pairs.npy (row-index pairs, dihedral-NCC-verified)."""
    from ml.data.dedupe import near_pairs, _bits, PHASH_T, DIH_T
    df = pd.read_csv(PROCESSED / "manifest.csv", low_memory=False)
    E = np.load(EMB)
    assert len(E) == len(df), "embeddings out of date: run `embed` first"
    features(workers)
    t = json.loads(CAL.read_text())["chosen"]["min_inliers"]
    sha = df.sha1.values
    knn = {tuple(sorted((sha[i], sha[j]))) for i, j in knn_candidates(E) if sha[i] != sha[j]}
    done = _read_cache(PAIR_CACHE, 5)
    todo = sorted(c for c in knn if c not in done)
    print(f"orb candidates={len(knn)} cached={len(knn) - len(todo)} to_verify={len(todo)}", flush=True)
    _verify_chunks(todo, _photo_sha, PAIR_CACHE, workers, chunk, lambda r: f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.4f}\n")
    done = _read_cache(PAIR_CACHE, 5)
    ph = {tuple(sorted((sha[i], sha[j]))) for i, j in list(near_pairs(_bits(df.phash), PHASH_T)) + list(near_pairs(_bits(df.phash_dih), DIH_T))
          if sha[i] != sha[j]}
    pdone = _read_cache(PHASH_CACHE, 3)
    ptodo = sorted(c for c in ph if c not in pdone)
    print(f"phash candidates={len(ph)} cached={len(ph) - len(ptodo)} to_verify={len(ptodo)}", flush=True)
    _verify_chunks(ptodo, _dih_sha, PHASH_CACHE, workers, chunk, lambda r: f"{r[0]}\t{r[1]}\t{r[2]:.4f}\n")
    pdone = _read_cache(PHASH_CACHE, 3)
    rows = {}
    for i, h in enumerate(sha):
        rows.setdefault(h, i)
    ok = [(rows[a], rows[b]) for (a, b) in knn if (v := done.get((a, b))) and v[0] >= t and v[1] >= ORB_MIN_NCC and v[2] >= ORB_MIN_OVERLAP]
    pok = [(rows[a], rows[b]) for (a, b) in ph if pdone.get((a, b), (-1,))[0] >= PHASH_MIN_NCC]
    np.save(PROCESSED / "copy_pairs.npy", np.array(ok, dtype=np.int64).reshape(-1, 2))
    np.save(PROCESSED / "phash_pairs.npy", np.array(pok, dtype=np.int64).reshape(-1, 2))
    lab = df.label.values
    stats = {"orb_candidates": len(knn), "orb_verified": len(ok), "orb_cross_label_verified": int(sum(lab[i] != lab[j] for i, j in ok)),
             "phash_candidates": len(ph), "phash_verified": len(pok), "phash_cross_label_candidates": int(sum(lab[rows[a]] != lab[rows[b]] for a, b in ph)),
             "phash_cross_label_verified": int(sum(lab[i] != lab[j] for i, j in pok)),
             "thresholds": {"min_inliers": t, "orb_min_ncc": ORB_MIN_NCC, "orb_min_overlap": ORB_MIN_OVERLAP, "phash_min_ncc": PHASH_MIN_NCC}}
    (REPORTS / "copy_verification.json").write_text(json.dumps({**report_meta(), **stats}, indent=1))
    print(json.dumps(stats), flush=True)


def embed():
    """Embeddings cached by sha1; only new images are run through the GPU."""
    df = pd.read_csv(PROCESSED / "manifest.csv")
    cache: dict[str, np.ndarray] = {}
    if EMB_CACHE.exists():
        z = np.load(EMB_CACHE, allow_pickle=False)
        cache = dict(zip(z["sha1"].tolist(), z["E"]))
    elif EMB.exists():   # migrate the first (row-aligned, pre-cache) run
        old = pd.read_csv(PROCESSED / "labeled_manifest_v1.csv") if (PROCESSED / "labeled_manifest_v1.csv").exists() else None
        if old is not None and len(old) == len(np.load(EMB)):
            cache = dict(zip(old.sha1.tolist(), np.load(EMB)))
    first = df.drop_duplicates("sha1")
    todo = first[~first.sha1.isin(cache.keys())]
    print(f"embeddings: {len(first) - len(todo)} cached, {len(todo)} to compute", flush=True)
    if len(todo):
        Enew = embed_items(model(), todo.img.tolist())
        cache.update(zip(todo.sha1.tolist(), Enew))
        keys = list(cache.keys())
        tmp = PROCESSED / "emb_cache.tmp.npz"
        np.savez(tmp, sha1=np.array(keys), E=np.stack([cache[k] for k in keys]))
        tmp.rename(EMB_CACHE)
    E = np.stack([cache[h] for h in df.sha1])
    np.save(EMB, E)
    print("embeddings", E.shape, flush=True)


if __name__ == "__main__":
    {"embed": embed, "calibrate": calibrate, "pairs": pairs}[sys.argv[1]]()
