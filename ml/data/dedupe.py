"""dedupe: cluster near-duplicates BEFORE splitting (CLAUDE.md non-negotiable 6).

Two levels:
  dup_cluster — copies of one photo, from VERIFIED edges only (embed_dedupe.py `pairs`):
                sha1 equal; pHash/dihedral-pHash candidates whose dihedral NCC ≥ 0.93; DINOv2-kNN candidates with
                ≥ 25 ORB/RANSAC inliers AND photometric NCC ≥ 0.6 over ≥ 25 % overlap.
                Raw pHash distance alone is NOT an edge: dermoscopy photos (centred blob on pink skin) collide
                (measured: 2,734 cross-label pHash links; unverified chaining built a 4,688-image "cluster").
                One representative kept (`keep`), largest original resolution.
  group       — what the splitter keeps together: dup_cluster ∪ source patient/lesion id (PAD-UFES, SkinDisNet).
Label noise: a dup_cluster whose verified copies carry different labels is dropped (the same photo filed twice).
External isolation: any pool image whose group touches an external-only image is removed from the pool.

Output: data/processed/manifest.csv gains dup_cluster, group, keep, drop_reason; reports/dedupe_log.csv, reports/dedupe.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ml.common.paths import PROCESSED, REPORTS, report_meta

PHASH_T = 6
DIH_T = 4
CAL = REPORTS / "dedupe_calibration.json"
COPY_PAIRS = PROCESSED / "copy_pairs.npy"
PHASH_PAIRS = PROCESSED / "phash_pairs.npy"


def _bits(hexes: pd.Series) -> np.ndarray:
    return np.array([int(h, 16) for h in hexes], dtype=np.uint64)


_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def popcount64(x: np.ndarray) -> np.ndarray:
    b = x.view(np.uint8).reshape(*x.shape, 8)
    return _POP[b].sum(-1, dtype=np.uint16)


class UF:
    def __init__(self, n):
        self.p = np.arange(n)

    def find(self, a):
        p = self.p
        r = a
        while p[r] != r:
            r = p[r]
        while p[a] != r:
            p[a], a = r, p[a]
        return r

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)

    def roots(self):
        return np.array([self.find(i) for i in range(len(self.p))])


def near_pairs(h: np.ndarray, t: int, chunk: int = 256):
    n = len(h)
    for s in range(0, n, chunk):
        blk = h[s:s + chunk]
        d = popcount64(blk[:, None] ^ h[None, :])
        ii, jj = np.nonzero(d <= t)
        ii = ii + s
        m = jj > ii
        yield from zip(ii[m].tolist(), jj[m].tolist())


def phash_candidates(df: pd.DataFrame):
    """pHash ≤ 6 or dihedral pHash ≤ 4 — candidate proposals only (verified in embed_dedupe.pairs)."""
    yield from near_pairs(_bits(df.phash), PHASH_T)
    yield from near_pairs(_bits(df.phash_dih), DIH_T)


def dup_clusters(df: pd.DataFrame, *verified: np.ndarray | None) -> np.ndarray:
    uf = UF(len(df))
    for edges in verified:
        if edges is not None:
            for i, j in edges.tolist():
                uf.union(i, j)
    for _, idx in df.groupby("sha1").indices.items():
        for k in idx[1:]:
            uf.union(idx[0], k)
    return uf.roots()


def groups(df: pd.DataFrame, dup: np.ndarray) -> np.ndarray:
    uf = UF(len(df))
    for _, idx in pd.Series(range(len(df))).groupby(dup).groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf.union(idx[0], k)
    pid = df.group_id.where(df.source.isin(["pad_ufes20", "skindisnet", "scin"]))   # sources with real patient / case ids
    for _, idx in pd.Series(range(len(df))).groupby(pid).groups.items():
        idx = list(idx)
        for k in idx[1:]:
            uf.union(idx[0], k)
    return uf.roots()


def main():
    df = pd.read_csv(PROCESSED / "manifest.csv", low_memory=False).reset_index(drop=True)
    df = df.drop(columns=[c for c in ["dup_cluster", "group", "keep", "drop_reason"] if c in df], errors="ignore")
    cp = np.load(COPY_PAIRS) if COPY_PAIRS.exists() else None
    pp = np.load(PHASH_PAIRS) if PHASH_PAIRS.exists() else None
    for e in (cp, pp):
        if e is not None and len(e):
            assert e.max() < len(df), "verified pairs out of date: rerun embed_dedupe embed + pairs"
    dup = dup_clusters(df, cp, pp)
    df["dup_cluster"] = dup
    df["group"] = groups(df, dup)
    thr = json.loads(CAL.read_text())["chosen"]["min_inliers"] if CAL.exists() else None
    n_emb = 0 if cp is None else int(len(cp))
    n_ph = 0 if pp is None else int(len(pp))
    df["drop_reason"] = ""
    # label conflicts inside a near-identical cluster → label noise → drop the cluster
    nlab = df.groupby("dup_cluster").label.transform("nunique")
    df.loc[nlab > 1, "drop_reason"] = "label_conflict"
    # external isolation
    ext_groups = set(df.loc[df.split_role == "external", "group"])
    df.loc[(df.split_role == "pool") & df.group.isin(ext_groups), "drop_reason"] = "touches_external"
    # one representative per dup_cluster (largest original resolution)
    ok = df[df.drop_reason == ""].assign(_px=lambda d: d.w * d.h)
    keep_idx = ok.sort_values("_px", ascending=False).groupby("dup_cluster").head(1).index
    df["keep"] = df.index.isin(keep_idx)
    df.to_csv(PROCESSED / "manifest.csv", index=False)

    sizes = df.groupby("dup_cluster").size()
    gsz = df[df.keep].groupby("group").size()
    multi = df[df.dup_cluster.isin(sizes[sizes > 1].index)].sort_values("dup_cluster")
    REPORTS.mkdir(exist_ok=True)
    multi[["dup_cluster", "group", "source", "source_label", "label", "path", "phash", "drop_reason"]].to_csv(
        REPORTS / "dedupe_log.csv", index=False)
    rep = {**report_meta(), "n_images": int(len(df)), "n_dup_clusters": int(df.dup_cluster.nunique()),
           "n_multi_clusters": int((sizes > 1).sum()), "n_kept": int(df.keep.sum()),
           "dropped": df.drop_reason.value_counts().to_dict(), "orb_min_inliers": thr, "orb_verified_copy_pairs": n_emb, "phash_verified_pairs": n_ph,
           "n_groups_kept": int(len(gsz)), "largest_groups": gsz.sort_values(ascending=False).head(10).tolist(),
           "kept_by_label_role": df[df.keep].groupby(["split_role", "label"]).size().unstack(0).fillna(0).astype(int).to_dict(),
           "cross_source_dup_clusters": int((df.groupby("dup_cluster").source.nunique() > 1).sum())}
    (REPORTS / "dedupe.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "kept_by_label_role"}, indent=1))
    print(df[df.keep].groupby(["label", "split_role"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
