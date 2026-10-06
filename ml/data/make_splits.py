"""Group-stratified splits → data/splits/{train,val,test,external_test}.csv + data/splits/classes.json

- Pool (non-external sources), keep==True rows only. Groups (dedupe.py) never straddle splits.
- StratifiedGroupKFold(k=7, fixed seed): fold 0 → test, fold 1 → val, rest → train (≈ 71/14.5/14.5).
- Frozen splits (v2, 2026-10-06): data/splits/frozen_v1.csv records the v1 image → split map. Images already split keep
  their split; a group containing v1 images goes wholly to their (majority) split; only groups with no v1 image (new
  sources, e.g. SCIN) are split by StratifiedGroupKFold. v1 test images therefore never move into training.
- D5 class floor: a class with < 150 distinct kept images in the pool is dropped from the trainable label set
  (logged; `other` absorbs nothing silently — the rows are removed and listed in reports/splits.json).
- external_test: external-only sources, one image per group (augmented copies of one original count once).
Columns: img,label,source,source_label,group,phash,real_q,skin_tone (+ PAD metadata when present).
"""
from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from ml.common.paths import PROCESSED, REPORTS, SEED, SPLITS, report_meta
from ml.common.paths import dataset_rev  # noqa: F401  (re-exported; used below and by older imports)
from ml.common.schema import label_keys

MIN_PER_CLASS = 150
COLS = ["img", "label", "source", "source_label", "group", "phash", "phash_dih", "sha1", "real_q", "skin_tone"]


def main():
    df = pd.read_csv(PROCESSED / "manifest.csv")
    df = df[df.keep].copy()
    df["real_q"] = df.source.isin(["pad_ufes20", "scin"])   # sources with real self-reported questionnaire answers
    if "skin_tone" not in df:
        df["skin_tone"] = "unknown"
    df["skin_tone"] = df.skin_tone.fillna("unknown")
    pool, ext = df[df.split_role == "pool"].copy(), df[df.split_role == "external"].copy()

    counts = pool.label.value_counts()
    dropped = {k: int(counts.get(k, 0)) for k in label_keys() if counts.get(k, 0) < MIN_PER_CLASS}
    classes = [k for k in label_keys() if k not in dropped]
    pool = pool[pool.label.isin(classes)]

    frozen_path = SPLITS / "frozen_v1.csv"
    if not frozen_path.exists() and (SPLITS / "train.csv").exists():   # first run after v1: record the v1 assignment
        pd.concat([pd.read_csv(SPLITS / f"{s}.csv", usecols=["img"]).assign(split=s) for s in ["train", "val", "test"]]
                  ).to_csv(frozen_path, index=False)
    frozen = pd.read_csv(frozen_path).set_index("img").split.to_dict() if frozen_path.exists() else {}
    pool["split"] = pool.img.map(frozen)
    grp = pool.dropna(subset=["split"]).groupby("group").split.agg(lambda x: x.value_counts().index[0])
    pool["split"] = pool.split.fillna(pool.group.map(grp))
    new = pool[pool.split.isna()]
    if len(new):
        sgkf = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
        fold = pd.Series(-1, index=new.index)
        for f, (_, te) in enumerate(sgkf.split(new, new.label, new.group)):
            fold.iloc[te] = f
        pool.loc[new.index, "split"] = fold.map(lambda f: "test" if f == 0 else "val" if f == 1 else "train")
    moved = sum(1 for i, sp in zip(pool.img, pool.split) if i in frozen and frozen[i] != sp)

    ext = ext[ext.label.isin(classes)].sort_values("img").groupby("group").head(1)
    extra = [c for c in pool.columns if c.startswith(("pad_", "sdn_", "scin_"))]
    SPLITS.mkdir(parents=True, exist_ok=True)
    out = {}
    for s in ["train", "val", "test"]:
        part = pool[pool.split == s][COLS + extra].sort_values("img").reset_index(drop=True)
        part.to_csv(SPLITS / f"{s}.csv", index=False)
        out[s] = part
    ext[COLS].sort_values("img").reset_index(drop=True).to_csv(SPLITS / "external_test.csv", index=False)
    out["external_test"] = ext
    (SPLITS / "classes.json").write_text(json.dumps({"classes": classes, "dropped_d5": dropped}, indent=1))

    share = {s: (out[s].label.value_counts(normalize=True) * 100).round(2).to_dict() for s in ["train", "val", "test"]}
    d6 = max(abs(share[a].get(k, 0) - share[b].get(k, 0)) for k in classes for a in share for b in share)
    rep = {**report_meta(dataset_rev=dataset_rev()), "classes": classes, "dropped_d5": dropped,
           "n": {s: int(len(v)) for s, v in out.items()},
           "per_class": {s: v.label.value_counts().to_dict() for s, v in out.items()},
           "share_pct": share, "D6_max_share_diff_pts": float(d6), "D6_pass": bool(d6 <= 3.0),
           "frozen_v1_images": len(frozen), "v1_images_moved": int(moved), "new_images_split": int(len(new)),
           "per_source": {s: v.source.value_counts().to_dict() for s, v in out.items()}}
    (REPORTS / "splits.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ["n", "dropped_d5", "D6_max_share_diff_pts"]}, indent=1))
    print(pd.DataFrame(rep["per_class"]).fillna(0).astype(int).to_string())


if __name__ == "__main__":
    main()
