"""Assign taxonomy labels to every enumerated source image → data/processed/labeled.csv + reports/label_map_report.json

Order: slug rule (DermNet topic slug in filename, only for `slug_sources`) → folder map → excluded.
Folders in `slug_only` never fall back to their (wrong) folder label. Every decision is counted in the report so
nothing is dropped silently (test D4). Class floor D5 (≥150 distinct images) is checked after dedupe in make_splits.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from ml.common.paths import PROCESSED, REPO, REPORTS, report_meta
from ml.common.schema import label_keys

MAP_FILE = REPO / "ml" / "data" / "label_map.json"
EXT_RE = re.compile(r"\.(jpe?g|png|bmp|webp)$", re.I)


def load_map(path: Path = MAP_FILE) -> dict:
    m = json.loads(path.read_text())
    m["_rules"] = [(re.compile(p, re.I), lab) for p, lab in m["slug_rules"]]
    return m


def slug_of(filename: str) -> str:
    s = EXT_RE.sub("", Path(filename).name).lower()
    s = re.sub(r"\.rf\.[0-9a-f]+", "", s)       # roboflow export hash
    s = re.sub(r"__(protect|watermarked)\w*", "", s)  # DermNet CDN crop/watermark suffix
    return s


def assign(m: dict, source: str, folder: str, filename: str) -> tuple[str | None, str]:
    """→ (label or None, reason). None = excluded."""
    folder = folder.strip()
    if folder in m.get("exclude_folders", {}).get(source, {}):
        return None, "exclude_folder"
    if source in m["slug_sources"]:
        s = slug_of(filename)
        for rx, lab in m["_rules"]:
            if rx.search(s):
                return (None, f"slug_exclude:{rx.pattern[:24]}") if lab == "EXCLUDE" else (lab, "slug")
    if folder in m.get("slug_only", {}).get(source, {}):
        return None, "slug_only_nomatch"
    lab = m["folder_map"].get(source, {}).get(folder)
    return (lab, "folder") if lab else (None, "unmapped_folder")


def main():
    m = load_map()
    keys = set(label_keys())
    df = pd.read_csv(PROCESSED / "sources.csv")
    res = [assign(m, r.source, str(r.source_label), r.path) for r in df.itertuples(index=False)]
    df["label"] = [a for a, _ in res]
    df["label_reason"] = [b for _, b in res]
    bad = set(df.label.dropna()) - keys
    assert not bad, f"label_map produces keys not in labels.json: {bad}"
    # D4: every (source, folder) must be accounted for
    seen = df.groupby(["source", "source_label"]).size()
    unaccounted = [f"{s}/{f}" for (s, f) in seen.index
                   if f not in m["folder_map"].get(s, {}) and f not in m.get("slug_only", {}).get(s, {})
                   and f not in m.get("exclude_folders", {}).get(s, {})]
    kept = df[df.label.notna()].copy()
    kept["split_role"] = kept.source.map(lambda s: "external" if s in m["external_only_sources"] else "pool")
    kept.to_csv(PROCESSED / "labeled.csv", index=False)
    cross = (df.assign(label=df.label.fillna("<excluded>"))
             .groupby(["source", "source_label", "label"]).size().reset_index(name="n"))
    rep = {**report_meta(), "label_map_version": m["version"], "n_enumerated": int(len(df)), "n_labeled": int(len(kept)),
           "unaccounted_folders": unaccounted, "by_reason": df.label_reason.str.split(":").str[0].value_counts().to_dict(),
           "by_label": kept.groupby(["split_role", "label"]).size().unstack(0).fillna(0).astype(int).to_dict(),
           "folder_to_label": cross.to_dict("records")}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "label_map_report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: rep[k] for k in ["n_enumerated", "n_labeled", "unaccounted_folders", "by_reason"]}, indent=1))
    print(kept.groupby(["label", "source"]).size().unstack(fill_value=0).to_string())
    assert not unaccounted, f"D4 fail: unaccounted folders {unaccounted}"


if __name__ == "__main__":
    main()
