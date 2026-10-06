"""reports/data_card.md — regenerated from splits + reports only, deterministic (test D7: byte-identical on rerun)."""
from __future__ import annotations

import json

import pandas as pd

from ml.common.paths import REPORTS, SPLITS
from ml.common.paths import dataset_rev

LICENSES = {
    "mgmitesh": "CC BY 4.0 (Kaggle page). Mixed provenance: DermNet-topic files, ISIC dermoscopy, web images. Pre-augmented copies removed.",
    "pacificrm": "Kaggle mirror of the DermNet 23-class set. DermNet NZ images: non-commercial educational use with attribution. Used because it is the only psoriasis/vitiligo/scabies source found.",
    "skindisnet": "CC BY-NC 4.0 (Mendeley yj3md44hxg v2).",
    "pad_ufes20": "CC BY 4.0 (Mendeley zr7vgbcyr2).",
    "skindiseasebd": "CC BY-NC 4.0 (Mendeley 9ggd3shdr7). Augmentation-only release; one image per near-duplicate group scored.",
    "scin": "CC BY 4.0 (Google Research + Stanford, github.com/google-research-datasets/scin). Crowdsourced US phone photos with dermatologist labels; consensus label (weight ≥ 0.5) only.",
}


def main():
    sp = {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ["train", "val", "test", "external_test"]}
    cls = json.loads((SPLITS / "classes.json").read_text())
    lm = json.loads((REPORTS / "label_map_report.json").read_text())
    dd = json.loads((REPORTS / "dedupe.json").read_text())
    cal = json.loads((REPORTS / "dedupe_calibration.json").read_text()) if (REPORTS / "dedupe_calibration.json").exists() else None
    tab = pd.DataFrame({s: d.label.value_counts() for s, d in sp.items()}).fillna(0).astype(int)
    tab.loc["TOTAL"] = tab.sum()
    src = pd.DataFrame({s: d.source.value_counts() for s, d in sp.items()}).fillna(0).astype(int)
    lines = [
        "# SkinNova data card", "",
        f"dataset_rev: `{dataset_rev()}` · label map {lm['label_map_version']}", "",
        "## Classes", f"Trainable: {', '.join(cls['classes'])}",
        f"Dropped by D5 (< 150 distinct images): {cls['dropped_d5'] or 'none'}", "",
        "## Images per split and class", tab.to_markdown(), "",
        "## Images per split and source", src.to_markdown(), "",
        "## Pipeline counts",
        f"- enumerated {lm['n_enumerated']} files (pre-augmented copies already skipped) → labeled {lm['n_labeled']}",
        f"- label decisions: {lm['by_reason']}",
        f"- near-identical clusters: {dd['n_dup_clusters']} ({dd['n_multi_clusters']} with >1 copy, "
        f"{dd['cross_source_dup_clusters']} spanning two sources) → kept {dd['n_kept']}",
        f"- dropped: {dd['dropped']}",
        f"- copy detection (DINOv2-s kNN candidates → ORB/RANSAC ≥ {dd['orb_min_inliers']} inliers): {dd['orb_verified_copy_pairs']} verified copy pairs",
    ]
    if cal:
        lines.append(f"  - calibration (test D2): planted-copy recall {cal['chosen']['planted_recall']:.3f} "
                     f"(pHash alone {cal['chosen']['phash_only_recall']:.3f}); borderline pairs inspected: "
                     f"{cal['visual_check']['false_merges']}/{cal['visual_check']['pairs']} false merges")
    lines += ["", "## Sources and licenses"] + [f"- **{k}**: {v}" for k, v in LICENSES.items()] + [
        "", "## Known limitations",
        "- Most inflammatory-class images are DermNet clinical photos (lighter skin tones over-represented, professional lighting).",
        "- Fitzpatrick labels exist only for PAD-UFES-20; per-tone metrics elsewhere are not possible.",
        "- `other` is heterogeneous (normal skin, bullous, lupus, lichen, warts, drug eruptions, rosacea, candida ...).",
        "- Labels come from folder names or DermNet topic slugs, not from a dermatologist review of each image (diy.md D8 spot-check).",
    ]
    (REPORTS / "data_card.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
