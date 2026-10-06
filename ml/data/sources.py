"""Enumerate raw images from every source → data/processed/sources.csv

Columns: path, source, source_label, group_id, split_hint, + metadata (PAD-UFES: answers-like fields, fitzpatrick).
- Pre-augmented copies (Keras-style "_0_1234" suffix, Windows "- Copy") are DROPPED: they are not new information
  and are the main leakage vector (plan §5). Their originals remain.
- group_id = patient/lesion id when the source has one; else a filename stem; dedupe later merges groups further.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from ml.common.paths import PROCESSED, RAW

EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AUG_RE = re.compile(r"(_0_\d+$)|( - Copy( \(\d+\))?$)|(\(\d+\)$)", re.IGNORECASE)


def is_augmented(p: Path) -> bool:
    return bool(AUG_RE.search(p.stem.strip()))


def stem_group(p: Path) -> str:
    s = AUG_RE.sub("", p.stem.strip())
    return re.sub(r"\s+", "", s).lower()


def folder_source(name: str, root: Path, drop_aug: bool = True) -> list[dict]:
    rows = []
    for p in root.rglob("*"):
        if p.suffix.lower() not in EXT or not p.is_file():
            continue
        if drop_aug and is_augmented(p):
            continue
        parts = p.relative_to(root).parts
        split_hint = next((x.lower() for x in parts if x.lower() in {"train", "val", "valid", "test"}), "")
        rows.append({"path": str(p), "source": name, "source_label": p.parent.name.strip(),
                     "group_id": f"{name}:{stem_group(p)}", "split_hint": split_hint})
    return rows


def pad_ufes(root: Path) -> list[dict]:
    meta = next(root.rglob("metadata.csv"))
    md = pd.read_csv(meta)
    imgs = {p.name: p for p in root.rglob("*.png")}
    rows = []
    for r in md.itertuples(index=False):
        p = imgs.get(r.img_id)
        if p is None:
            continue
        fitz = r.fitspatrick if "fitspatrick" in md.columns else None
        tone = "unknown"
        if pd.notna(fitz):
            f = int(fitz); tone = "fitz_1_2" if f <= 2 else "fitz_3_4" if f <= 4 else "fitz_5_6"
        rows.append({"path": str(p), "source": "pad_ufes20", "source_label": r.diagnostic,
                     "group_id": f"pad_ufes20:{r.patient_id}", "split_hint": "", "skin_tone": tone,
                     "pad_itch": r.itch, "pad_grew": r.grew, "pad_hurt": r.hurt, "pad_changed": r.changed,
                     "pad_bleed": r.bleed, "pad_elevation": r.elevation, "pad_region": r.region, "pad_age": r.age,
                     "pad_biopsed": r.biopsed})
    return rows


def skindisnet(root: Path) -> list[dict]:
    """SkinDisNet v2 'Preprocessed' originals only (the 'Augmented' folders are 7 derived copies per original).
    Metadata gives the patient id (group for splitting), age and lesion location."""
    meta = next(root.rglob("SkinDisNet_Metadata.csv"))
    md = pd.read_csv(meta)
    base = meta.parent / "Preprocessed"
    rows = []
    for r in md.itertuples(index=False):
        p = base / r.Folder_name / f"{r.Image_id}.jpg"
        if not p.exists():
            continue
        rows.append({"path": str(p), "source": "skindisnet", "source_label": r.Folder_name,
                     "group_id": f"skindisnet:{r.Patient_id}", "split_hint": "",
                     "sdn_age": r.Age, "sdn_sex": r.Sex, "sdn_location": r.Leision_location})
    return rows


def scin(root: Path, min_weight: float = 0.5) -> list[dict]:
    """SCIN (Google Research + Stanford, CC BY 4.0): crowdsourced US phone photos, 1–3 images per case, dermatologist
    differential with weights. A case is used only when one condition holds ≥ `min_weight` of the dermatologists' weight
    (consensus); its name becomes source_label and label_map.json `folder_map.scin` maps it. Case id = patient group.
    Self-reported questionnaire fields are kept (scin_*) for SFT prompts with real answers."""
    import ast
    cases = pd.read_csv(root / "scin_cases.csv").merge(pd.read_csv(root / "scin_labels.csv"), on="case_id")
    rows = []
    for r in cases.itertuples(index=False):
        w = r.weighted_skin_condition_label
        if not isinstance(w, str) or w in ("", "{}"):
            continue
        cond, wt = max(ast.literal_eval(w).items(), key=lambda kv: kv[1])
        if wt < min_weight:
            continue
        fst = r.dermatologist_fitzpatrick_skin_type_label_1 if isinstance(r.dermatologist_fitzpatrick_skin_type_label_1, str) else \
            (r.fitzpatrick_skin_type if isinstance(r.fitzpatrick_skin_type, str) and r.fitzpatrick_skin_type.startswith("FST") else None)
        tone = "unknown" if not fst else "fitz_1_2" if fst in ("FST1", "FST2") else "fitz_3_4" if fst in ("FST3", "FST4") else "fitz_5_6"
        parts = [c[len("body_parts_"):] for c in cases.columns if c.startswith("body_parts_") and getattr(r, c) == "YES"]
        symptoms = [c[len("condition_symptoms_"):] for c in cases.columns if c.startswith("condition_symptoms_") and getattr(r, c) == "YES"]
        for k in ("image_1_path", "image_2_path", "image_3_path"):
            ip = getattr(r, k)
            if not isinstance(ip, str):
                continue
            path = root / "images" / Path(ip).name
            if not path.exists() or path.stat().st_size == 0:
                continue
            rows.append({"path": str(path), "source": "scin", "source_label": cond, "group_id": f"scin:{r.case_id}", "split_hint": "",
                         "skin_tone": tone, "scin_weight": wt, "scin_age": r.age_group, "scin_sex": r.sex_at_birth,
                         "scin_duration": r.condition_duration, "scin_parts": ";".join(parts), "scin_symptoms": ";".join(symptoms),
                         "scin_fever": r.other_symptoms_fever == "YES", "scin_shot": getattr(r, k.replace("_path", "_shot_type"))})
    return rows


SOURCES = {
    "mgmitesh": lambda: folder_source("mgmitesh", RAW / "mgmitesh"),
    "pacificrm": lambda: folder_source("pacificrm", RAW / "pacificrm"),
    "skindisnet": lambda: skindisnet(RAW / "skindisnet"),
    "skindiseasebd": lambda: folder_source("skindiseasebd", RAW / "skindiseasebd", drop_aug=False),  # aug-only release; dedupe collapses copies
    "pad_ufes20": lambda: pad_ufes(RAW / "pad_ufes20"),
    "scin": lambda: scin(RAW / "scin"),
}


def main(only: list[str] | None = None):
    rows = []
    for name, fn in SOURCES.items():
        if only and name not in only:
            continue
        r = fn()
        print(f"{name}: {len(r)} images")
        rows += r
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "sources.csv", index=False)
    print(df.groupby(["source", "source_label"]).size().to_string())


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
