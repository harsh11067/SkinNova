"""Questionnaire answers for SFT/eval cases (plan §8.2).

real_q=True  — built from the source's own patient metadata (PAD-UFES-20: itch/hurt/grew/changed/bleed/region/age;
               SkinDisNet: location/age). Fields the source never asked are filled from the class prior and listed
               in `imputed` so the report can separate truly-real fields.
real_q=False — sampled from per-class plausible distributions below (clinical-textbook level, written for this
               project; they shape *inputs* only, never the medical facts, which come from condition cards).
"""
from __future__ import annotations

import random

from ml.common.schema import AGE_BANDS

# weights are relative; keys must be valid contract enums
PRIORS = {
    "eczema_atopic": dict(site={"hand": 4, "arm": 3, "leg": 2, "neck": 2, "face": 2, "foot": 1, "back": 1},
                          dur={"1_4w": 3, "1_6m": 4, "gt_6m": 3, "lt_1w": 1}, itch=(2, 3), pain=(0, 1),
                          chg={"no": 3, "spreading": 2, "unsure": 2}, bleed=.15, fever=.02, others=.05, product=.15,
                          age={"lt_12": 3, "12_17": 2, "18_39": 3, "40_59": 1, "60_plus": 1}),
    "contact_dermatitis": dict(site={"hand": 5, "face": 2, "neck": 2, "arm": 2, "foot": 1}, dur={"lt_1w": 4, "1_4w": 3, "1_6m": 1},
                               itch=(1, 3), pain=(0, 2), chg={"no": 2, "spreading": 2, "unsure": 1}, bleed=.05, fever=.01,
                               others=.03, product=.7, age={"12_17": 1, "18_39": 4, "40_59": 3, "60_plus": 1}),
    "seborrheic_dermatitis": dict(site={"scalp": 5, "face": 4, "chest": 1}, dur={"1_6m": 4, "gt_6m": 4, "1_4w": 1},
                                  itch=(1, 2), pain=(0, 0), chg={"no": 4, "unsure": 1}, bleed=.02, fever=.0, others=.02,
                                  product=.05, age={"18_39": 3, "40_59": 3, "60_plus": 2, "lt_12": 1}),
    "tinea": dict(site={"groin": 4, "foot": 3, "abdomen": 2, "back": 2, "arm": 2, "nails": 2, "leg": 1, "face": 1},
                  dur={"1_4w": 4, "1_6m": 4, "lt_1w": 1, "gt_6m": 1}, itch=(2, 3), pain=(0, 1),
                  chg={"spreading": 5, "no": 2, "unsure": 1}, bleed=.05, fever=.01, others=.25, product=.05,
                  age={"12_17": 2, "18_39": 4, "40_59": 3, "60_plus": 1, "lt_12": 1}),
    "scabies": dict(site={"hand": 4, "abdomen": 3, "groin": 2, "arm": 2, "leg": 1}, dur={"lt_1w": 2, "1_4w": 5, "1_6m": 1},
                    itch=(3, 3), pain=(0, 1), chg={"spreading": 4, "no": 1, "unsure": 1}, bleed=.15, fever=.02, others=.75,
                    product=.03, age={"lt_12": 3, "12_17": 2, "18_39": 3, "40_59": 1, "60_plus": 1}),
    "acne": dict(site={"face": 7, "back": 2, "chest": 2}, dur={"1_6m": 4, "gt_6m": 4, "1_4w": 1}, itch=(0, 1), pain=(0, 2),
                 chg={"no": 3, "spreading": 1, "unsure": 1}, bleed=.05, fever=.0, others=.02, product=.15,
                 age={"12_17": 5, "18_39": 4, "40_59": 1}),
    "psoriasis": dict(site={"scalp": 3, "arm": 3, "leg": 3, "back": 2, "nails": 1, "hand": 1}, dur={"gt_6m": 5, "1_6m": 3, "1_4w": 1},
                      itch=(1, 2), pain=(0, 1), chg={"no": 2, "spreading": 2, "unsure": 1}, bleed=.1, fever=.01, others=.02,
                      product=.02, age={"18_39": 3, "40_59": 3, "60_plus": 2, "12_17": 1}),
    "vitiligo": dict(site={"face": 3, "hand": 3, "arm": 2, "leg": 2, "neck": 1}, dur={"1_6m": 4, "gt_6m": 5, "1_4w": 1},
                     itch=(0, 0), pain=(0, 0), chg={"no": 3, "spreading": 3, "unsure": 1}, bleed=.0, fever=.0, others=.03,
                     product=.02, age={"lt_12": 2, "12_17": 2, "18_39": 3, "40_59": 2, "60_plus": 1}),
    "benign_lesion": dict(site={"back": 3, "chest": 2, "face": 3, "arm": 2, "neck": 2, "leg": 1, "abdomen": 1},
                          dur={"gt_6m": 7, "1_6m": 2}, itch=(0, 1), pain=(0, 0), chg={"no": 7, "unsure": 2, "growing": 1},
                          bleed=.04, fever=.0, others=.0, product=.0, age={"18_39": 2, "40_59": 4, "60_plus": 4}),
    "suspicious_lesion": dict(site={"face": 4, "back": 3, "arm": 2, "chest": 2, "leg": 2, "neck": 1, "hand": 1},
                              dur={"1_6m": 4, "gt_6m": 4, "1_4w": 1}, itch=(0, 1), pain=(0, 1),
                              chg={"growing": 3, "changing_color": 2, "changing_shape": 2, "no": 2, "unsure": 1},
                              bleed=.3, fever=.0, others=.0, product=.0, age={"40_59": 4, "60_plus": 5, "18_39": 1}),
    "other": dict(site={k: 1 for k in ["face", "scalp", "neck", "chest", "back", "abdomen", "arm", "hand", "leg", "foot", "groin"]},
                  dur={"lt_1w": 2, "1_4w": 3, "1_6m": 2, "gt_6m": 2}, itch=(0, 3), pain=(0, 2),
                  chg={"no": 2, "spreading": 2, "unsure": 2, "growing": 1}, bleed=.1, fever=.05, others=.05, product=.1,
                  age={k: 1 for k in AGE_BANDS}),
}

PAD_REGION = {"FACE": "face", "NOSE": "face", "LIP": "face", "EAR": "face", "FOREHEAD": "face", "CHEEK": "face",
              "SCALP": "scalp", "NECK": "neck", "CHEST": "chest", "BACK": "back", "ABDOMEN": "abdomen",
              "ARM": "arm", "FOREARM": "arm", "HAND": "hand", "THIGH": "leg", "FOOT": "foot"}
SDN_REGION = {"hand": "hand", "finger": "hand", "palm": "hand", "wrist": "hand", "face": "face", "cheek": "face",
              "forehead": "face", "nose": "face", "chin": "face", "lip": "face", "ear": "face", "scalp": "scalp", "head": "scalp",
              "neck": "neck", "chest": "chest", "breast": "chest", "back": "back", "abdomen": "abdomen", "stomach": "abdomen",
              "waist": "abdomen", "arm": "arm", "forearm": "arm", "elbow": "arm", "axilla": "arm", "shoulder": "arm",
              "leg": "leg", "thigh": "leg", "knee": "leg", "buttock": "groin", "groin": "groin", "genital": "groin",
              "foot": "foot", "toe": "foot", "ankle": "foot", "sole": "foot", "nail": "nails"}


def _w(rng: random.Random, d: dict):
    ks = list(d); return rng.choices(ks, weights=[d[k] for k in ks])[0]


def age_band(age) -> str | None:
    try:
        a = float(age)
    except (TypeError, ValueError):
        return None
    if a != a:
        return None
    return "lt_12" if a < 12 else "12_17" if a < 18 else "18_39" if a < 40 else "40_59" if a < 60 else "60_plus"


def sample(label: str, rng: random.Random) -> dict:
    p = PRIORS[label]
    return dict(body_site=_w(rng, p["site"]), duration=_w(rng, p["dur"]), itch=rng.randint(*p["itch"]),
                pain=rng.randint(*p["pain"]), changing=_w(rng, p["chg"]), bleeding_or_crusting=rng.random() < p["bleed"],
                fever_or_unwell=rng.random() < p["fever"], others_affected=rng.random() < p["others"],
                new_product_or_exposure=rng.random() < p["product"], age_band=_w(rng, p["age"]), skin_tone="unknown",
                free_text="", source="tap")


def _tf(v) -> bool | None:
    s = str(v).strip().upper()
    return True if s == "TRUE" else False if s == "FALSE" else None


def from_row(row: dict, label: str, rng: random.Random) -> tuple[dict, bool, list[str]]:
    """→ (answers, real_q, imputed_fields)."""
    a = sample(label, rng)
    imputed = list(a.keys())
    real = False
    if row.get("source") == "pad_ufes20":
        real = True
        site = PAD_REGION.get(str(row.get("pad_region", "")).upper())
        for k, v in [("body_site", site), ("age_band", age_band(row.get("pad_age")))]:
            if v: a[k] = v; imputed.remove(k)
        itch, hurt, grew, changed, bleed = (_tf(row.get(f"pad_{k}")) for k in ["itch", "hurt", "grew", "changed", "bleed"])
        if itch is not None: a["itch"] = rng.choice([1, 2]) if itch else 0; imputed.remove("itch")
        if hurt is not None: a["pain"] = rng.choice([1, 2]) if hurt else 0; imputed.remove("pain")
        if grew is not None or changed is not None:
            a["changing"] = "growing" if grew else ("changing_shape" if changed else "no"); imputed.remove("changing")
        if bleed is not None: a["bleeding_or_crusting"] = bleed; imputed.remove("bleeding_or_crusting")
        if row.get("skin_tone") in {"fitz_1_2", "fitz_3_4", "fitz_5_6"}: a["skin_tone"] = row["skin_tone"]; imputed.remove("skin_tone")
    elif row.get("source") == "skindisnet":
        real = True
        loc = str(row.get("sdn_location", "")).strip().lower()
        site = next((v for k, v in SDN_REGION.items() if k in loc), None)
        for k, v in [("body_site", site), ("age_band", age_band(row.get("sdn_age")))]:
            if v: a[k] = v; imputed.remove(k)
    elif row.get("source") == "scin":
        real = True
        sites = {SCIN_PART[x] for x in str(row.get("scin_parts") or "").split(";") if x in SCIN_PART}
        dur = SCIN_DURATION.get(str(row.get("scin_duration")))
        age = SCIN_AGE.get(str(row.get("scin_age")))
        for k, v in [("body_site", next(iter(sites)) if len(sites) == 1 else None), ("duration", dur), ("age_band", age)]:
            if v: a[k] = v; imputed.remove(k)
        sym = {x for x in str(row.get("scin_symptoms") or "").split(";") if x}
        if sym:   # an unticked symptom counts as "no" only when the person ticked something on that question
            none = "no_relevant_experience" in sym
            a["itch"] = 0 if none or "itching" not in sym else rng.choice([1, 2]); imputed.remove("itch")   # SCIN has presence only
            a["pain"] = 0 if none or "pain" not in sym else rng.choice([1, 2]); imputed.remove("pain")
            a["bleeding_or_crusting"] = (not none) and "bleeding" in sym; imputed.remove("bleeding_or_crusting")
            if "increasing_size" in sym or "darkening" in sym:
                a["changing"] = "growing" if "increasing_size" in sym else "changing_color"; imputed.remove("changing")
        if _tf(row.get("scin_fever")):
            a["fever_or_unwell"] = True; imputed.remove("fever_or_unwell")
        if row.get("skin_tone") in {"fitz_1_2", "fitz_3_4", "fitz_5_6"}: a["skin_tone"] = row["skin_tone"]; imputed.remove("skin_tone")
    for k in ("free_text", "source"):
        if k in imputed: imputed.remove(k)
    return a, real, imputed


# SCIN self-reported fields → contracts enums (ambiguous ones left unanswered, i.e. imputed and marked)
SCIN_PART = {"arm": "arm", "palm": "hand", "back_of_hand": "hand", "torso_front": "chest", "torso_back": "back",
             "genitalia_or_groin": "groin", "leg": "leg", "foot_top_or_side": "foot", "foot_sole": "foot"}
SCIN_DURATION = {"ONE_DAY": "lt_1w", "LESS_THAN_ONE_WEEK": "lt_1w", "ONE_TO_FOUR_WEEKS": "1_4w", "ONE_TO_THREE_MONTHS": "1_6m",
                 "MORE_THAN_ONE_YEAR": "gt_6m", "MORE_THAN_FIVE_YEARS": "gt_6m"}
SCIN_AGE = {"AGE_18_TO_29": "18_39", "AGE_30_TO_39": "18_39", "AGE_40_TO_49": "40_59", "AGE_50_TO_59": "40_59",
            "AGE_60_TO_69": "60_plus", "AGE_70_TO_79": "60_plus", "AGE_80_OR_ABOVE": "60_plus"}
