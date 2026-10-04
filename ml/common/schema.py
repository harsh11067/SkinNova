"""Pydantic mirrors of docs/contracts.md. Kotlin `model/` must match these shapes exactly."""
from __future__ import annotations

import json
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from ml.common.paths import LABELS

TIERS = ["LOW", "MODERATE", "HIGH", "URGENT"]
TIER_RANK = {t: i for i, t in enumerate(TIERS)}
LIKELIHOODS = ["higher", "possible", "less_likely"]
UNCERTAINTY = ["low", "moderate", "high"]

BODY_SITES = ["face", "scalp", "neck", "chest", "back", "abdomen", "arm", "hand", "leg", "foot", "groin", "nails", "other"]
DURATIONS = ["lt_1w", "1_4w", "1_6m", "gt_6m"]
CHANGING = ["no", "growing", "changing_color", "changing_shape", "spreading", "unsure"]
AGE_BANDS = ["lt_12", "12_17", "18_39", "40_59", "60_plus"]
SKIN_TONES = ["fitz_1_2", "fitz_3_4", "fitz_5_6", "unknown"]


def load_labels() -> dict:
    return json.loads(LABELS.read_text())


def label_keys() -> list[str]:
    return [c["key"] for c in load_labels()["classes"]]


def max_tier(*tiers: str) -> str:
    return max((t for t in tiers if t), key=lambda t: TIER_RANK[t], default="LOW")


class QuestionnaireAnswers(BaseModel):
    body_site: Literal[tuple(BODY_SITES)]  # type: ignore[valid-type]
    duration: Literal[tuple(DURATIONS)]  # type: ignore[valid-type]
    itch: int = Field(ge=0, le=3)
    pain: int = Field(ge=0, le=3)
    changing: Literal[tuple(CHANGING)]  # type: ignore[valid-type]
    bleeding_or_crusting: bool
    fever_or_unwell: bool
    others_affected: bool
    new_product_or_exposure: bool
    age_band: Literal[tuple(AGE_BANDS)]  # type: ignore[valid-type]
    skin_tone: Literal[tuple(SKIN_TONES)] = "unknown"  # type: ignore[valid-type]
    free_text: str = Field(default="", max_length=200)
    source: Literal["tap", "voice_confirmed"] = "tap"


class Category(BaseModel):
    key: str
    likelihood: Literal["higher", "possible", "less_likely"]
    why: str = Field(max_length=200)


class Uncertainty(BaseModel):
    level: Literal["low", "moderate", "high"]
    reasons: list[str] = []


class Triage(BaseModel):
    tier: Literal["LOW", "MODERATE", "HIGH", "URGENT"]
    advice: str


class AnalysisOutput(BaseModel):
    possible_categories: list[Category] = Field(min_length=1, max_length=3)
    uncertainty: Uncertainty
    explanation: str
    what_would_help: list[str] = []
    self_care_info: list[str] = []
    triage: Triage
    disagreement_with_image_model: bool = False

    @field_validator("possible_categories")
    @classmethod
    def _unique(cls, v):
        keys = [c.key for c in v]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate category keys")
        return v


class CvScore(BaseModel):
    key: str
    p: float


class TimelineMetrics(BaseModel):
    align_ok: bool
    coin_in_both: bool = False
    area_ratio: Optional[float] = None
    contrast_delta: Optional[float] = None
    noise_contrast: float = 0.0
    confidence: Literal["ok", "low"] = "low"
    lesion_type: bool = False
