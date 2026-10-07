"""CV dataset + transforms. Eval preprocessing is mirrored in android CvClassifier.kt (assets/cv/preprocess.json):
center-crop to square (shorter side) → resize 384×384 bilinear → RGB float /255 → [Shades-of-Gray, if the checkpoint says
"color_constancy": "sog6"] → (x-mean)/std, NHWC on device.
"""
from __future__ import annotations

import io
import random

import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import functional as F

from ml.common.paths import REPO

SIZE = 384
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def center_square(im: Image.Image) -> Image.Image:
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))


class PhoneDegrade:
    """Phone-realistic corruption: mild blur, JPEG re-compression. Applied with probability p each."""

    def __init__(self, p_blur=0.25, p_jpeg=0.4):
        self.p_blur, self.p_jpeg = p_blur, p_jpeg

    def __call__(self, im):
        if random.random() < self.p_blur:
            im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.6)))
        if random.random() < self.p_jpeg:
            b = io.BytesIO(); im.save(b, "JPEG", quality=random.randint(35, 90)); b.seek(0)
            im = Image.open(b).convert("RGB")
        return im


class ShadesOfGray:
    """Shades-of-Gray colour constancy (Finlayson & Trezzi 2004, p = 6): each channel's illuminant is the Minkowski p-mean of
    the image; dividing by it (scaled so a neutral illuminant leaves the image unchanged) removes lighting / camera colour
    casts. On the [0,1] CHW tensor after resizing, in float64 — CvClassifier.kt does the same maths."""

    def __init__(self, p: int = 6):
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        d = x.double()
        e = d.pow(self.p).mean(dim=(1, 2)).pow(1.0 / self.p)
        e = e / e.norm().clamp_min(1e-12)
        return (d / (e * 3 ** 0.5).clamp_min(1e-6)[:, None, None]).clamp(0, 1).float()


def _cc(cc: str | None) -> list:
    if cc in (None, "none"):
        return []
    assert cc == "sog6", cc
    return [ShadesOfGray(6)]


def eval_transform(cc: str | None = None):
    return T.Compose([T.Lambda(center_square), T.Resize((SIZE, SIZE), interpolation=T.InterpolationMode.BILINEAR),
                      T.ToTensor(), *_cc(cc), T.Normalize(MEAN, STD)])


def train_transform(cc: str | None = None):
    return T.Compose([
        T.RandomResizedCrop(SIZE, scale=(0.45, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(25)], p=0.5),
        T.ColorJitter(0.3, 0.3, 0.2, 0.03),   # lighting / white balance; hue kept small so skin tone isn't remapped
        PhoneDegrade(),
        T.ToTensor(), *_cc(cc), T.Normalize(MEAN, STD),
    ])


class SkinDS(Dataset):
    def __init__(self, df: pd.DataFrame, classes: list[str], tf, random_labels: bool = False, seed: int = 0):
        self.paths = [str(REPO / p) if not str(p).startswith("/") else str(p) for p in df.img]
        idx = {k: i for i, k in enumerate(classes)}
        self.y = [idx[k] for k in df.label]
        if random_labels:  # C1(b) leakage probe: labels shuffled → val acc must stay near chance
            rnd = random.Random(seed); self.y = self.y[:]; rnd.shuffle(self.y)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        im = Image.open(self.paths[i]).convert("RGB")
        return self.tf(im), self.y[i]


def preprocess_numpy(path: str):
    """Exact eval preprocessing as numpy NHWC float32 — used for TFLite parity (C3) and Kotlin parity dumps (C4)."""
    import numpy as np
    im = Image.open(path).convert("RGB")
    im = F.resize(center_square(im), [SIZE, SIZE], interpolation=T.InterpolationMode.BILINEAR)
    x = np.asarray(im, dtype=np.float32) / 255.0
    return ((x - np.array(MEAN, np.float32)) / np.array(STD, np.float32))[None]
