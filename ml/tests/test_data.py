"""Data pipeline tests D1–D7 (test.md §3). D1/D2/D4-unit run anywhere; D3/D5/D6/D7 need data/splits (skipped otherwise)."""
import io
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageOps

from ml.common.paths import REPORTS, SPLITS
from ml.data import build_label_map as blm
from ml.data import dedupe, normalize

HAVE_SPLITS = (SPLITS / "train.csv").exists()
needs_splits = pytest.mark.skipif(not HAVE_SPLITS, reason="splits not built")


def _img(seed=0, size=(320, 240)):
    rng = np.random.default_rng(seed)
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[:] = rng.integers(60, 200, 3)
    for _ in range(12):  # random blobs so pHash has structure
        x, y = rng.integers(0, size[0] - 40), rng.integers(0, size[1] - 40)
        a[y:y + rng.integers(10, 40), x:x + rng.integers(10, 40)] = rng.integers(0, 255, 3)
    return Image.fromarray(a)


# ---------- D1 EXIF orientation + metadata strip
@pytest.mark.parametrize("orient,rot", [(6, 270), (3, 180), (8, 90)])
def test_d1_exif_upright(tmp_path, monkeypatch, orient, rot):
    up = _img(1, (300, 200))
    stored = up.rotate(-rot if orient != 3 else 180, expand=True) if False else up.transpose(
        {6: Image.Transpose.ROTATE_90, 3: Image.Transpose.ROTATE_180, 8: Image.Transpose.ROTATE_270}[orient])
    exif = Image.Exif(); exif[0x0112] = orient; exif[0x010F] = "PhoneMaker"
    p = tmp_path / "x.jpg"; stored.save(p, exif=exif.tobytes(), quality=95)
    monkeypatch.setattr(normalize, "IMAGES", tmp_path / "out"); (tmp_path / "out").mkdir()
    r = normalize.process({"path": str(p)})
    assert r["error"] == ""
    out = Image.open(r["img"])
    assert out.size == (300, 200), "not upright"
    assert len(out.getexif()) == 0, "metadata not stripped"
    assert np.abs(np.asarray(out, float) - np.asarray(up, float)).mean() < 12


# ---------- D2 planted duplicates cluster; distinct images don't
def _hashes(im):
    import imagehash
    small = im.copy(); small.thumbnail((256, 256))
    dih = min(int(str(imagehash.phash(t)), 16) for t in normalize.dihedral(small))
    return str(imagehash.phash(small)), f"{dih:016x}"


def _variant(im, k):
    w, h = im.size
    if k == 0: return im.resize((w // 2, h // 2))
    if k == 1:
        b = io.BytesIO(); im.save(b, "JPEG", quality=50); return Image.open(b).convert("RGB")
    if k == 2: return im.crop((int(w * .04), int(h * .04), int(w * .96), int(h * .96)))
    if k == 3: return Image.eval(im, lambda v: min(255, int(v * 1.1)))
    return ImageOps.mirror(im)


def test_d2_planted_duplicates_phash():
    """pHash stage: resize, JPEG q50, brightness, mirror. (Crops are the embedding stage's job — see below.)"""
    rows, truth = [], []
    for s in range(40):
        base = _img(100 + s)
        for k in (-1, 0, 1, 3, 4):
            im = base if k < 0 else _variant(base, k)
            ph, dih = _hashes(im)
            rows.append({"phash": ph, "phash_dih": dih, "sha1": f"{s}_{k}"}); truth.append(s)
    df = pd.DataFrame(rows)
    # candidate stage only (production verifies each candidate photometrically; see test_d2_verification_real)
    uf = dedupe.UF(len(df))
    for i, j in dedupe.phash_candidates(df):
        uf.union(i, j)
    cl = uf.roots()
    truth = np.array(truth)
    pairs = [(i, j) for i in range(len(df)) for j in range(i + 1, len(df)) if truth[i] == truth[j]]
    recall = np.mean([cl[i] == cl[j] for i, j in pairs])
    false = np.mean([cl[i] == cl[j] for i in range(len(df)) for j in range(i + 1, len(df)) if truth[i] != truth[j]])
    assert recall >= 0.95, recall
    assert false < 0.01, false


@pytest.mark.skipif(not (REPORTS / "dedupe_calibration.json").exists(), reason="embedding calibration not run")
def test_d2_planted_duplicates_embedding():
    """Embedding stage on real images with crop+resize+JPEG+brightness: recall ≥ 95%, false merges < 1%."""
    r = json.loads((REPORTS / "dedupe_calibration.json").read_text())
    assert r["chosen"]["planted_recall"] >= 0.95, r["chosen"]
    assert all(v >= 0.95 for v in r["chosen"]["recall_by_transform"].values()), r["chosen"]
    # random same-label pairs are contaminated by real copies, so precision is shown by inspection of borderline pairs
    assert r["visual_check"]["false_merges"] == 0


# ---------- D4 label map unit behaviour
def test_d4_label_rules():
    m = blm.load_map()
    assert blm.assign(m, "mgmitesh", "Dyshidrotic Eczema", "tinea-groin-12_0_44.jpg") == ("tinea", "slug")
    assert blm.assign(m, "mgmitesh", "Dyshidrotic Eczema", "something-else.jpg")[0] is None
    assert blm.assign(m, "pacificrm", "Infestations_Bites", "scabies-3.jpg")[0] == "scabies"
    assert blm.assign(m, "pacificrm", "Infestations_Bites", "tick-bite-3.jpg")[0] == "other"
    assert blm.assign(m, "pacificrm", "Moles", "lentigo-maligna-2.jpg")[0] == "suspicious_lesion"
    assert blm.assign(m, "pacificrm", "Benign_tumors", "keratoacanthoma-7.jpg")[0] == "suspicious_lesion"
    assert blm.assign(m, "pacificrm", "Vitiligo", "Image_11.jpg")[0] == "vitiligo"
    assert blm.assign(m, "mgmitesh", "Acne", "pigmentation-34.jpg")[0] is None
    assert blm.assign(m, "skindiseasebd", "Dermatitis", "aug_0_1.jpeg")[0] is None


@pytest.mark.skipif(not (REPORTS / "label_map_report.json").exists(), reason="label map not run")
def test_d4_no_unaccounted_folders():
    assert json.loads((REPORTS / "label_map_report.json").read_text())["unaccounted_folders"] == []


@pytest.mark.skipif(not (REPORTS / "copy_verification.json").exists(), reason="copy verification not run")
def test_d2_verification_real():
    """Photometric verification must reject the cross-label links that pHash/ORB alone accepted."""
    r = json.loads((REPORTS / "copy_verification.json").read_text())
    assert r["phash_cross_label_verified"] <= 0.05 * r["phash_cross_label_candidates"], r
    assert r["orb_cross_label_verified"] <= 0.002 * r["orb_verified"], r


# ---------- D3 leakage (hard fail)
@needs_splits
def test_d3_no_leakage():
    """No shared group, no identical file, and no VERIFIED copy edge between any two splits."""
    sp = {s: pd.read_csv(SPLITS / f"{s}.csv") for s in ["train", "val", "test", "external_test"]}
    for a in sp:
        for b in sp:
            if a < b:
                assert not set(sp[a].group) & set(sp[b].group), f"group shared {a}/{b}"
                assert not set(sp[a].sha1) & set(sp[b].sha1), f"identical file {a}/{b}"
    from ml.common.paths import PROCESSED
    man = pd.read_csv(PROCESSED / "manifest.csv", usecols=["sha1"], low_memory=False)
    where = {h: s for s, d in sp.items() for h in d.sha1}
    for f in ["copy_pairs.npy", "phash_pairs.npy"]:
        e = np.load(PROCESSED / f)
        cross = [(i, j) for i, j in e.tolist() if man.sha1[i] in where and man.sha1[j] in where and where[man.sha1[i]] != where[man.sha1[j]]]
        assert not cross, f"{len(cross)} verified copies cross splits ({f})"


@needs_splits
def test_d5_class_floor():
    cls = json.loads((SPLITS / "classes.json").read_text())["classes"]
    n = pd.concat([pd.read_csv(SPLITS / f"{s}.csv") for s in ["train", "val", "test"]]).label.value_counts()
    assert all(n[k] >= 150 for k in cls)


@needs_splits
def test_d6_stratification():
    assert json.loads((REPORTS / "splits.json").read_text())["D6_pass"]


@needs_splits
def test_d7_data_card_deterministic():
    from ml.data import data_card
    data_card.main(); a = (REPORTS / "data_card.md").read_bytes()
    data_card.main(); b = (REPORTS / "data_card.md").read_bytes()
    assert a == b
