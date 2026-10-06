# SkinNova data card

dataset_rev: `a2a7226bedaa` · label map v2

## Classes
Trainable: eczema_atopic, contact_dermatitis, tinea, scabies, acne, psoriasis, vitiligo, benign_lesion, suspicious_lesion, other
Dropped by D5 (< 150 distinct images): {'seborrheic_dermatitis': 69}

## Images per split and class
| label              |   train |   val |   test |   external_test |
|:-------------------|--------:|------:|-------:|----------------:|
| acne               |     632 |   126 |    118 |               0 |
| benign_lesion      |    6151 |  1206 |   1242 |               0 |
| contact_dermatitis |     531 |   144 |    134 |               0 |
| eczema_atopic      |    1425 |   309 |    261 |              91 |
| other              |    5646 |  1117 |   1129 |               0 |
| psoriasis          |     634 |   123 |    126 |               0 |
| scabies            |     249 |    48 |     53 |              10 |
| suspicious_lesion  |    5007 |  1003 |    993 |               0 |
| tinea              |     961 |   193 |    189 |              26 |
| vitiligo           |     476 |    94 |     97 |               9 |
| TOTAL              |   21712 |  4363 |   4342 |             136 |

## Images per split and source
| source        |   train |   val |   test |   external_test |
|:--------------|--------:|------:|-------:|----------------:|
| mgmitesh      |    8556 |  1711 |   1713 |               0 |
| pacificrm     |    8059 |  1609 |   1609 |               0 |
| pad_ufes20    |    1582 |   297 |    316 |               0 |
| scin          |    2907 |   609 |    583 |               0 |
| skindiseasebd |       0 |     0 |      0 |             136 |
| skindisnet    |     608 |   137 |    121 |               0 |

## Pipeline counts
- enumerated 52739 files (pre-augmented copies already skipped) → labeled 51392
- label decisions: {'folder': 38100, 'slug': 13292, 'exclude_folder': 636, 'slug_exclude': 390, 'slug_only_nomatch': 321}
- near-identical clusters: 30755 (7170 with >1 copy, 1253 spanning two sources) → kept 30622
- dropped: {'': 50769, 'label_conflict': 574, 'touches_external': 25}
- copy detection (DINOv2-s kNN candidates → ORB/RANSAC ≥ 25 inliers): 62744 verified copy pairs
  - calibration (test D2): planted-copy recall 0.995 (pHash alone 0.907); borderline pairs inspected: 0/12 false merges

## Sources and licenses
- **mgmitesh**: CC BY 4.0 (Kaggle page). Mixed provenance: DermNet-topic files, ISIC dermoscopy, web images. Pre-augmented copies removed.
- **pacificrm**: Kaggle mirror of the DermNet 23-class set. DermNet NZ images: non-commercial educational use with attribution. Used because it is the only psoriasis/vitiligo/scabies source found.
- **skindisnet**: CC BY-NC 4.0 (Mendeley yj3md44hxg v2).
- **pad_ufes20**: CC BY 4.0 (Mendeley zr7vgbcyr2).
- **skindiseasebd**: CC BY-NC 4.0 (Mendeley 9ggd3shdr7). Augmentation-only release; one image per near-duplicate group scored.
- **scin**: CC BY 4.0 (Google Research + Stanford, github.com/google-research-datasets/scin). Crowdsourced US phone photos with dermatologist labels; consensus label (weight ≥ 0.5) only.

## Known limitations
- Most inflammatory-class images are DermNet clinical photos (lighter skin tones over-represented, professional lighting).
- Fitzpatrick labels exist only for PAD-UFES-20; per-tone metrics elsewhere are not possible.
- `other` is heterogeneous (normal skin, bullous, lupus, lichen, warts, drug eruptions, rosacea, candida ...).
- Labels come from folder names or DermNet topic slugs, not from a dermatologist review of each image (diy.md D8 spot-check).
