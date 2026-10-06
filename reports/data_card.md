# SkinNova data card

dataset_rev: `a138e2d1b593` · label map v2

## Classes
Trainable: eczema_atopic, contact_dermatitis, tinea, scabies, acne, psoriasis, vitiligo, benign_lesion, suspicious_lesion, other
Dropped by D5 (< 150 distinct images): {'seborrheic_dermatitis': 41}

## Images per split and class
| label              |   train |   val |   test |   external_test |
|:-------------------|--------:|------:|-------:|----------------:|
| acne               |     551 |   114 |    113 |               0 |
| benign_lesion      |    6119 |  1202 |   1227 |               0 |
| contact_dermatitis |     153 |    44 |     43 |               0 |
| eczema_atopic      |     860 |   186 |    158 |              91 |
| other              |    4080 |   805 |    812 |               0 |
| psoriasis          |     538 |   107 |    109 |               0 |
| scabies            |     234 |    48 |     50 |              10 |
| suspicious_lesion  |    4968 |   997 |    992 |               0 |
| tinea              |     830 |   157 |    158 |              26 |
| vitiligo           |     472 |    94 |     97 |               9 |
| TOTAL              |   18805 |  3754 |   3759 |             136 |

## Images per split and source
| source        |   train |   val |   test |   external_test |
|:--------------|--------:|------:|-------:|----------------:|
| mgmitesh      |    8149 |  1619 |   1642 |               0 |
| pacificrm     |    8466 |  1701 |   1680 |               0 |
| pad_ufes20    |    1582 |   297 |    316 |               0 |
| skindiseasebd |       0 |     0 |      0 |             136 |
| skindisnet    |     608 |   137 |    121 |               0 |

## Pipeline counts
- enumerated 48138 files (pre-augmented copies already skipped) → labeled 47125
- label decisions: {'folder': 33833, 'slug': 13292, 'slug_exclude': 390, 'slug_only_nomatch': 321, 'exclude_folder': 302}
- near-identical clusters: 26626 (7071 with >1 copy, 1252 spanning two sources) → kept 26495
- dropped: {'': 46530, 'label_conflict': 570, 'touches_external': 25}
- copy detection (DINOv2-s kNN candidates → ORB/RANSAC ≥ 25 inliers): 62648 verified copy pairs
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
