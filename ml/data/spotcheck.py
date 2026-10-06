"""test.md D8 helper: 20 random train images per class → reports/spotcheck/<class>.jpg contact sheets (for Harsh's review)."""
import json
import pandas as pd
from PIL import Image, ImageDraw
from ml.common.paths import REPO, REPORTS, SPLITS

def main(n=20, seed=11):
    out = REPORTS / "spotcheck"; out.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(SPLITS / "train.csv")
    for lab, g in tr.groupby("label"):
        s = g.sample(min(n, len(g)), random_state=seed)
        sheet = Image.new("RGB", (5 * 200, 4 * 220), "white"); d = ImageDraw.Draw(sheet)
        for k, r in enumerate(s.itertuples()):
            im = Image.open(REPO / r.img).convert("RGB"); im.thumbnail((196, 196))
            x, y = (k % 5) * 200, (k // 5) * 220
            sheet.paste(im, (x + 2, y + 2)); d.text((x + 4, y + 202), f"{k:02d} {r.source[:9]} {str(r.source_label)[:14]}", fill="black")
        sheet.save(out / f"{lab}.jpg", quality=85)
        s[["img", "source", "source_label"]].to_csv(out / f"{lab}.csv")
    print("wrote", sorted(p.name for p in out.glob("*.jpg")))

if __name__ == "__main__":
    main()
