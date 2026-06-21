import csv
import random
from pathlib import Path

random.seed(42)
rows = []
with open("datasets/str_manifest_combined.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        text = r["text"].strip()
        path = Path(r["crop_path"]).resolve()
        if text and path.exists():
            rows.append((str(path), text))

random.shuffle(rows)
n_val = int(len(rows) * 0.15)
val_rows, train_rows = rows[:n_val], rows[n_val:]

Path("parseq_gt").mkdir(exist_ok=True)
with open("parseq_gt/train_gt.txt", "w", encoding="utf-8") as f:
    for p, t in train_rows:
        f.write(f"{p} {t}\n")
with open("parseq_gt/val_gt.txt", "w", encoding="utf-8") as f:
    for p, t in val_rows:
        f.write(f"{p} {t}\n")

print(f"train={len(train_rows)} val={len(val_rows)}")
