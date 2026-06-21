import cv2
import os
import random

random.seed(42)

LABELS_DIR = "datasets/imprint/labels/train"
IMAGES_DIR = "datasets/imprint/images/train"
OUT_DIR = "qa_label_check"
os.makedirs(OUT_DIR, exist_ok=True)

COLORS = {0: (0, 255, 0), 1: (0, 0, 255)}  # imprint=green, score_line=red
NAMES = {0: "imprint", 1: "score_line"}

label_files = [f for f in os.listdir(LABELS_DIR) if f.endswith(".txt") and os.path.getsize(os.path.join(LABELS_DIR, f)) > 0]
sample = random.sample(label_files, min(16, len(label_files)))

tiles = []
for lf in sample:
    img_name = lf.replace(".txt", ".jpg")
    img_path = os.path.join(IMAGES_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    h, w = img.shape[:2]
    with open(os.path.join(LABELS_DIR, lf)) as f:
        for line in f:
            parts = line.split()
            cls = int(parts[0])
            xc, yc, bw, bh = map(float, parts[1:5])
            x1 = int((xc - bw / 2) * w)
            y1 = int((yc - bh / 2) * h)
            x2 = int((xc + bw / 2) * w)
            y2 = int((yc + bh / 2) * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), COLORS.get(cls, (255, 255, 0)), 2)
            cv2.putText(img, NAMES.get(cls, str(cls)), (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS.get(cls, (255, 255, 0)), 1)
    img = cv2.resize(img, (300, 300))
    tiles.append(img)
    cv2.imwrite(os.path.join(OUT_DIR, img_name), img)

# build contact sheet 4x4
import numpy as np
rows = []
for i in range(0, len(tiles), 4):
    row_tiles = tiles[i:i+4]
    while len(row_tiles) < 4:
        row_tiles.append(np.zeros((300, 300, 3), dtype="uint8"))
    rows.append(cv2.hconcat(row_tiles))
sheet = cv2.vconcat(rows)
cv2.imwrite(os.path.join(OUT_DIR, "_contact_sheet.jpg"), sheet)
print("saved", len(tiles), "tiles to", OUT_DIR)
