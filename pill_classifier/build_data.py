"""
build_data.py
==============================================================
알약 외형(색/모양) 임베딩 학습용 데이터 준비
  입력: pill_yolo/raw_data/splited_aug/{front,back}/<NDC>_*.jpg (각 NDC 1장씩)
  출력: manifest.csv (ndc,front_path,back_path,split)
        NDC 단위로 train/val 분리(같은 약이 train/val 양쪽에 누수되지 않게)
==============================================================
"""
import csv
import random
import re
from pathlib import Path

RAW = Path(__file__).parent.parent / "pill_yolo" / "raw_data" / "splited_aug"
OUT = Path(__file__).parent / "manifest.csv"
VAL_RATIO = 0.15
SEED = 42


def ndc_of(p: Path):
    m = re.match(r"^([0-9]{5}-[0-9]{4}-[0-9]{2})", p.name)
    return m.group(1) if m else None


def main():
    front_by_ndc = {ndc_of(p): p for p in (RAW / "front").glob("*.jpg") if ndc_of(p)}
    back_by_ndc = {ndc_of(p): p for p in (RAW / "back").glob("*.jpg") if ndc_of(p)}
    ndcs = sorted(set(front_by_ndc) & set(back_by_ndc))
    print(f"front-only: {len(set(front_by_ndc)-set(back_by_ndc))}, "
          f"back-only: {len(set(back_by_ndc)-set(front_by_ndc))}, "
          f"both: {len(ndcs)}")

    random.seed(SEED)
    shuffled = ndcs[:]
    random.shuffle(shuffled)
    n_val = int(len(shuffled) * VAL_RATIO)
    val_ndcs = set(shuffled[:n_val])

    rows = []
    for ndc in ndcs:
        split = "val" if ndc in val_ndcs else "train"
        rows.append([ndc, str(front_by_ndc[ndc]), str(back_by_ndc[ndc]), split])

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ndc", "front_path", "back_path", "split"])
        w.writerows(rows)

    n_train = sum(1 for r in rows if r[3] == "train")
    print(f"train={n_train} classes, val={len(rows)-n_train} classes -> {OUT}")


if __name__ == "__main__":
    main()
