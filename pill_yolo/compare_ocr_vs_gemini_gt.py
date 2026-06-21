import csv
import re
from collections import defaultdict
from pathlib import Path


def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n]


def norm(s):
    return re.sub(r"\s+", "", s).upper()


def stem_of(name):
    # crop filename like front_xxx_0.png or front_xxx_0
    name = Path(name).stem
    return re.sub(r"_\d+$", "", name)


# load Gemini GT, grouped by image stem
gt_by_image = defaultdict(list)
with open("datasets/str_val_gt/str_manifest.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        stem = stem_of(row["crop_path"])
        gt_by_image[stem].append(row["text"].strip())

# load our OCR predictions, grouped by image stem
pred_by_image = defaultdict(list)
for ln in Path("ocr_check_full/_ocr_results.txt").read_text().splitlines():
    if not ln.strip():
        continue
    fname, confpart, ocrpart = ln.split("\t")
    stem = stem_of(fname)
    ocr_text = ocrpart.split("=", 1)[1].strip("'")
    if ocr_text.strip():
        pred_by_image[stem].append(ocr_text)

common = set(gt_by_image) & set(pred_by_image)
only_gt = set(gt_by_image) - set(pred_by_image)
print(f"GT 있는 이미지: {len(gt_by_image)}, 우리 OCR 있는 이미지: {len(pred_by_image)}, 공통: {len(common)}")
print(f"GT는 있는데 우리가 아무것도 못 찾은 이미지: {len(only_gt)}")

total_gt, exact, cer_sum = 0, 0, 0.0
for stem in common:
    preds = [norm(p) for p in pred_by_image[stem]]
    for gt in gt_by_image[stem]:
        gtn = norm(gt)
        if not gtn:
            continue
        total_gt += 1
        best_cer = min((levenshtein(gtn, p) / max(1, len(gtn)) for p in preds), default=1.0)
        cer_sum += best_cer
        if best_cer == 0.0:
            exact += 1

print(f"\n공통 이미지 기준 GT 텍스트 단위:")
print(f"  exact match (이미지 내 best-match) : {exact}/{total_gt} = {exact/max(1,total_gt):.3f}")
print(f"  평균 CER (최선 매칭)               : {cer_sum/max(1,total_gt):.3f}")
print(f"\n참고: only_gt(우리가 박스 자체를 놓친 이미지) 비율 = {len(only_gt)}/{len(gt_by_image)} = {len(only_gt)/max(1,len(gt_by_image)):.3f}")
