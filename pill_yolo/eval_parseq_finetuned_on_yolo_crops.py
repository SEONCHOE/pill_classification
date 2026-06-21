import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

sys.path.insert(0, str(Path(__file__).parent / "parseq"))
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

CKPT = "outputs/parseq-tiny/2026-06-20_15-17-55/checkpoints/epoch=48-step=5292-val_accuracy=48.6842-val_NED=66.6859.ckpt"


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
    name = Path(name).stem
    return re.sub(r"_\d+$", "", name)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_from_checkpoint(CKPT).eval().to(device)
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    crop_dir = Path("ocr_check_full")
    crops = sorted([p for p in crop_dir.glob("*.png")])
    pred_by_image = defaultdict(list)
    with torch.inference_mode():
        for cp in crops:
            img = Image.open(cp).convert("RGB")
            x = transform(img).unsqueeze(0).to(device)
            logits = model(x).softmax(-1)
            pred, _ = model.tokenizer.decode(logits)
            pred_by_image[stem_of(cp.name)].append(pred[0])

    import csv
    gt_by_image = defaultdict(list)
    with open("datasets/str_manifest_combined.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            gt_by_image[stem_of(row["crop_path"])].append(row["text"].strip())

    common = set(gt_by_image) & set(pred_by_image)
    only_gt = set(gt_by_image) - set(pred_by_image)
    print(f"GT 있는 이미지: {len(gt_by_image)}, 우리 YOLO+PARSeq 있는 이미지: {len(pred_by_image)}, 공통: {len(common)}")
    print(f"GT는 있는데 박스 자체를 놓친 이미지: {len(only_gt)}")

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
            exact += (best_cer == 0.0)

    print(f"\n[YOLO crop + PARSeq finetuned] exact match : {exact}/{total_gt} = {exact/max(1,total_gt):.3f}")
    print(f"[YOLO crop + PARSeq finetuned] 평균 CER     : {cer_sum/max(1,total_gt):.3f}")


if __name__ == "__main__":
    main()
