import csv
import re
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "parseq"))


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


def main():
    model = torch.hub.load(str(Path(__file__).parent / "parseq"), "parseq", source="local", pretrained=True).eval()
    img_transform = torch.hub.load(str(Path(__file__).parent / "parseq"), "_get_transform", source="local", img_size=model.hparams.img_size) \
        if False else None

    from strhub.data.module import SceneTextDataModule
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)

    rows = []
    with open("datasets/str_manifest_combined.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["text"].strip():
                rows.append((r["crop_path"], r["text"].strip()))

    # use last 150 (val GT) as eval slice to avoid overlap bias, but really all are unseen to PARSeq anyway
    total, exact, cer_sum = 0, 0, 0.0
    for path, gt in rows:
        if not Path(path).exists():
            continue
        img = Image.open(path).convert("RGB")
        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            logits = model(x)
        pred = model.tokenizer.decode(logits.softmax(-1))[0][0]
        gtn, predn = norm(gt), norm(pred)
        if not gtn:
            continue
        total += 1
        d = levenshtein(gtn, predn) / max(1, len(gtn))
        cer_sum += d
        exact += (d == 0.0)

    print(f"PARSeq zero-shot: n={total} exact={exact}/{total}={exact/max(1,total):.3f} meanCER={cer_sum/max(1,total):.3f}")


if __name__ == "__main__":
    main()
