"""
clip_filter.py
==============================================================
CLIP zero-shot으로 drug_info_2006 추출 crop을 "알약 사진 vs 텍스트/기타"로 점수화.
  - 라벨 불필요. 각 crop을 알약 프롬프트 / 비알약 프롬프트와의 유사도로 점수.
  - 확실한 알약(통과) / 확실한 비알약(폐기) / 회색지대(Gemini 검증 후보) 3분할.

실행:
    python clip_filter.py --crops_dir pilot_out_100/crops --out clip_scores.csv
==============================================================
"""
import argparse
import csv
from pathlib import Path

import torch
import open_clip
from PIL import Image

PILL_PROMPTS = [
    "a close-up photo of a single pill or capsule",
    "a photograph of a medicine tablet on a plain background",
    "a photo of a round white pill",
    "a photo of a colored capsule",
]
NEG_PROMPTS = [
    "a patch of printed text and numbers",
    "a blank background with no object",
    "a small icon or logo badge",
    "a fragment of a document with korean text",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops_dir", required=True)
    ap.add_argument("--out", default="clip_scores.csv")
    ap.add_argument("--low", type=float, default=0.40, help="이 점수 미만=폐기")
    ap.add_argument("--high", type=float, default=0.60, help="이 점수 이상=통과")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(dev).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    with torch.inference_mode():
        pill_tok = tokenizer(PILL_PROMPTS).to(dev)
        neg_tok = tokenizer(NEG_PROMPTS).to(dev)
        pill_emb = model.encode_text(pill_tok).float()
        neg_emb = model.encode_text(neg_tok).float()
        pill_emb /= pill_emb.norm(dim=-1, keepdim=True)
        neg_emb /= neg_emb.norm(dim=-1, keepdim=True)
        pill_centroid = pill_emb.mean(0, keepdim=True)
        neg_centroid = neg_emb.mean(0, keepdim=True)

    crops = sorted(Path(args.crops_dir).glob("*.png"))
    rows = []
    counts = {"pass": 0, "gray": 0, "drop": 0}
    with torch.inference_mode():
        for cp in crops:
            img = preprocess(Image.open(cp).convert("RGB")).unsqueeze(0).to(dev)
            emb = model.encode_image(img).float()
            emb /= emb.norm(dim=-1, keepdim=True)
            sp = (emb @ pill_centroid.T).item()
            sn = (emb @ neg_centroid.T).item()
            # softmax 2-way 확률(알약 쪽)
            score = torch.softmax(torch.tensor([sp, sn]) / 0.05, dim=0)[0].item()
            if score >= args.high:
                verdict = "pass"
            elif score < args.low:
                verdict = "drop"
            else:
                verdict = "gray"
            counts[verdict] += 1
            rows.append([cp.name, f"{score:.4f}", verdict])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["crop", "pill_score", "verdict"])
        w.writerows(rows)

    n = len(rows)
    print(f"total={n}  pass={counts['pass']} ({counts['pass']/n:.1%})  "
          f"gray={counts['gray']} ({counts['gray']/n:.1%})  drop={counts['drop']} ({counts['drop']/n:.1%})")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
