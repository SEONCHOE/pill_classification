"""
infer_pipeline.py
==============================================================
알약 각인 인식 엔드투엔드 추론 파이프라인
  알약 이미지 -> YOLO26(각인 영역 탐지) -> PARSeq(STR, fine-tuned) -> lexicon 보정 -> 텍스트

주의: 이건 "각인 텍스트 인식"까지만 하는 파이프라인이다.
      각인 텍스트만으로는 알약을 완전히 식별할 수 없음(같은 각인의 다른 외형 약 존재) ->
      최종 알약 식별에는 별도의 외형 분류 단계(색/모양/크기, 아직 미구현)가 반드시 더 필요하다.
      이 스크립트의 출력은 그 분류기의 입력(후보 좁히기 + lexicon)으로 쓰는 용도다.

실행:
    python infer_pipeline.py --image path/to/pill.jpg
    python infer_pipeline.py --images_dir path/to/dir
==============================================================
"""
import argparse
import re
import sys
from pathlib import Path

import torch
from PIL import Image
from ultralytics import YOLO

# load_from_checkpoint 내부에서 쓰는 torch.load가 weights_only=True 기본값과 충돌함
# (우리가 직접 만든 체크포인트라 신뢰 가능 -> False로 패치)
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

sys.path.insert(0, str(Path(__file__).parent / "parseq"))
from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint

YOLO_WEIGHTS = "/home/user/Github/runs/detect/runs/imprint/yolo26n_p2_detect-3/weights/best.pt"
PARSEQ_CKPT = "outputs/parseq-tiny/2026-06-20_15-17-55/checkpoints/epoch=48-step=5292-val_accuracy=48.6842-val_NED=66.6859.ckpt"
LEXICON_PATH = "drug_imprints.txt"
IMPRINT_CLS = 0


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


def lexicon_correct(pred, lexicon, max_dist=2):
    if not lexicon or pred in lexicon:
        return pred
    best, bd = pred, max_dist + 1
    for cand in lexicon:
        d = levenshtein(pred, cand)
        if d < bd:
            best, bd = cand, d
    return best if bd <= max_dist else pred


def pad_box(b, pad, w, h):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    return (max(0, x1 - bw * pad), max(0, y1 - bh * pad),
            min(w, x2 + bw * pad), min(h, y2 + bh * pad))


class ImprintReader:
    def __init__(self, yolo_weights=YOLO_WEIGHTS, parseq_ckpt=PARSEQ_CKPT,
                 lexicon_path=LEXICON_PATH, conf=0.1, pad=0.15, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf = conf
        self.pad = pad
        self.yolo = YOLO(yolo_weights)
        self.parseq = load_from_checkpoint(parseq_ckpt).eval().to(self.device)
        self.transform = SceneTextDataModule.get_transform(self.parseq.hparams.img_size)
        self.lexicon = []
        if lexicon_path and Path(lexicon_path).exists():
            self.lexicon = [l.strip() for l in open(lexicon_path, encoding="utf-8") if l.strip()]

    def read(self, image_path):
        """returns list of {bbox, det_conf, text_raw, text_corrected}"""
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        res = self.yolo.predict(str(image_path), imgsz=960, conf=self.conf, verbose=False)[0]
        results = []
        if res.boxes is None:
            return results
        for b, c, dconf in zip(res.boxes.xyxy.cpu().numpy(),
                                res.boxes.cls.cpu().numpy(),
                                res.boxes.conf.cpu().numpy()):
            if int(c) != IMPRINT_CLS:
                continue
            x1, y1, x2, y2 = map(int, pad_box(tuple(b), self.pad, w, h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img.crop((x1, y1, x2, y2))
            x = self.transform(crop).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                logits = self.parseq(x).softmax(-1)
            pred, _ = self.parseq.tokenizer.decode(logits)
            text_raw = pred[0]
            text_corrected = lexicon_correct(text_raw, self.lexicon)
            results.append({
                "bbox": (x1, y1, x2, y2),
                "det_conf": float(dconf),
                "text_raw": text_raw,
                "text_corrected": text_corrected,
            })
        return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="단일 이미지 경로")
    ap.add_argument("--images_dir", help="이미지 폴더(일괄 처리)")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pad", type=float, default=0.15)
    args = ap.parse_args()

    if not args.image and not args.images_dir:
        ap.error("--image 또는 --images_dir 중 하나는 필요합니다")

    reader = ImprintReader(conf=args.conf, pad=args.pad)

    paths = [Path(args.image)] if args.image else sorted(
        p for p in Path(args.images_dir).rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png"))

    for p in paths:
        results = reader.read(p)
        texts = [r["text_corrected"] for r in results]
        print(f"{p.name}: {texts if texts else '(각인 없음/탐지 실패)'}")
        for r in results:
            print(f"    bbox={r['bbox']} det_conf={r['det_conf']:.2f} "
                  f"raw='{r['text_raw']}' -> corrected='{r['text_corrected']}'")


if __name__ == "__main__":
    main()
