"""
make_crops_and_ocr.py
==============================================================
우리 YOLO 모델(conf=0.1)의 예측 박스로 imprint crop을 만들고
EasyOCR로 읽어본다 (정성적 엔드투엔드 확인, GT 텍스트 없이).
==============================================================
"""
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

IMPRINT_CLS = 0


def pad_box(b, pad, w, h):
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    return (max(0, x1 - bw * pad), max(0, y1 - bh * pad),
            min(w, x2 + bw * pad), min(h, y2 + bh * pad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="ocr_check")
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--pad", type=float, default=0.15)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)

    img_paths = sorted([p for p in Path(args.images).rglob("*")
                         if p.suffix.lower() in (".jpg", ".jpeg", ".png")])[:args.limit]

    results_txt = []
    for ip in img_paths:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = model.predict(str(ip), imgsz=960, conf=args.conf, verbose=False)[0]
        if res.boxes is None:
            continue
        crop_idx = 0
        for b, c, conf in zip(res.boxes.xyxy.cpu().numpy(),
                               res.boxes.cls.cpu().numpy(),
                               res.boxes.conf.cpu().numpy()):
            if int(c) != IMPRINT_CLS:
                continue
            x1, y1, x2, y2 = map(int, pad_box(tuple(b), args.pad, w, h))
            if x2 <= x1 or y2 <= y1:
                continue
            crop = img[y1:y2, x1:x2]
            cpath = out_dir / f"{ip.stem}_{crop_idx}.png"
            cv2.imwrite(str(cpath), crop)
            ocr_out = reader.readtext(crop)
            text = " ".join([t[1] for t in ocr_out])
            results_txt.append(f"{ip.name}_{crop_idx}\tdet_conf={conf:.2f}\tocr='{text}'")
            crop_idx += 1

    (out_dir / "_ocr_results.txt").write_text("\n".join(results_txt), encoding="utf-8")
    print("\n".join(results_txt))
    print(f"\n{len(results_txt)} crops -> {out_dir}")


if __name__ == "__main__":
    main()
