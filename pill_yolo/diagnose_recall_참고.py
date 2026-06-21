"""
diagnose_recall.py
================================================================
각인 탐지 recall 진단 (학습 더 돌리기 전에 실행)

세 가지를 한 번에:
  1) per-class 지표  — imprint vs score_line 분리 (어느 클래스가 끌어내리나)
  2) confidence sweep — conf 낮출 때 recall이 살아나는가 (= 운영 임계값 문제인가)
  3) 이미지 단위 커버리지 recall — STR로 넘길 만큼 각인을 덮었나
     (COCO instance mAP보다 실제 목적에 맞는 지표)

실행:
    python diagnose_recall.py --weights runs/imprint/.../weights/best.pt \
                              --data imprint_det.yaml --imgsz 960

요구: pip install ultralytics pyyaml numpy
================================================================
"""
import argparse
from pathlib import Path
import numpy as np
import yaml
from ultralytics import YOLO

IMPRINT_CLS = 0   # imprint 클래스 id
SCORE_CLS = 1     # score_line 클래스 id


# ----------------------------------------------------------------
# GT 라벨 로드 (YOLO 포맷 → 픽셀 xyxy)
# ----------------------------------------------------------------
def load_gt(label_path, w, h):
    """반환: {cls: [ (x1,y1,x2,y2), ... ]}"""
    boxes = {}
    if not Path(label_path).exists():
        return boxes
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        x1 = (xc - bw / 2) * w
        y1 = (yc - bh / 2) * h
        x2 = (xc + bw / 2) * w
        y2 = (yc + bh / 2) * h
        boxes.setdefault(cls, []).append((x1, y1, x2, y2))
    return boxes


def pad_box(b, pad, w, h):
    """박스를 pad 비율만큼 확장(실제 crop 패딩 시뮬레이션)."""
    x1, y1, x2, y2 = b
    bw, bh = x2 - x1, y2 - y1
    return (max(0, x1 - bw * pad), max(0, y1 - bh * pad),
            min(w, x2 + bw * pad), min(h, y2 + bh * pad))


def coverage(gt, preds):
    """GT 박스가 예측 박스들의 합집합으로 얼마나 덮였나 (0~1)."""
    x1, y1, x2, y2 = map(int, gt)
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    mask = np.zeros((bh, bw), dtype=bool)
    for p in preds:
        ix1, iy1 = max(x1, p[0]), max(y1, p[1])
        ix2, iy2 = min(x2, p[2]), min(y2, p[3])
        if ix2 > ix1 and iy2 > iy1:
            mask[int(iy1 - y1):int(iy2 - y1), int(ix1 - x1):int(ix2 - x1)] = True
    return mask.mean()


# ----------------------------------------------------------------
# 1) per-class 지표
# ----------------------------------------------------------------
def per_class_metrics(model, data, imgsz, split):
    print("\n" + "=" * 60)
    print("[1] per-class 지표 (best-F1 임계값 기준)")
    print("=" * 60)
    m = model.val(data=data, imgsz=imgsz, split=split, verbose=False)
    names = model.names
    idxs = m.box.ap_class_index
    print(f"{'class':<12}{'precision':>11}{'recall':>9}{'mAP50':>9}{'mAP50-95':>10}")
    for i, c in enumerate(idxs):
        print(f"{names[c]:<12}{m.box.p[i]:>11.3f}{m.box.r[i]:>9.3f}"
              f"{m.box.ap50[i]:>9.3f}{m.box.ap[i]:>10.3f}")
    print(f"{'(all)':<12}{m.box.mp:>11.3f}{m.box.mr:>9.3f}"
          f"{m.box.map50:>9.3f}{m.box.map:>10.3f}")
    print("\n해석: imprint와 score_line recall 격차가 크면 score_line이 전체를")
    print("      끌어내리는 것 → single_cls(imprint only) 재학습 검토.")
    return m


# ----------------------------------------------------------------
# 2) confidence sweep
# ----------------------------------------------------------------
def conf_sweep(model, data, imgsz, split, confs):
    print("\n" + "=" * 60)
    print("[2] confidence sweep — conf 낮출 때 recall 회복되나")
    print("=" * 60)
    print(f"{'conf':>8}{'precision':>11}{'recall':>9}{'mAP50':>9}")
    best = None
    for c in confs:
        m = model.val(data=data, imgsz=imgsz, split=split, conf=c, verbose=False)
        print(f"{c:>8.3f}{m.box.mp:>11.3f}{m.box.mr:>9.3f}{m.box.map50:>9.3f}")
        if best is None or m.box.mr > best[1]:
            best = (c, m.box.mr)
    print(f"\n최대 recall ≈ {best[1]:.3f} (conf={best[0]})")
    print("해석: 낮은 conf에서 recall이 크게 오르면 = 모델은 각인을 '보고는 있음'.")
    print("      STR이 FP를 걸러주니 낮은 conf로 운영하면 사실상 해결.")


# ----------------------------------------------------------------
# 3) 이미지 단위 커버리지 recall (STR 관점)
# ----------------------------------------------------------------
def image_level_coverage(model, data, imgsz, conf, pad, cov_thr):
    print("\n" + "=" * 60)
    print(f"[3] 이미지 단위 커버리지 recall (conf={conf}, pad={pad}, "
          f"cov_thr={cov_thr})")
    print("=" * 60)
    with open(data) as f:
        d = yaml.safe_load(f)
    root = Path(d["path"])
    val_dir = root / d["val"]
    img_paths = [p for p in val_dir.rglob("*")
                 if p.suffix.lower() in (".jpg", ".jpeg", ".png")]

    total_gt, recovered = 0, 0
    img_with_imprint, img_ok = 0, 0

    for ip in img_paths:
        # GT
        lp = Path(str(ip).replace("/images/", "/labels/"))
        lp = lp.with_suffix(".txt")
        import cv2
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = load_gt(lp, w, h)
        gt_imprints = gt.get(IMPRINT_CLS, [])
        if not gt_imprints:
            continue  # 각인 없는 면은 recall 대상 아님
        img_with_imprint += 1

        # 예측 (imprint 클래스만, 낮은 conf)
        res = model.predict(str(ip), imgsz=imgsz, conf=conf, verbose=False)[0]
        preds = []
        if res.boxes is not None:
            for b, c in zip(res.boxes.xyxy.cpu().numpy(),
                            res.boxes.cls.cpu().numpy()):
                if int(c) == IMPRINT_CLS:
                    preds.append(pad_box(tuple(b), pad, w, h))

        all_cov = True
        for g in gt_imprints:
            total_gt += 1
            cov = coverage(g, preds)
            if cov >= cov_thr:
                recovered += 1
            else:
                all_cov = False
        if all_cov:
            img_ok += 1

    print(f"GT imprint 박스 단위 recall : {recovered}/{total_gt} "
          f"= {recovered/max(1,total_gt):.3f}")
    print(f"이미지 단위(각인 전부 회수)  : {img_ok}/{img_with_imprint} "
          f"= {img_ok/max(1,img_with_imprint):.3f}")
    print("\n해석: 이 값이 COCO recall(0.30)보다 높으면, 실제 STR 용도로는")
    print("      이미 쓸 만하다는 의미. 박스가 GT를 '충분히' 덮으면 OK로 침.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default="imprint_det.yaml")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="커버리지 측정 시 운영 conf")
    ap.add_argument("--pad", type=float, default=0.15,
                    help="예측 박스 패딩 비율(crop 패딩 시뮬레이션)")
    ap.add_argument("--cov_thr", type=float, default=0.70,
                    help="GT를 이만큼 덮으면 '회수'로 간주")
    args = ap.parse_args()

    model = YOLO(args.weights)

    per_class_metrics(model, args.data, args.imgsz, args.split)
    conf_sweep(model, args.data, args.imgsz, args.split,
               confs=[0.001, 0.01, 0.05, 0.10, 0.25, 0.50])
    image_level_coverage(model, args.data, args.imgsz,
                         args.conf, args.pad, args.cov_thr)

    print("\n" + "=" * 60)
    print("다음 판단 가이드")
    print("=" * 60)
    print("- [2]에서 낮은 conf recall이 높다      → 낮은 conf로 운영(즉시 해결)")
    print("- [1]에서 imprint recall만 보면 양호    → single_cls 재학습")
    print("- [3] 커버리지 recall이 높다           → 사실상 STR에 쓸 만함")
    print("- 셋 다 낮다                          → 구조적 레버(알약 crop, "
          "batch↑/Colab, 데이터 증량)로")


if __name__ == "__main__":
    main()
