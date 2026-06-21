"""
train_yolo26_imprint.py
==============================================================
알약 각인 영역 탐지 — YOLO26 fine-tuning 스크립트

설치:
    pip install -U ultralytics            # YOLO26 포함 (2026.01~)

실행:
    python train_yolo26_imprint.py        # 기본: P2 detection
    USE_P2=0 python train_yolo26_imprint.py   # 표준 head로
    TASK=obb  python train_yolo26_imprint.py  # OBB 변형(데이터 포맷 다름, 아래 주석 참고)

핵심 설계:
  - 베이스: YOLO26 (NMS-free, DFL 제거, STAL=작은객체 라벨할당)
  - P2 head: 작은 각인 영역 탐지 강화
  - augmentation: 임의 회전/밝기/부분가림 = 사용자 촬영 robust
==============================================================
"""

import os
from ultralytics import YOLO

# ----------------------------------------------------------------
# 0. 스위치 (환경변수로 토글)
# ----------------------------------------------------------------
USE_P2 = os.getenv("USE_P2", "1") == "1"     # 1=P2 small-object head 사용
TASK   = os.getenv("TASK", "detect")          # "detect" 또는 "obb"
SCALE  = os.getenv("SCALE", "n")              # n/s/m/l/x (모바일이면 n 또는 s)

DATA   = "imprint_det.yaml"                    # OBB면 imprint_obb.yaml로 교체
PROJECT = "runs/imprint"
NAME    = f"yolo26{SCALE}_{'p2' if USE_P2 else 'std'}_{TASK}"

# ----------------------------------------------------------------
# 1. 모델 로드
# ----------------------------------------------------------------
# 표준 detection:
#   - P2: 사전학습 .pt 가중치가 없으므로 YAML로 구조를 만들고
#         COCO 가중치를 .load()로 전이(호환 레이어만 적용)
#   - 비P2: 사전학습 .pt 직접 사용
if TASK == "obb":
    # OBB는 회전 박스. 데이터 라벨이 8좌표(폴리곤) 정규화 포맷이어야 함.
    # 각인의 회전각까지 추정 → 이후 STR 정렬(rectification)에 유리.
    base = f"yolo26{SCALE}-obb.pt"
    model = YOLO(base)
else:
    if USE_P2:
        cfg = f"yolo26{SCALE}-p2.yaml"
        model = YOLO(cfg).load(f"yolo26{SCALE}.pt")   # 구조=P2, 가중치=COCO 전이
    else:
        model = YOLO(f"yolo26{SCALE}.pt")

# ----------------------------------------------------------------
# 2. 학습 하이퍼파라미터
# ----------------------------------------------------------------
train_args = dict(
    data=DATA,
    task=TASK,

    # ---- 기본 학습 ----
    epochs=200,
    patience=30,            # 30 epoch 개선 없으면 조기 종료
    batch=4,               # GPU 2GB(GTX 1050) → autobatch(-1)는 OOM, 4로 고정 확인됨
    imgsz=960,             # 각인=작은 객체 → 640보다 키움 (여유되면 1280)
    device=0,              # GPU id. CPU면 "cpu"
    workers=2,
    seed=42,
    deterministic=True,
    cache=False,           # 데이터 작고 빠른 디스크면 "ram"으로 가속

    # ---- 옵티마이저 / LR ----
    optimizer="auto",      # YOLO26 권장(MuSGD 자동 적용)
    cos_lr=True,           # 코사인 LR 스케줄
    lr0=0.01,
    lrf=0.01,
    weight_decay=0.0005,
    warmup_epochs=3.0,

    # ---- 전이학습 안정화 ----
    freeze=10,             # 초기 backbone 일부 동결(소규모 데이터 과적합 완화).
                           # 데이터 충분하면 0으로.

    # ============================================================
    # ---- Augmentation : 각인 탐지 특화 ----
    # ============================================================
    # 기하: 사용자가 임의 각도/거리에서 촬영
    degrees=180.0,         # 임의 회전 (각인 방향 불특정)
    flipud=0.5,            # 상하 뒤집기 (방향 무의미)
    fliplr=0.5,            # 좌우 뒤집기
    translate=0.1,
    scale=0.5,             # 거리 변화 대응
    shear=2.0,
    perspective=0.0005,    # 약한 원근(카메라 각도)

    # 색/조명: 음각 각인=저대비 → 밝기 변동에 robust해야
    hsv_h=0.015,           # 색상 변화는 작게(알약 색 보존)
    hsv_s=0.5,
    hsv_v=0.5,             # 밝기 변동 강화 (그림자/조명)

    # 가림/혼합
    erasing=0.4,           # 부분 가림(각인 일부 손상/지워짐 시뮬레이션)
    mosaic=0.5,            # 단일 중앙객체라 1.0은 과함 → 0.5
    close_mosaic=10,       # 마지막 10 epoch mosaic 끔(미세조정 안정화)
    mixup=0.0,
    copy_paste=0.0,

    # ---- 로깅/저장 ----
    project=PROJECT,
    name=NAME,
    plots=True,
    save=True,
    val=True,
)

# 단일 클래스(각인만)로 갈 경우:
# train_args["single_cls"] = True

# ----------------------------------------------------------------
# 3. 학습
# ----------------------------------------------------------------
results = model.train(**train_args)

# ----------------------------------------------------------------
# 4. 검증 (mAP50, mAP50-95, P, R)
# ----------------------------------------------------------------
metrics = model.val(data=DATA, imgsz=960, split="val")
print("mAP50    :", metrics.box.map50)
print("mAP50-95 :", metrics.box.map)
print("precision:", metrics.box.mp)
print("recall   :", metrics.box.mr)

# ----------------------------------------------------------------
# 5. 모바일/엣지 export (NMS-free라 export가 깔끔함)
# ----------------------------------------------------------------
# 온디바이스 대상에 맞춰 선택:
model.export(format="tflite", imgsz=960, int8=True)   # 안드로이드/엣지(양자화)
# model.export(format="coreml", imgsz=960)            # iOS
# model.export(format="onnx",   imgsz=960)            # 범용
# model.export(format="openvino", imgsz=960)          # Intel CPU/NPU

print("Done. weights:", os.path.join(PROJECT, NAME, "weights", "best.pt"))
