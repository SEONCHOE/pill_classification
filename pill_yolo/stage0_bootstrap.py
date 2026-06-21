"""
stage0_bootstrap.py
===================
Stage 0: Gemini 2.5 Flash로 알약 이미지에서 각인 bbox를 자동 생성 → YOLO 라벨 포맷으로 저장.

배경:
  RxIMAGE / AI Hub 모두 알약 전체 박스 라벨만 있고 "각인 영역 bbox"는 없음.
  Gemini에게 0~1000 정규화 좌표로 bbox를 요청한 뒤 YOLO 포맷(0~1)으로 변환.
  이후 10~20% 수동 QA(stage0_visualize.py)로 품질 확인.

출력 디렉터리 구조:
  datasets/imprint/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt    ← 이 스크립트가 생성
    labels/val/*.txt

실행:
  python stage0_bootstrap.py --img_dir <알약이미지폴더> --split train
  python stage0_bootstrap.py --img_dir <알약이미지폴더> --split val --sample 50
"""

import os
import json
import re
import argparse
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv(Path(__file__).parent.parent / "pill_multiagent" / ".env")

# ── 설정 ──────────────────────────────────────────────────────────────────────

GEMINI_MODEL  = os.getenv("BOOTSTRAP_MODEL", "gemini-2.5-flash")
DATASET_ROOT  = Path(__file__).parent / "datasets" / "imprint"
RETRY_LIMIT   = 3
RETRY_DELAY   = 5   # 초

# YOLO 클래스 인덱스 (imprint_det.yaml과 동일하게)
CLS_IMPRINT    = 0
CLS_SCORE_LINE = 1

# ── Gemini 프롬프트 ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """당신은 의약품 각인 영역 탐지 전문가입니다.
알약 이미지에서 각인(imprint) 영역과 분할선(score line)을 찾아
bounding box 좌표를 정확히 반환합니다."""

USER_PROMPT = """이 알약 이미지를 분석하여 각인 영역의 bounding box를 반환하세요.

규칙:
- 좌표계: 이미지 좌상단 (0,0), 우하단 (1000,1000) 기준 정규화 좌표
- 각인(imprint): 표면에 새겨진 문자/숫자/기호/마크 영역 (양각·음각 포함)
- 분할선(score_line): 알약 가운데를 가로지르는 홈/선
- 각인이 없으면 boxes를 빈 배열로 반환

반드시 아래 JSON 형식으로만 답하세요:
{
  "boxes": [
    {
      "class": "imprint" 또는 "score_line",
      "x1": 0~1000,
      "y1": 0~1000,
      "x2": 0~1000,
      "y2": 0~1000,
      "confidence": 0.0~1.0,
      "text_hint": "보이는 텍스트 (선택)"
    }
  ],
  "notes": "특이사항 (음각여부 등)"
}"""


# ── 좌표 변환 ─────────────────────────────────────────────────────────────────

def xyxy1000_to_yolo(x1, y1, x2, y2):
    """0~1000 좌표 → YOLO xc,yc,w,h (0~1 정규화)"""
    xc = (x1 + x2) / 2 / 1000
    yc = (y1 + y2) / 2 / 1000
    w  = (x2 - x1) / 1000
    h  = (y2 - y1) / 1000
    # 0~1 범위 클리핑
    xc = max(0.0, min(1.0, xc))
    yc = max(0.0, min(1.0, yc))
    w  = max(0.001, min(1.0, w))
    h  = max(0.001, min(1.0, h))
    return xc, yc, w, h


def parse_response(raw: str) -> list[dict]:
    """Gemini 응답에서 boxes 파싱 (코드펜스 처리 포함)"""
    text = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        return data.get("boxes", [])
    except json.JSONDecodeError:
        return []


# ── Gemini 호출 ───────────────────────────────────────────────────────────────

def get_bbox_from_gemini(img_bytes: bytes, client: genai.Client) -> list[dict]:
    """단일 이미지에서 bbox 목록 반환"""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    types.Part.from_text(text=USER_PROMPT),
                ],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    max_output_tokens=2048,
                    temperature=0.1,   # 낮은 temperature → 좌표 안정성
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
                raise RuntimeError("MAX_TOKENS로 잘림 — max_output_tokens 조정 필요")
            return parse_response(response.text.strip())
        except Exception as e:
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY * attempt)
            else:
                raise
    return []


# ── 라벨 저장 ─────────────────────────────────────────────────────────────────

def boxes_to_yolo_lines(boxes: list[dict]) -> list[str]:
    lines = []
    for b in boxes:
        cls_name = b.get("class", "imprint")
        cls_idx  = CLS_IMPRINT if cls_name == "imprint" else CLS_SCORE_LINE

        try:
            xc, yc, w, h = xyxy1000_to_yolo(
                float(b["x1"]), float(b["y1"]),
                float(b["x2"]), float(b["y2"]),
            )
        except (KeyError, ValueError):
            continue

        lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


# ── 메인 ──────────────────────────────────────────────────────────────────────

def run(img_dir: str, split: str, sample: int | None, dry_run: bool):
    img_dir = Path(img_dir)
    img_files = sorted(
        [p for p in img_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if sample:
        img_files = img_files[:sample]

    out_img_dir = DATASET_ROOT / "images" / split
    out_lbl_dir = DATASET_ROOT / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"대상 이미지: {len(img_files)}장  →  {out_lbl_dir}")
    if dry_run:
        print("[DRY RUN] Gemini 호출 없이 구조만 확인합니다.")
        return

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    skipped, success, no_box = 0, 0, 0
    log_rows = []

    for img_path in tqdm(img_files, desc="라벨 생성"):
        lbl_path = out_lbl_dir / (img_path.stem + ".txt")

        # 이미 라벨 있으면 건너뜀 (재시작 대응)
        if lbl_path.exists():
            skipped += 1
            continue

        img_bytes = img_path.read_bytes()
        try:
            boxes = get_bbox_from_gemini(img_bytes, client)
        except Exception as e:
            tqdm.write(f"[SKIP] {img_path.name}: {e}")
            skipped += 1
            continue

        lines = boxes_to_yolo_lines(boxes)

        # 라벨 파일 저장 (각인 없어도 빈 파일 생성 → YOLO background 이미지로 활용)
        lbl_path.write_text("\n".join(lines), encoding="utf-8")

        # 이미지도 out_img_dir에 심볼릭 링크 또는 복사
        dst_img = out_img_dir / img_path.name
        if not dst_img.exists():
            import shutil
            shutil.copy2(img_path, dst_img)

        if lines:
            success += 1
        else:
            no_box += 1

        log_rows.append({
            "image": img_path.name,
            "n_boxes": len(lines),
            "label_file": str(lbl_path),
        })

    # 결과 요약
    print(f"\n완료: 성공={success}, 박스없음={no_box}, 건너뜀={skipped}")
    print(f"라벨 저장: {out_lbl_dir}")

    # CSV 로그 저장
    if log_rows:
        import csv
        log_path = DATASET_ROOT / f"bootstrap_log_{split}.csv"
        with open(log_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "n_boxes", "label_file"])
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"로그: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Gemini bbox 라벨 부트스트랩")
    parser.add_argument("--img_dir", required=True, help="원본 알약 이미지 폴더")
    parser.add_argument("--split",   default="train", choices=["train", "val", "test"])
    parser.add_argument("--sample",  type=int, default=None, help="테스트용 샘플 수 제한")
    parser.add_argument("--dry_run", action="store_true", help="Gemini 호출 없이 구조 확인만")
    args = parser.parse_args()

    run(args.img_dir, args.split, args.sample, args.dry_run)
