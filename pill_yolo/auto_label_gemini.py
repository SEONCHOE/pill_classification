"""
auto_label_gemini.py
==============================================================
MLLM(Gemini 2.5 Flash) few-shot 자동 라벨링
  - 입력: 알약 이미지 폴더
  - 한 번의 호출로 각인 영역 bbox + 각인 텍스트 + 분할선 bbox 동시 출력
  - 산출물:
      (A) YOLO detection 라벨  : labels/*.txt   (class xc yc w h, 0~1)
      (B) STR 학습용 manifest  : str_manifest.csv (crop_path,text)
      (C) 각인 crop 이미지      : crops/*.png

설치:
    pip install -U google-genai pillow

키:
    export GEMINI_API_KEY=...        # 또는 GOOGLE_API_KEY

실행:
    python auto_label_gemini.py --images ./datasets/raw --out ./datasets/imprint

핵심 주의(코드 밖):
  - 사람 골드셋으로 이 라벨 품질을 먼저 검증할 것(IoU/문자정확도).
  - 저신뢰/기하이상 샘플은 review_queue.csv로 빠져 사람이 검수.
==============================================================
"""
import os, csv, json, argparse, time
from pathlib import Path
from PIL import Image
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / "pill_multiagent" / ".env")

# ----------------------------------------------------------------
# 출력 스키마 (structured output 강제)
# ----------------------------------------------------------------
class Det(BaseModel):
    label: str          # "imprint" | "score_line"
    text: str           # 각인 문자열 (score_line이면 "")
    box_2d: list[int]   # [ymin, xmin, ymax, xmax], 0~1000 정규화 (Gemini 규약)
    confidence: float   # 0~1

class Result(BaseModel):
    detections: list[Det]

CLASS_MAP = {"imprint": 0, "score_line": 1}   # imprint_det.yaml과 일치

PROMPT = """You are labeling pharmaceutical pill images for a detection dataset.
Detect on the pill:
1) "imprint": each region containing engraved or printed characters/marks. Provide the exact characters in `text` (uppercase letters, digits, symbols as seen). If engraving is debossed/low-contrast, still localize it tightly.
2) "score_line": the dividing score line groove, if present. Set text to "".

Rules:
- box_2d format is [ymin, xmin, ymax, xmax], normalized 0-1000.
- Make boxes as TIGHT as possible around the marks only (not the whole pill).
- If you are unsure, still return the box but lower the confidence.
- Do NOT include the pill outline as an imprint.
Return JSON only."""

# (선택) few-shot 예시: (이미지경로, Result JSON dict) 튜플 목록
# 예시를 채우면 정확도가 오릅니다. 비워두면 zero-shot.
FEWSHOT = [
    # ("examples/ex1.png", {"detections":[{"label":"imprint","text":"AB123","box_2d":[420,380,560,640],"confidence":0.95}]}),
]

# ----------------------------------------------------------------
def build_contents(img_bytes, mime):
    """few-shot 예시 + 현재 이미지로 contents 구성"""
    contents = [PROMPT]
    for ex_path, ex_json in FEWSHOT:
        with open(ex_path, "rb") as f:
            contents.append(types.Part.from_bytes(data=f.read(), mime_type="image/png"))
        contents.append("Expected JSON:\n" + json.dumps(ex_json, ensure_ascii=False))
    contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
    return contents


def to_yolo(box_2d):
    """[ymin,xmin,ymax,xmax] 0~1000  ->  (xc,yc,w,h) 0~1"""
    ymin, xmin, ymax, xmax = box_2d
    xc = (xmin + xmax) / 2.0 / 1000.0
    yc = (ymin + ymax) / 2.0 / 1000.0
    w  = (xmax - xmin) / 1000.0
    h  = (ymax - ymin) / 1000.0
    return xc, yc, w, h


def qa_ok(label, text, xc, yc, w, h, conf, conf_thr):
    """기하/신뢰도 자동 QA. 통과 못하면 review로."""
    if conf < conf_thr:                      return False, "low_conf"
    if not (0 <= xc <= 1 and 0 <= yc <= 1):  return False, "out_of_bounds"
    if not (0.01 < w < 0.95 and 0.01 < h < 0.95): return False, "bad_size"
    ar = w / max(h, 1e-6)
    if not (0.05 < ar < 20):                 return False, "bad_aspect"
    if label == "imprint" and len(text.strip()) == 0: return False, "empty_text"
    return True, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--pad", type=float, default=0.08, help="STR crop 여유 비율")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "crops").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=Result,
        temperature=0.0,
    )

    manifest_path = out / "str_manifest.csv"
    review_path = out / "review_queue.csv"
    if not manifest_path.exists():
        manifest_path.write_text("crop_path,text\n", encoding="utf-8")
    if not review_path.exists():
        review_path.write_text("image,label,reason\n", encoding="utf-8")

    done_stems = {p.stem for p in (out / "images").glob("*.png")}
    str_rows, review_rows = [], []
    img_paths = [p for p in Path(args.images).rglob("*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
                 and p.stem not in done_stems]
    print(f"{len(img_paths)} images (resume: {len(done_stems)} already done, skipped)")

    for i, p in enumerate(img_paths):
        img_str_rows, img_review_rows = [], []
        try:
            raw = p.read_bytes()
            mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
            resp = client.models.generate_content(
                model=args.model, contents=build_contents(raw, mime), config=cfg)
            result: Result = resp.parsed
        except Exception as e:
            print(f"[ERR] {p.name}: {e}")
            img_review_rows.append([str(p), "api_error", str(e)])
            _flush(manifest_path, review_path, img_str_rows, img_review_rows)
            time.sleep(1.0)
            continue

        img = Image.open(p).convert("RGB")
        W, H = img.size
        label_lines, crop_idx = [], 0

        for d in (result.detections if result else []):
            if d.label not in CLASS_MAP:
                continue
            try:
                xc, yc, w, h = to_yolo(d.box_2d)
            except Exception as e:
                img_review_rows.append([str(p), d.label, f"bad_box_2d:{e}"])
                continue
            ok, reason = qa_ok(d.label, d.text, xc, yc, w, h, d.confidence, args.conf)
            if not ok:
                img_review_rows.append([str(p), d.label, reason])
                continue

            label_lines.append(f"{CLASS_MAP[d.label]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

            # imprint면 STR 학습용 crop 저장 (약간 패딩)
            if d.label == "imprint":
                pw, ph = w * args.pad, h * args.pad
                x1 = int(max(0, (xc - w/2 - pw)) * W); x2 = int(min(1, (xc + w/2 + pw)) * W)
                y1 = int(max(0, (yc - h/2 - ph)) * H); y2 = int(min(1, (yc + h/2 + ph)) * H)
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
                    cpath = out / "crops" / f"{p.stem}_{crop_idx}.png"
                    crop.save(cpath)
                    img_str_rows.append([str(cpath), d.text.strip()])
                    crop_idx += 1

        # YOLO 라벨 + 이미지 사본 저장 (라벨이 하나라도 있을 때)
        if label_lines:
            (out / "labels" / f"{p.stem}.txt").write_text("\n".join(label_lines))
            img.save(out / "images" / f"{p.stem}.png")

        str_rows.extend(img_str_rows)
        review_rows.extend(img_review_rows)
        _flush(manifest_path, review_path, img_str_rows, img_review_rows)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(img_paths)}")

    print(f"STR samples (이번 실행) : {len(str_rows)}")
    print(f"review queue (이번 실행): {len(review_rows)}  (사람 검수 필요)")
    print(f"-> {out}/images, /labels, /crops, str_manifest.csv, review_queue.csv")


def _flush(manifest_path, review_path, str_rows, review_rows):
    if str_rows:
        with open(manifest_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(str_rows)
    if review_rows:
        with open(review_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(review_rows)


if __name__ == "__main__":
    main()
