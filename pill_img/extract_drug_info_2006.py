"""
extract_drug_info_2006.py
==============================================================
drug_info_2006.pdf (한국 의약품 식별정보 참고서, 896페이지)에서
약물명 + 알약 사진을 Gemini Vision으로 추출.

배경: 각 페이지가 텍스트 레이어 없는 통짜 스캔 이미지 1장이라
      일반 PDF 파싱(임베드 이미지 분리)으로는 약물명/사진을 못 나눔.
      -> Gemini Vision으로 페이지 스크린샷을 통째로 보여주고
         "약물 항목 + 사진 bbox" 구조화 출력을 받음(auto_label_gemini.py와 같은 패턴).

실행(파일럿, 일부 페이지만):
    python extract_drug_info_2006.py --pages 50,200,500,895 --out pilot_out
==============================================================
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

import fitz
from PIL import Image
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv(Path(__file__).parent.parent / "pill_multiagent" / ".env")

PDF_PATH = Path(__file__).parent / "drug_info_2006.pdf"
RENDER_DPI = 200


class PillPhoto(BaseModel):
    box_2d: list[int]   # [ymin, xmin, ymax, xmax], 0~1000 정규화
    color: str          # 보이는 색상(예: "흰색", "주황/흰색")
    shape: str          # 보이는 모양(예: "원형", "장방형 캡슐")


class DrugEntry(BaseModel):
    name: str            # 약물명(보이는 대로, 영문/한글)
    dosage_text: str     # 함량/제형 등 부가 텍스트(없으면 "")
    photos: list[PillPhoto]


class PageResult(BaseModel):
    page_type: str       # "photo_card" | "code_index_table" | "other"
    entries: list[DrugEntry]


PROMPT = """This image is a page from a Korean pharmaceutical drug identification reference book.
There are two distinct page layouts in this book — first decide which one this page is:

- "photo_card": each drug has a purple name header, then a small flat colored square reference
  icon (a tiny solid-color badge with a short code, NOT a real photo) next to product/price text,
  and BELOW that a separate, LARGER, photorealistic photo of the actual pill/capsule (with visible
  shading, highlights, and printed imprint text on its surface).
- "code_index_table": a dense table with columns like 표시/분할선/색깔/모양/제형/제품명/페이지,
  sorted alphabetically by imprint code, with tiny icon-sized thumbnails per row.
- "other": cover page, table of contents, blank page, etc.

If page_type == "photo_card": for each drug entry, extract:
1) name: the drug name exactly as shown (verbatim).
2) dosage_text: nearby dosage/formulation text (empty string if none).
3) photos: bounding boxes ONLY for the large photorealistic pill/capsule photos —
   DO NOT box the small flat-colored square reference icon, DO NOT box any text/price/barcode area.
   - box_2d: [ymin, xmin, ymax, xmax], normalized 0-1000, TIGHT around the real photo only.
   - color: color(s) visible in the photo.
   - shape: shape visible in the photo (round, oblong, capsule, etc).

If page_type == "code_index_table" or "other": return entries as an empty array
(the small table thumbnails are too low-quality to use as reference photos; we only want the
structured text in that case, which this schema doesn't capture — just flag the page_type).

Return JSON only."""


def render_page(page_num: int, out_dir: Path) -> Path:
    doc = fitz.open(PDF_PATH)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=RENDER_DPI)
    out_path = out_dir / f"page{page_num:04d}.png"
    pix.save(out_path)
    doc.close()
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True, help="쉼표구분 0-based 페이지 번호, 예: 50,200,500,895")
    ap.add_argument("--out", default="pilot_out")
    ap.add_argument("--model", default="gemini-2.5-flash")
    args = ap.parse_args()

    page_nums = [int(p) for p in args.pages.split(",")]
    out_dir = Path(args.out)
    (out_dir / "page_renders").mkdir(parents=True, exist_ok=True)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    cfg = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=PageResult,
        temperature=0.0,
    )

    # incremental CSV + resume: 이미 처리한 페이지는 done_pages.txt로 추적
    csv_path = out_dir / "extracted.csv"
    done_path = out_dir / "done_pages.txt"
    if not csv_path.exists():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["page", "name", "dosage_text", "color", "shape", "crop_path"])
    done_pages = set()
    if done_path.exists():
        done_pages = {int(x) for x in done_path.read_text().split() if x.strip().isdigit()}

    total = 0
    for pn in page_nums:
        if pn in done_pages:
            continue
        img_path = render_page(pn, out_dir / "page_renders")
        print(f"page {pn}: rendered -> {img_path}", flush=True)
        try:
            raw = img_path.read_bytes()
            resp = client.models.generate_content(
                model=args.model,
                contents=[PROMPT, types.Part.from_bytes(data=raw, mime_type="image/png")],
                config=cfg,
            )
            result: PageResult = resp.parsed
        except Exception as e:
            print(f"  [ERR] page {pn}: {e}", flush=True)
            time.sleep(1.0)
            continue

        img = Image.open(img_path).convert("RGB")
        W, H = img.size
        print(f"  page_type: {result.page_type if result else 'N/A'}  entries: {len(result.entries) if result else 0}", flush=True)
        page_rows = []
        for ei, entry in enumerate(result.entries if result else []):
            for pi, photo in enumerate(entry.photos):
                box = photo.box_2d
                if not isinstance(box, (list, tuple)) or len(box) != 4:
                    print(f"    [skip] bad box_2d on page {pn}: {box}", flush=True)
                    continue
                ymin, xmin, ymax, xmax = box
                x1, y1 = int(xmin / 1000 * W), int(ymin / 1000 * H)
                x2, y2 = int(xmax / 1000 * W), int(ymax / 1000 * H)
                if x2 <= x1 or y2 <= y1:
                    continue
                crop = img.crop((x1, y1, x2, y2))
                cpath = out_dir / "crops" / f"page{pn:04d}_e{ei}_p{pi}.png"
                crop.save(cpath)
                page_rows.append([pn, entry.name, entry.dosage_text, photo.color, photo.shape, str(cpath)])

        # 페이지 단위 즉시 flush (크래시해도 손실 최소화)
        if page_rows:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(page_rows)
            total += len(page_rows)
        with open(done_path, "a", encoding="utf-8") as f:
            f.write(f"{pn}\n")

    print(f"\n이번 실행 {total}개 추출 (누적은 {csv_path} 참고) -> {out_dir}/crops/", flush=True)


if __name__ == "__main__":
    main()
