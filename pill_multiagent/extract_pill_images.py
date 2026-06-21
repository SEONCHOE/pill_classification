"""
drug_info_2006.pdf에서 알약 이미지 추출 스크립트

drug_info_2006.pdf 구조 (약물정보집):
  - 각 약물마다 1~2 페이지: 알약 사진 + 약물명 + 설명
  - 이미지와 주변 텍스트를 함께 추출하여 drug_name 매핑

실행:
  pip install pymupdf
  python extract_pill_images.py

출력:
  pill_img/extracted/       ← 추출된 이미지 파일들
  pill_img/pill_image_meta.csv  ← {image_file, drug_name, page_num, description}
"""

import os
import re
import csv
import json

import fitz  # PyMuPDF


PDF_PATH = os.path.join(os.path.dirname(__file__), "../pill_img/drug_info_2006.pdf")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "../pill_img/extracted")
META_CSV = os.path.join(os.path.dirname(__file__), "../pill_img/pill_image_meta.csv")
FAISS_META_JSON = os.path.join(os.path.dirname(__file__), "models/pill_image_meta_for_faiss.json")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_META_JSON), exist_ok=True)

# ── 텍스트에서 약물명 추출 ─────────────────────────────────────────────────────
# 약물정보집의 일반적 패턴: 약물명이 굵은 텍스트 또는 페이지 상단에 위치
# 실제 PDF 구조에 따라 아래 패턴 조정 필요

def extract_drug_name_from_text(text: str) -> str:
    """
    페이지 텍스트에서 약물명을 추출한다.
    약물명은 보통:
      - 줄 첫머리에 단독으로 있는 고유명사 (한글+영문 혼합)
      - 괄호 앞 단어 (예: 아스피린(Aspirin))
      - 숫자나 일반 단어가 아닌 첫 번째 명사구
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return ""

    for line in lines[:5]:  # 페이지 상단 5줄에서 약물명 탐색
        # 영문자+한글 혼합이거나 한글 약물명 패턴
        if re.match(r"^[가-힣a-zA-Z][\w가-힣\s\(\)·/-]{1,40}$", line):
            # 너무 짧거나 일반 단어 제외
            if len(line) >= 2 and not line.isdigit():
                return line.strip()

    return lines[0] if lines else ""


# ── PDF 이미지 추출 메인 ─────────────────────────────────────────────────────

def extract_images_from_pdf(
    min_width: int = 80,
    min_height: int = 80,
) -> list[dict]:
    """
    PDF 각 페이지에서 이미지를 추출하고 주변 텍스트로 약물명을 매핑한다.

    Returns:
        [{image_file, drug_name, page_num, description}]
    """
    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    print(f"총 {total_pages}페이지 처리 시작...")

    records = []
    img_counter = 0

    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text("text")
        drug_name = extract_drug_name_from_text(page_text)

        # 페이지에서 이미지 목록 가져오기
        image_list = page.get_images(full=True)

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)

            width  = base_image["width"]
            height = base_image["height"]
            ext    = base_image["ext"]  # jpeg, png 등

            # 너무 작은 이미지(로고, 구분선 등) 제외
            if width < min_width or height < min_height:
                continue

            img_filename = f"page{page_num+1:04d}_img{img_index+1:02d}_{img_counter:05d}.{ext}"
            img_path = os.path.join(OUT_DIR, img_filename)

            with open(img_path, "wb") as f:
                f.write(base_image["image"])

            records.append({
                "image_file": img_filename,
                "drug_name": drug_name,
                "page_num": page_num + 1,
                "width": width,
                "height": height,
                "description": page_text[:200].replace("\n", " ").strip(),
            })
            img_counter += 1

        if (page_num + 1) % 100 == 0:
            print(f"  {page_num + 1}/{total_pages} 페이지 완료 ({img_counter}개 이미지)")

    doc.close()
    return records


# ── 결과 저장 ─────────────────────────────────────────────────────────────────

def save_results(records: list[dict]):
    # CSV 저장 (검토용)
    with open(META_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["image_file", "drug_name", "page_num", "width", "height", "description"])
        writer.writeheader()
        writer.writerows(records)

    # FAISS 인덱스 빌드용 JSON 저장
    faiss_meta = [
        {
            "drug_name": r["drug_name"],
            "drug_code": "",  # drug_info_2006.pdf에는 코드 없음
            "image_path": r["image_file"],
            "source": "drug_info_2006",
        }
        for r in records
        if r["drug_name"]  # 약물명 매핑 성공한 것만
    ]
    with open(FAISS_META_JSON, "w", encoding="utf-8") as f:
        json.dump(faiss_meta, f, ensure_ascii=False, indent=2)

    print(f"\n완료!")
    print(f"  추출 이미지: {len(records)}개 → {OUT_DIR}")
    print(f"  메타데이터 CSV: {META_CSV}")
    print(f"  FAISS 빌드용 JSON: {FAISS_META_JSON}")

    # 약물명 매핑 통계
    named = [r for r in records if r["drug_name"]]
    print(f"  약물명 매핑 성공: {len(named)}/{len(records)} ({len(named)/max(len(records),1)*100:.0f}%)")

    # 샘플 출력
    print("\n샘플 (첫 5개):")
    for r in records[:5]:
        print(f"  [{r['page_num']}p] {r['drug_name']:30s} → {r['image_file']}")


# ── 실행 후 수동 검토 안내 ────────────────────────────────────────────────────

REVIEW_GUIDE = """
=== 이미지 추출 후 수동 검토 사항 ===

1. pill_img/pill_image_meta.csv 열어서 drug_name 컬럼 확인
   - 약물명이 잘못 잡힌 경우 수동 수정

2. 이미지 품질 확인
   - extracted/ 폴더에서 실제 알약 이미지인지 확인
   - 표지/목차/도표 이미지는 제거

3. FAISS 인덱스 빌드
   python build_aihub_index.py \\
     --image_dir ../pill_img/extracted \\
     --meta_csv ../pill_img/pill_image_meta.csv

   단, build_aihub_index.py의 COL_* 변수를 CSV 컬럼명에 맞게 수정:
     COL_DRUG_NAME = "drug_name"
     COL_FRONT_IMG = "image_file"
"""

if __name__ == "__main__":
    records = extract_images_from_pdf()
    save_results(records)
    print(REVIEW_GUIDE)
