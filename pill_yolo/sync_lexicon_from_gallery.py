"""
sync_lexicon_from_gallery.py
==============================================================
외형(분류기) 갤러리에 새 알약 레퍼런스 이미지가 추가될 때,
그 이미지들의 각인 텍스트를 Gemini로 뽑아서 약물사전(lexicon)에 합친다.

배경: 외형 갤러리(FAISS, pill_classifier/)와 STR lexicon(drug_imprints.txt)은
      서로 다른 파이프라인이라 자동으로 동기화되지 않음. 갤러리에 새 NDC를
      추가할 때마다 이 스크립트를 같이 돌려서 두 시스템이 같은 약 목록을
      알게 만든다.

실행:
    python sync_lexicon_from_gallery.py --images_dir <새 알약 이미지 폴더>
    (이미 처리된 이미지는 자동으로 skip — datasets/str_*_gt/images 기준)
==============================================================
"""
import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
LEXICON_DEFAULT = HERE / "drug_imprints.txt"
ALREADY_PROCESSED_DIRS = [
    HERE / "datasets" / "str_train_gt" / "images",
    HERE / "datasets" / "str_val_gt" / "images",
]


def core_id(stem: str) -> str:
    """front_/back_ 접두사를 떼고 비교용 핵심 식별자만 남김."""
    for prefix in ("front_", "back_"):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


def already_processed_stems():
    stems = set()
    for d in ALREADY_PROCESSED_DIRS:
        if d.exists():
            stems |= {core_id(p.stem) for p in d.glob("*.png")}
    # 과거에 다른 sync 실행으로 만들어진 datasets/str_gallery_*_gt 도 포함
    for d in (HERE / "datasets").glob("str_gallery_*_gt"):
        img_dir = d / "images"
        if img_dir.exists():
            stems |= {core_id(p.stem) for p in img_dir.glob("*.png")}
    return stems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True, help="새 알약 레퍼런스 이미지 폴더")
    ap.add_argument("--lexicon", default=str(LEXICON_DEFAULT))
    ap.add_argument("--out_dir", default=None, help="OCR 중간결과 저장 위치(기본: datasets/str_gallery_<n>_gt)")
    ap.add_argument("--max_dist_dedup", action="store_true",
                     help="(예약) 향후 fuzzy dedup 옵션")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    all_imgs = [p for p in images_dir.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    processed = already_processed_stems()
    new_imgs = [p for p in all_imgs if core_id(p.stem) not in processed]

    print(f"전체 이미지: {len(all_imgs)}, 이미 처리됨: {len(all_imgs)-len(new_imgs)}, 새로 처리할 이미지: {len(new_imgs)}")
    if not new_imgs:
        print("새로 처리할 이미지가 없습니다. 종료.")
        return

    # out_dir 결정 (중복 방지용 인덱스 증가)
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        i = 1
        while (HERE / "datasets" / f"str_gallery_{i}_gt").exists():
            i += 1
        out_dir = HERE / "datasets" / f"str_gallery_{i}_gt"

    # 새 이미지만 모아서 임시 폴더에 복사 (auto_label_gemini.py는 폴더 단위로 동작)
    tmp_in = out_dir / "_input_new_only"
    tmp_in.mkdir(parents=True, exist_ok=True)
    for p in new_imgs:
        shutil.copy(p, tmp_in / p.name)

    print(f"-> auto_label_gemini.py 실행 (입력: {tmp_in}, 출력: {out_dir})")
    subprocess.run([sys.executable, str(HERE / "auto_label_gemini.py"),
                     "--images", str(tmp_in), "--out", str(out_dir)], check=True)

    # 결과 manifest에서 고유 텍스트 추출 -> lexicon에 병합
    manifest_path = out_dir / "str_manifest.csv"
    new_texts = set()
    if manifest_path.exists():
        with open(manifest_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row["text"].strip()
                if t:
                    new_texts.add(t)

    lexicon_path = Path(args.lexicon)
    existing = set()
    if lexicon_path.exists():
        existing = {l.strip() for l in lexicon_path.read_text(encoding="utf-8").splitlines() if l.strip()}

    merged = sorted(existing | new_texts)
    added = len(merged) - len(existing)
    lexicon_path.write_text("\n".join(merged) + "\n", encoding="utf-8")

    print(f"신규 텍스트 {len(new_texts)}개 중 {added}개가 사전에 새로 추가됨 "
          f"(기존 {len(existing)} -> 총 {len(merged)}) -> {lexicon_path}")


if __name__ == "__main__":
    main()
