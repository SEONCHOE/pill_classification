"""
AI Hub 의약품 낱알 이미지 데이터셋 → FAISS 인덱스 구축 스크립트

AI Hub 데이터 구조 (일반적):
  images/
    ├── 앞면/
    │   ├── 약품코드_앞면.jpg
    │   └── ...
    └── 뒷면/
        ├── 약품코드_뒷면.jpg
        └── ...
  metadata.csv  (품목기준코드, 품목명, 각인, 색상, 제형 등)

실행 예:
  python build_aihub_index.py \
    --image_dir /data/aihub/images \
    --meta_csv /data/aihub/metadata.csv \
    --output_dir ./models
"""

import argparse
import os
import json
import pickle

import numpy as np
import pandas as pd
import cv2
import faiss

# ── 커맨드라인 인수 ────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", required=True, help="AI Hub 이미지 루트 경로")
parser.add_argument("--meta_csv",  required=True, help="메타데이터 CSV 경로")
parser.add_argument("--output_dir", default="./models", help="인덱스 저장 경로")
parser.add_argument("--model_path", default="./models/efficientnet_arcface.h5")
parser.add_argument("--img_size", type=int, default=224)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# ── AI Hub 메타데이터 컬럼 매핑 ──────────────────────────────────────────────
# AI Hub 컬럼명이 다를 수 있으므로 환경에 맞게 수정
COL_DRUG_CODE  = "품목기준코드"   # 약물 고유 코드
COL_DRUG_NAME  = "품목명"         # 약물명
COL_FRONT_IMG  = "앞면이미지"     # 앞면 파일명
COL_BACK_IMG   = "뒷면이미지"     # 뒷면 파일명 (없으면 앞면 재사용)
COL_IMPRINT    = "각인"           # 각인 텍스트
COL_COLOR      = "색상"           # 색상
COL_SHAPE      = "모양"           # 제형/모양


def preprocess(img_bytes: bytes, size: int) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("디코딩 실패")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (size, size))
    return img.astype(np.float32) / 255.0


def load_image(path: str, size: int) -> np.ndarray:
    with open(path, "rb") as f:
        return preprocess(f.read(), size)


def main():
    print("=== AI Hub 데이터 → FAISS 인덱스 구축 ===")

    # 1. 메타데이터 로드
    df = pd.read_csv(args.meta_csv, encoding="utf-8-sig")
    print(f"총 {len(df)}개 약물 데이터")

    # 컬럼 확인
    for col in [COL_DRUG_CODE, COL_DRUG_NAME, COL_FRONT_IMG]:
        if col not in df.columns:
            print(f"[경고] 컬럼 '{col}' 없음. 실제 컬럼명 확인 필요: {list(df.columns)}")

    # 2. 임베딩 모델 로드
    from keras.models import load_model
    print(f"모델 로드: {args.model_path}")
    model = load_model(args.model_path)
    embed_dim = model.output_shape[-1]
    print(f"임베딩 차원: {embed_dim}")

    # 3. 임베딩 추출
    all_vectors = []
    all_meta = []
    failed = 0

    for i in range(0, len(df), args.batch_size):
        batch = df.iloc[i : i + args.batch_size]
        batch_imgs = []
        batch_meta_tmp = []

        for _, row in batch.iterrows():
            try:
                front_path = os.path.join(args.image_dir, str(row[COL_FRONT_IMG]))
                img = load_image(front_path, args.img_size)
                batch_imgs.append(img)

                meta = {
                    "drug_name":  str(row[COL_DRUG_NAME]),
                    "drug_code":  str(row[COL_DRUG_CODE]),
                    "imprint":    str(row.get(COL_IMPRINT, "")),
                    "color":      str(row.get(COL_COLOR, "")),
                    "shape":      str(row.get(COL_SHAPE, "")),
                    "image_path": str(row[COL_FRONT_IMG]),
                }
                batch_meta_tmp.append(meta)
            except Exception as e:
                failed += 1
                continue

        if not batch_imgs:
            continue

        batch_arr = np.stack(batch_imgs, axis=0)
        vecs = model.predict(batch_arr, verbose=0).astype(np.float32)

        # L2 정규화
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-10
        vecs = vecs / norms

        all_vectors.append(vecs)
        all_meta.extend(batch_meta_tmp)

        if (i // args.batch_size) % 20 == 0:
            print(f"  {i + len(batch)}/{len(df)} 완료 (실패: {failed})")

    all_vectors = np.vstack(all_vectors)
    print(f"\n임베딩 완료: {len(all_meta)}개 / 실패: {failed}개")

    # 4. FAISS 인덱스 생성 (Inner Product, L2 정규화 후 IP = cosine similarity)
    index = faiss.IndexFlatIP(embed_dim)
    index.add(all_vectors)

    # 5. 저장
    index_path = os.path.join(args.output_dir, "pill_index.faiss")
    meta_path  = os.path.join(args.output_dir, "pill_index_meta.pkl")
    labels_path = os.path.join(args.output_dir, "drug_labels.json")

    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(all_meta, f)

    # drug_labels.json (약물명 리스트)
    drug_names = sorted(set(m["drug_name"] for m in all_meta))
    with open(labels_path, "w", encoding="utf-8") as f:
        json.dump(drug_names, f, ensure_ascii=False, indent=2)

    print(f"\n저장 완료:")
    print(f"  FAISS 인덱스 : {index_path}  ({len(all_meta)}개 벡터)")
    print(f"  메타데이터   : {meta_path}")
    print(f"  약물 레이블  : {labels_path}  ({len(drug_names)}종)")


if __name__ == "__main__":
    main()
