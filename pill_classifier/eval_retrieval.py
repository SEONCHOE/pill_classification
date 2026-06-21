"""
eval_retrieval.py
==============================================================
FAISS 갤러리에서 top-1/top-5 NDC 검색 정확도 평가.
  - train 클래스: 학습 중 ArcFace head가 직접 본 클래스 (참고용, 당연히 높아야 함)
  - val 클래스: 학습 중 한 번도 안 본 클래스 (진짜 일반화 테스트, 핵심 지표)
  쿼리=front, 갤러리=back (자기 자신 제외하고 검색)
==============================================================
"""
import pickle
from pathlib import Path

import faiss
import numpy as np

OUT = "runs/arcface"


def main():
    index = faiss.read_index(str(Path(OUT) / "gallery.faiss"))
    with open(Path(OUT) / "gallery_meta.pkl", "rb") as f:
        entries = pickle.load(f)  # (ndc, view, path, split)

    all_vecs = index.reconstruct_n(0, index.ntotal)

    for target_split in ("train", "val"):
        query_idx = [i for i, e in enumerate(entries) if e[1] == "front" and e[3] == target_split]
        gallery_idx = [i for i, e in enumerate(entries) if e[1] == "back"]
        gallery_vecs = all_vecs[gallery_idx].astype("float32")
        gallery_ndcs = [entries[i][0] for i in gallery_idx]

        sub_index = faiss.IndexFlatIP(gallery_vecs.shape[1])
        sub_index.add(gallery_vecs)

        top1, top5, n = 0, 0, 0
        for qi in query_idx:
            q_ndc = entries[qi][0]
            q_vec = all_vecs[qi:qi+1].astype("float32")
            D, I = sub_index.search(q_vec, 5)
            ranked_ndcs = [gallery_ndcs[i] for i in I[0]]
            n += 1
            top1 += (ranked_ndcs[0] == q_ndc)
            top5 += (q_ndc in ranked_ndcs)

        print(f"[{target_split}] n={n} top1={top1/max(1,n):.3f} top5={top5/max(1,n):.3f}")


if __name__ == "__main__":
    main()
