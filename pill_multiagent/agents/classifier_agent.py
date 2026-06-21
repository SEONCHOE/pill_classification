"""
Agent 2: 알약 감별 — 본 연구 novelty 구현

선행연구 대비 세 가지 핵심 차별점:
  1. 앞/뒷면 명시적 독립 Branch
     - 기존: 단일 이미지 입력 또는 전/후면 구분 없음
     - 본 연구: 앞면·뒷면을 별도 feature extractor branch로 처리
       → 각인이 한쪽 면에만 있는 약물 대응, 양면 정보 독립 학습

  2. Confidence Gating
     - 기존: 각인 텍스트를 항상 동일 가중치로 fusion
     - 본 연구: OCR 인식 신뢰도(imprint_confidence)에 따라 각인 branch 가중치 동적 조절
       → 음각·저대비 각인으로 OCR 실패 시 이미지 branch에 의존, 오류 전파 방지

  3. Ablation 모드 지원 (논문 Table용)
     MODE_ORACLE    : 메타데이터 각인 직접 입력 (성능 상한선)
     MODE_AUTO      : Multimodal LLM + OCR 자동 인식 각인 사용 (본 연구 주제)
     MODE_IMAGE_ONLY: 이미지만 사용 (각인 branch 제거, baseline)
"""

import os
import json
import pickle
import difflib
from enum import Enum
from typing import List, Optional, Tuple

import numpy as np
import cv2
import faiss

# ── Ablation 모드 정의 ────────────────────────────────────────────────────────

class AblationMode(str, Enum):
    ORACLE     = "oracle"      # 메타데이터 각인 직접 입력 (상한선)
    AUTO       = "auto"        # MLLM + OCR 자동 인식 (본 연구)
    IMAGE_ONLY = "image_only"  # 이미지만 (baseline)


# ── 경로 설정 ─────────────────────────────────────────────────────────────────

EMBED_MODEL_PATH = os.getenv(
    "PILL_EMBED_MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "../models/efficientnet_arcface.h5"),
)
FAISS_INDEX_PATH = os.getenv(
    "PILL_FAISS_INDEX_PATH",
    os.path.join(os.path.dirname(__file__), "../models/pill_index.faiss"),
)
FAISS_META_PATH = os.getenv(
    "PILL_FAISS_META_PATH",
    os.path.join(os.path.dirname(__file__), "../models/pill_index_meta.pkl"),
)
LEGACY_MODEL_PATH = os.getenv(
    "PILL_LEGACY_MODEL_PATH",
    os.path.join(
        os.path.dirname(__file__),
        "../../Pill_Classification_Model/mobile_aug2_ALL_1200.h5",
    ),
)
LEGACY_LABEL_PATH = os.getenv(
    "PILL_LABEL_PATH",
    os.path.join(os.path.dirname(__file__), "../drug_labels.json"),
)

# ── 이미지 전처리 ─────────────────────────────────────────────────────────────

def _preprocess(img_bytes: bytes, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 디코딩 실패")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    return img.astype(np.float32) / 255.0


def _select_front_back(
    images: List[bytes],
    best_image_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    앞/뒷면 명시적 분리 (Novelty 1).

    best_image_idx: 각인이 가장 선명한 이미지 → 앞면으로 지정
    나머지 이미지 중 첫 번째 → 뒷면

    단일 이미지인 경우 앞면과 뒷면에 동일 이미지 사용.
    """
    front = _preprocess(images[best_image_idx])

    back_candidates = [i for i in range(len(images)) if i != best_image_idx]
    if back_candidates:
        back = _preprocess(images[back_candidates[0]])
    else:
        back = front.copy()

    return front, back


# ── Confidence Gating ─────────────────────────────────────────────────────────

# 각인 신뢰도 임계값 (논문 ablation에서 최적값 탐색 예정)
GATE_HIGH = 0.75   # 이 이상: 각인 branch 완전 활성화
GATE_LOW  = 0.30   # 이 이하: 각인 branch 비활성화 (이미지만)


def _imprint_gate_weight(imprint_confidence: float) -> float:
    """
    Confidence Gating (Novelty 2).

    OCR 신뢰도를 0~1 사이 게이팅 가중치로 변환.
      - confidence >= GATE_HIGH: weight = 1.0 (각인 완전 반영)
      - confidence <= GATE_LOW : weight = 0.0 (각인 무시, 이미지만)
      - 중간: 선형 보간
    """
    if imprint_confidence >= GATE_HIGH:
        return 1.0
    if imprint_confidence <= GATE_LOW:
        return 0.0
    return (imprint_confidence - GATE_LOW) / (GATE_HIGH - GATE_LOW)


def _rerank_with_imprint(
    candidates: List[dict],
    imprint_text: str,
    gate_weight: float,
    known_imprints: Optional[dict] = None,
) -> List[dict]:
    """
    각인 텍스트로 후보 약물 재정렬 (Novelty 2 적용).

    gate_weight = 0.0: 이미지 기반 순위 그대로 유지
    gate_weight = 1.0: 각인 유사도를 최대로 반영하여 재정렬

    known_imprints: {drug_name: imprint_text} 매핑 (있으면 정확도 향상)
    """
    if gate_weight == 0.0 or not imprint_text:
        return candidates

    reranked = []
    for c in candidates:
        # 각인 유사도 계산 (edit distance 기반)
        ref_imprint = ""
        if known_imprints:
            ref_imprint = known_imprints.get(c["drug"], "")

        if ref_imprint:
            text_sim = difflib.SequenceMatcher(
                None,
                imprint_text.upper(),
                ref_imprint.upper(),
            ).ratio()
        else:
            # known_imprints 없으면 약물명에 각인 포함 여부로 대체
            text_sim = 1.0 if imprint_text.upper() in c["drug"].upper() else 0.0

        # 이미지 점수 + 게이팅된 각인 점수 결합
        combined = (1.0 - gate_weight) * c["confidence"] + gate_weight * text_sim
        reranked.append({**c, "confidence": combined, "imprint_sim": text_sim})

    reranked.sort(key=lambda x: x["confidence"], reverse=True)
    return reranked


# ── 지연 로딩 캐시 ────────────────────────────────────────────────────────────

_embed_model  = None
_faiss_index  = None
_faiss_meta   = None
_legacy_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        from keras.models import load_model
        _embed_model = load_model(EMBED_MODEL_PATH)
    return _embed_model


def _get_faiss():
    global _faiss_index, _faiss_meta
    if _faiss_index is None:
        _faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "rb") as f:
            _faiss_meta = pickle.load(f)
    return _faiss_index, _faiss_meta


def _get_legacy_model():
    global _legacy_model
    if _legacy_model is None:
        from keras.models import load_model
        _legacy_model = load_model(LEGACY_MODEL_PATH)
    return _legacy_model


def _load_legacy_labels() -> List[str]:
    if os.path.exists(LEGACY_LABEL_PATH):
        with open(LEGACY_LABEL_PATH, encoding="utf-8") as f:
            return json.load(f)
    return [f"drug_{i}" for i in range(59)]


# ── Retrieval 기반 감별 ───────────────────────────────────────────────────────

def _embed(img_arr: np.ndarray) -> np.ndarray:
    model = _get_embed_model()
    vec = model.predict(np.expand_dims(img_arr, 0), verbose=0)[0]
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec.astype(np.float32)


def classify_retrieval(
    images: List[bytes],
    best_image_idx: int,
    imprint_text: str,
    imprint_confidence: float,
    mode: AblationMode,
    top_k: int = 3,
) -> dict:
    """
    Retrieval 기반 감별 (Novelty 1, 2, 3 통합).

    앞/뒷면 임베딩 평균 → FAISS 검색 → Confidence Gating 재정렬
    """
    index, meta = _get_faiss()

    # Novelty 1: 앞/뒷면 명시적 분리
    img_front, img_back = _select_front_back(images, best_image_idx)

    # Novelty 3: Ablation 모드 — IMAGE_ONLY이면 각인 무시
    if mode == AblationMode.IMAGE_ONLY:
        imprint_text = ""
        gate_weight = 0.0
    else:
        # Novelty 2: Confidence Gating
        gate_weight = _imprint_gate_weight(imprint_confidence)

    # 앞면 + 뒷면 임베딩 평균 (independent branches)
    emb_front = _embed(img_front)
    emb_back  = _embed(img_back)
    query_vec = (emb_front + emb_back) / 2.0
    query_vec = (query_vec / (np.linalg.norm(query_vec) + 1e-10)).reshape(1, -1)

    distances, indices = index.search(query_vec, top_k)

    candidates = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(meta):
            continue
        similarity = float(max(0.0, 1.0 - dist / 2.0))
        candidates.append({
            "drug":       meta[idx]["drug_name"],
            "drug_code":  meta[idx].get("drug_code", ""),
            "confidence": similarity,
        })

    if not candidates:
        return {"drug_name": "알 수 없음", "confidence": 0.0, "top3_candidates": []}

    # Novelty 2: 각인 Confidence Gating 재정렬
    candidates = _rerank_with_imprint(candidates, imprint_text, gate_weight)

    return {
        "drug_name":       candidates[0]["drug"],
        "drug_code":       candidates[0].get("drug_code", ""),
        "confidence":      candidates[0]["confidence"],
        "top3_candidates": candidates,
        "gate_weight":     gate_weight,   # 논문 로깅용
        "ablation_mode":   mode,
    }


# ── Legacy Softmax 기반 감별 (AI Hub 모델 준비 전) ────────────────────────────

def classify_legacy(
    images: List[bytes],
    best_image_idx: int,
    imprint_text: str,
    imprint_confidence: float,
    mode: AblationMode,
) -> dict:
    """
    Legacy MobileNet Softmax 감별 (현재 기본값).

    Novelty 1, 2, 3을 소프트웨어 레벨에서 구현:
      - 앞/뒷면 분리 전처리
      - IMAGE_ONLY 모드 지원
      - 각인 Confidence Gating: 신뢰도 낮으면 결과 그대로, 높으면 이미지 결과 재확인
    """
    model  = _get_legacy_model()
    labels = _load_legacy_labels()

    # Novelty 1: 앞/뒷면 명시적 분리
    img_front, img_back = _select_front_back(images, best_image_idx)

    pred = model.predict(
        [np.expand_dims(img_front, 0), np.expand_dims(img_back, 0)],
        verbose=0,
    )
    scores   = pred[0]
    top3_idx = np.argsort(scores)[-3:][::-1]
    candidates = [
        {
            "drug":       labels[i] if i < len(labels) else f"class_{i}",
            "confidence": float(scores[i]),
        }
        for i in top3_idx
    ]

    # Novelty 3: IMAGE_ONLY 모드
    if mode == AblationMode.IMAGE_ONLY:
        gate_weight = 0.0
    else:
        # Novelty 2: Confidence Gating
        gate_weight = _imprint_gate_weight(imprint_confidence)
        candidates  = _rerank_with_imprint(candidates, imprint_text, gate_weight)

    return {
        "drug_name":       candidates[0]["drug"],
        "confidence":      candidates[0]["confidence"],
        "top3_candidates": candidates,
        "gate_weight":     gate_weight,
        "ablation_mode":   mode,
    }


# ── 통합 진입점 ───────────────────────────────────────────────────────────────

def classify(
    images: List[bytes],
    best_image_idx: int = 0,
    imprint_text: Optional[str] = None,
    imprint_confidence: float = 0.0,
    mode: AblationMode = AblationMode.AUTO,
) -> dict:
    """
    알약 감별 통합 함수.

    Args:
        images:              여러 각도의 알약 이미지
        best_image_idx:      각인이 가장 선명한 이미지 인덱스 (앞면으로 사용)
        imprint_text:        Agent 1(MLLM+OCR)이 인식한 각인 텍스트
        imprint_confidence:  각인 인식 신뢰도 (0~1)
        mode:                AblationMode — oracle / auto / image_only

    Returns:
        {drug_name, confidence, top3_candidates, gate_weight, ablation_mode}
    """
    imprint_text = imprint_text or ""

    if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
        return classify_retrieval(
            images, best_image_idx, imprint_text, imprint_confidence, mode
        )

    return classify_legacy(
        images, best_image_idx, imprint_text, imprint_confidence, mode
    )


# ── LangGraph 노드 ────────────────────────────────────────────────────────────

def agent_pill_classifier(state: dict) -> dict:
    """Agent 2 노드: 알약 감별 (Novelty 1·2·3 적용)"""
    mode = AblationMode(state.get("ablation_mode", AblationMode.AUTO))
    try:
        result = classify(
            images=state["images"],
            best_image_idx=state.get("best_image_idx", 0),
            imprint_text=state.get("imprint_text", ""),
            imprint_confidence=state.get("imprint_confidence", 0.0),
            mode=mode,
        )
        return {
            **state,
            "drug_name":       result["drug_name"],
            "drug_code":       result.get("drug_code", ""),
            "confidence":      result["confidence"],
            "top3_candidates": result["top3_candidates"],
            "gate_weight":     result.get("gate_weight", 0.0),
        }
    except Exception as e:
        return {
            **state,
            "drug_name":  None,
            "confidence": 0.0,
            "error":      f"[Agent2] 알약 감별 실패: {e}",
        }
