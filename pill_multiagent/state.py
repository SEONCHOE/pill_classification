from typing import TypedDict, Optional, List


class PillState(TypedDict):
    # ── 입력 ──────────────────────────────────────────
    images: List[bytes]                     # 여러 각도 이미지 (1~4장)
    patient_type: Optional[str]             # 노인/임산부/간질환/신질환/소아
    current_medications: Optional[List[str]]

    # ── Agent 1: 각인 탐지 결과 ───────────────────────
    imprint_text: Optional[str]             # OCR로 인식된 각인 텍스트
    imprint_confidence: Optional[float]     # 각인 인식 신뢰도
    best_image_idx: Optional[int]           # 각인이 가장 선명한 이미지 인덱스
    imprint_notes: Optional[str]            # 음각·분할선 등 특이사항

    # ── Agent 2: 알약 감별 결과 ───────────────────────
    drug_name: Optional[str]                # 감별된 약물명
    drug_code: Optional[str]               # 약물 코드
    confidence: Optional[float]            # 감별 신뢰도
    top3_candidates: Optional[List[dict]]  # 상위 3개 후보 [{drug, confidence}]

    # ── Agent 3: 안전정보 결과 ────────────────────────
    safety_info: Optional[str]             # 최종 안전정보 텍스트

    # ── Ablation / 실험 제어 ──────────────────────────
    ablation_mode: Optional[str]           # "oracle" | "auto" | "image_only"
    gate_weight: Optional[float]           # Confidence Gate 적용 가중치 (0~1)

    # ── 흐름 제어 ─────────────────────────────────────
    error: Optional[str]
    needs_confirmation: Optional[bool]     # 사용자 확인 필요 여부
