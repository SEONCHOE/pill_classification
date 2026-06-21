# 한국 알약 각인 탐지 — 단계별 Fine-tuning 계획

YOLO26 각인 탐지기를 RxIMAGE(미국)로 먼저 학습한 뒤,
약물사전 + AI Hub 경구약제 이미지로 한국 알약에 도메인 적응시키는 단계 계획.

---

## Stage 0 — 라벨 부트스트랩 (가장 중요한 사전작업)

**문제:** RxIMAGE도, AI Hub 경구약제 데이터도 "각인 영역 bbox" 라벨이
대개 없음. (AI Hub는 보통 알약 전체 박스 + 약 분류 라벨이지, 각인 영역 박스가 아님)

**해결:** 앞서 논의한 하이브리드로 각인 박스를 생성
1. Gemini 2.5 Flash로 각인 영역 bbox 자동 제안 (단일 호출, JSON 출력)
2. 좌표를 YOLO 포맷(`<cls> <xc> <yc> <w> <h>`, 0~1)으로 변환
3. **수동 QA**: 일부(예: 10~20%) 검수·보정 → 품질 확인
4. 분할선(score_line)은 별도 클래스로 라벨

> 팁: Gemini는 0~1000 정규화 좌표를 반환 → /1000 후 YOLO 포맷 변환.
> 박스가 느슨하면 약간 패딩 축소 또는 SAM으로 refine.

---

## Stage 1 — RxIMAGE로 베이스 학습 (영문/미국 알약)

- 데이터: RxIMAGE 각인 박스 (Stage 0에서 생성)
- 설정: `train_yolo26_imprint.py` 그대로 (P2 detection, imgsz 960)
- 목표: "각인 영역"이라는 일반적 시각 패턴을 먼저 학습
- 산출물: `best.pt` (베이스 가중치)

평가: mAP50 ≥ 0.9 목표(영역 탐지는 비교적 쉬움). 낮으면 라벨 품질/해상도 점검.

---

## Stage 2 — 한국 알약 도메인 적응

- 데이터: **AI Hub 경구약제 이미지** + 보유 약물사전 매칭분
  - 주의: AI Hub 데이터 **이용 약관/라이선스** 확인 후 사용
  - 라벨 포맷이 다르면(예: 다른 좌표계/JSON) YOLO 포맷으로 변환 스크립트 필요
  - 각인 박스가 없으면 Stage 0 부트스트랩을 한국 데이터에도 반복
- 방법: Stage 1의 `best.pt`에서 이어 학습(continue) — 처음부터 X
  ```python
  model = YOLO("runs/imprint/yolo26n_p2_detect/weights/best.pt")
  model.train(data="imprint_kr.yaml", epochs=100, imgsz=960,
              freeze=10, lr0=0.005)   # 더 낮은 LR로 미세조정
  ```
- 도메인 갭 요인(한국 특화):
  - 식약처 **의약품 식별표시** 체계(마크/문자/숫자 조합)
  - 한글 각인·제조사 로고가 섞일 수 있음
  - 조명/배경/해상도 분포 차이

---

## Stage 3 — STR(문자 인식) 한국 대응

각인 탐지(YOLO) 다음 단계인 STR도 한국 알약에 맞게 확장:

- **vocabulary 확장**: 영숫자(A–Z,0–9) + 한글 자모/문자 + 식별표시 심볼
- **lexicon = 약물사전**: 보유 약물사전의 각인 문자열 집합을
  closed-set 후보로 사용 → edit-distance 보정으로 OCR 오류 교정
  (513종처럼 후보가 닫혀 있으면 정확도 크게 상승)
- 경량 STR(CRNN) 권장(모바일). 합성 각인 데이터로 인식 학습 보강.

---

## Stage 4 — 평가 설계

| 단계 | 지표 |
|---|---|
| 각인 탐지(YOLO) | mAP50, mAP50-95, precision, recall (클래스별) |
| crop 품질 | 예측 박스가 실제 각인 전체를 포함하는 비율(완전성) |
| STR 인식 | 글자정확도(CER), 시퀀스정확도, edit distance |
| 보정 효과 | lexicon 보정 전/후 정확도 차이(ablation) |
| End-to-end | 최종 알약 분류 accuracy / top-k |

ablation 권장: (a)각인만 / (b)외관만 / (c)외관+각인 →
멀티모달 융합의 기여 정량화.

---

## 전체 런타임 파이프라인 (참고)

```
촬영(앞/뒷면)
   → YOLO26-P2 : 각인 영역 detection (온디바이스, NMS-free)
   → 전용 STR  : 각인 문자 인식 (CRNN, 한글+영숫자)
   → lexicon 보정 : 약물사전 closed-set으로 교정
   → MobileNet 멀티인풋 : 앞+뒤+각인텍스트 → 알약 식별
   → 안전정보 LLM 에이전트 : 상호작용/금기 (DB grounding/RAG)
```

런타임에는 MLLM이 없음. Gemini Flash는 Stage 0 라벨 부트스트랩과
저신뢰 폴백에서만 사용.

---

## 체크리스트

- [ ] AI Hub 경구약제 데이터 라이선스/이용약관 확인
- [ ] 각인 박스 라벨 부트스트랩 + 수동 QA 파이프라인 구축
- [ ] 데이터 분할 시 같은 약의 앞/뒷면이 train/val에 누수되지 않게 분리
- [ ] 한국 식별표시·한글 각인 케이스 별도 수집(롱테일)
- [ ] STR vocabulary에 한글 포함 여부 데이터로 확정
- [ ] 약물사전 = 분류 라벨공간 + STR 보정 lexicon 통일
