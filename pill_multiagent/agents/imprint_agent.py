"""
Agent 1: 각인 탐지 (Localization + OCR)
여러 각도의 알약 이미지를 Multimodal LLM에 동시에 전달하여
각인 영역을 탐지(localization)하고 텍스트를 추출(OCR)한다.

선행연구 대비 차별점:
  - YOLO/Tesseract 2단계 파이프라인 대신 MLLM 단일 호출로 localization+OCR 통합
  - 다각도 이미지를 한 번에 전달하여 최적 각도를 모델이 직접 선택
  - 음각(debossed) 각인도 맥락 추론으로 처리

IMPRINT_MODEL 환경변수로 모델 교체 가능 (ablation 실험용):
  gemini-2.5-flash    기본값, 가성비 최고
  claude-sonnet-4-6   성능 상한선
  claude-haiku-4-5    Claude 계열 가성비
  gpt-4o              OpenAI 비교군
  gpt-4o-mini         최저비용 비교점
"""

import os
import base64
import json
import re
import io
from typing import List

IMPRINT_MODEL         = os.getenv("IMPRINT_MODEL", "gemini-2.5-flash")
IMPRINT_MODEL_FALLBACK = os.getenv("IMPRINT_MODEL_FALLBACK", "claude-haiku-4-5")

# ── JSON 파싱 유틸 ────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """마크다운 코드펜스(```json```) 포함 여부에 관계없이 JSON 추출"""
    # 코드펜스 제거
    text = re.sub(r"```(?:json)?\s*", "", raw).strip()
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {
        "imprint_text": "",
        "confidence": 0.0,
        "best_image_index": 0,
        "location": "none",
        "notes": raw,
    }


# ── 공통 프롬프트 ──────────────────────────────────────────────────────────────

IMPRINT_SYSTEM_PROMPT = """당신은 알약 각인 분석 전문가입니다.
주어진 알약 이미지들을 분석하여 각인(imprint) 정보를 정확하게 추출합니다.

알약 각인의 특성:
- 알약 표면에 새겨진 문자, 숫자, 기호, 로고
- 볼록하게 양각(embossed) 또는 오목하게 음각(debossed) 형태
- 분할선(score line)은 각인이 아님
- 여러 면에 서로 다른 각인이 있을 수 있음
"""

IMPRINT_USER_PROMPT = """위 알약 이미지 {n}장을 모두 분석하여 각인 정보를 추출해주세요.

분석 기준:
1. 모든 이미지에서 알약 표면의 문자/숫자/기호를 찾으세요
2. 가장 선명하게 각인이 보이는 이미지를 선택하세요
3. 음각(debossed) 각인은 그림자로만 보일 수 있으니 주의하세요
4. 분할선(가운데 선)은 각인 텍스트에서 제외하세요

반드시 아래 JSON 형식으로만 답하세요 (JSON 외 텍스트 없이):
{{
    "imprint_text": "인식된 각인 텍스트 (없으면 빈 문자열, 앞면과 뒷면이 다르면 '/'로 구분)",
    "confidence": 0.0에서 1.0 사이의 신뢰도,
    "best_image_index": 각인이 가장 선명한 이미지의 인덱스 (0부터 시작),
    "location": "front / back / both / none 중 하나",
    "notes": "음각여부, 분할선 유무, 특이사항 등"
}}"""


# ── Gemini ────────────────────────────────────────────────────────────────────

def _detect_with_gemini(images: List[bytes], model: str) -> dict:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

    parts = []
    for i, img_bytes in enumerate(images):
        parts.append(types.Part.from_text(text=f"[이미지 {i + 1}]"))
        parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
    parts.append(types.Part.from_text(text=IMPRINT_USER_PROMPT.format(n=len(images))))

    response = client.models.generate_content(
        model=model,
        contents=parts,
        config=types.GenerateContentConfig(
            system_instruction=IMPRINT_SYSTEM_PROMPT,
            max_output_tokens=512,
        ),
    )

    raw = response.text.strip()
    return _parse_json(raw)


# ── Claude ────────────────────────────────────────────────────────────────────

def _detect_with_claude(images: List[bytes], model: str) -> dict:
    import anthropic

    client = anthropic.Anthropic()

    content = []
    for i, img_bytes in enumerate(images):
        content.append({"type": "text", "text": f"[이미지 {i + 1}]"})
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": base64.standard_b64encode(img_bytes).decode("utf-8"),
            },
        })
    content.append({"type": "text", "text": IMPRINT_USER_PROMPT.format(n=len(images))})

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=IMPRINT_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    )

    raw = response.content[0].text.strip()
    return _parse_json(raw)


# ── OpenAI ────────────────────────────────────────────────────────────────────

def _detect_with_openai(images: List[bytes], model: str) -> dict:
    from openai import OpenAI

    client = OpenAI()

    content = []
    for i, img_bytes in enumerate(images):
        content.append({"type": "text", "text": f"[이미지 {i + 1}]"})
        b64 = base64.standard_b64encode(img_bytes).decode("utf-8")
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
        })
    content.append({"type": "text", "text": IMPRINT_USER_PROMPT.format(n=len(images))})

    response = client.chat.completions.create(
        model=model,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": IMPRINT_SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ],
    )

    raw = response.choices[0].message.content.strip()
    return _parse_json(raw)


# ── 통합 진입점 ───────────────────────────────────────────────────────────────

def detect_imprint(images: List[bytes], model: str = IMPRINT_MODEL) -> dict:
    """여러 각도의 알약 이미지에서 각인 탐지 및 텍스트 추출.

    Gemini 503/429 발생 시 IMPRINT_MODEL_FALLBACK 모델로 자동 전환.
    """
    try:
        return _call_model(images, model)
    except Exception as primary_err:
        if model.startswith("gemini") and IMPRINT_MODEL_FALLBACK != model:
            import warnings
            warnings.warn(
                f"[Agent1] {model} 실패 ({primary_err.__class__.__name__}), "
                f"{IMPRINT_MODEL_FALLBACK}로 fallback",
                stacklevel=2,
            )
            return _call_model(images, IMPRINT_MODEL_FALLBACK)
        raise


def _call_model(images: List[bytes], model: str) -> dict:
    if model.startswith("gemini"):
        return _detect_with_gemini(images, model)
    elif model.startswith("claude"):
        return _detect_with_claude(images, model)
    else:
        return _detect_with_openai(images, model)


# ── LangGraph 노드 ────────────────────────────────────────────────────────────

def agent_imprint_detector(state: dict) -> dict:
    """Agent 1 노드: 각인 탐지 (Localization + OCR)"""
    try:
        result = detect_imprint(state["images"])
        return {
            **state,
            "imprint_text":       result.get("imprint_text", ""),
            "imprint_confidence": result.get("confidence", 0.0),
            "best_image_idx":     result.get("best_image_index", 0),
            "imprint_notes":      result.get("notes", ""),
        }
    except Exception as e:
        return {
            **state,
            "imprint_text":       "",
            "imprint_confidence": 0.0,
            "best_image_idx":     0,
            "error":              f"[Agent1] 각인 탐지 실패: {e}",
        }
