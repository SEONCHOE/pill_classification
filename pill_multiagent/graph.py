"""
LangGraph 오케스트레이터
세 에이전트를 연결하고 신뢰도에 따른 분기를 처리한다.

흐름:
  imprint_agent → classify_agent
                      ├─ conf ≥ 0.85 → safety_agent → END
                      ├─ 0.5 ≤ conf < 0.85 → confirm_agent → safety_agent → END
                      └─ conf < 0.5 또는 오류 → fallback_agent → safety_agent → END
"""

import base64
import json
import re
from typing import Literal

import anthropic
from langgraph.graph import StateGraph, END

from state import PillState
from agents import (
    agent_imprint_detector,
    agent_pill_classifier,
    agent_safety_provider,
)

# ── 신뢰도 기반 분기 ──────────────────────────────────────────────────────────

def route_after_classification(
    state: PillState,
) -> Literal["safety_agent", "confirm_agent", "fallback_agent"]:
    if state.get("error") or not state.get("drug_name"):
        return "fallback_agent"
    conf = state.get("confidence", 0.0)
    if conf >= 0.85:
        return "safety_agent"
    elif conf >= 0.50:
        return "confirm_agent"
    return "fallback_agent"


# ── confirm 노드: 사용자 확인 요청 플래그 세팅 ───────────────────────────────

def agent_confirm(state: PillState) -> PillState:
    """신뢰도 중간(50~85%): 사용자에게 후보 선택 요청 플래그"""
    return {**state, "needs_confirmation": True}


# ── fallback 노드: Multimodal LLM이 직접 감별 시도 ───────────────────────────

_FALLBACK_PROMPT = """\
이 알약을 분석해주세요.
{imprint_info}
이미지를 보고 알약이 어떤 약인지 추정해주세요.

반드시 아래 JSON 형식으로만 답하세요:
{{"drug_name": "추정 약물명", "confidence": 0.0~1.0}}
"""


def agent_fallback(state: PillState) -> PillState:
    """신뢰도 낮거나 분류 실패 시 Claude Vision으로 직접 감별"""
    client = anthropic.Anthropic()

    best_idx = state.get("best_image_idx") or 0
    img_bytes = state["images"][best_idx]
    img_b64 = base64.standard_b64encode(img_bytes).decode()

    imprint_text = state.get("imprint_text", "")
    imprint_info = f"각인 텍스트: '{imprint_text}'" if imprint_text else "각인 텍스트: 인식 불가"

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": _FALLBACK_PROMPT.format(imprint_info=imprint_info),
                    },
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    json_match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        result = json.loads(json_match.group()) if json_match else {}
        return {
            **state,
            "drug_name": result.get("drug_name", "알 수 없음"),
            "confidence": float(result.get("confidence", 0.3)),
            "error": None,
        }
    except Exception:
        return {**state, "drug_name": "알 수 없음", "confidence": 0.0}


# ── 그래프 조립 ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(PillState)

    g.add_node("imprint_agent", agent_imprint_detector)
    g.add_node("classify_agent", agent_pill_classifier)
    g.add_node("safety_agent", agent_safety_provider)
    g.add_node("confirm_agent", agent_confirm)
    g.add_node("fallback_agent", agent_fallback)

    g.set_entry_point("imprint_agent")
    g.add_edge("imprint_agent", "classify_agent")

    g.add_conditional_edges(
        "classify_agent",
        route_after_classification,
        {
            "safety_agent": "safety_agent",
            "confirm_agent": "confirm_agent",
            "fallback_agent": "fallback_agent",
        },
    )

    # confirm / fallback 이후 → 안전정보
    g.add_edge("confirm_agent", "safety_agent")
    g.add_edge("fallback_agent", "safety_agent")
    g.add_edge("safety_agent", END)

    return g.compile()


pill_app = build_graph()
