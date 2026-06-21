"""
Agent 3: 안전정보 제공
기존 pill_GPT의 RAG 시스템을 재활용하여 약물 안전정보를 제공한다.
FAISS + Chroma 앙상블 Retriever → GPT-4o 응답 생성.
"""

import os
from typing import List, Optional

from langchain_community.vectorstores import FAISS, Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# 벡터 DB 경로 — pill_GPT/pill_vector_db (repo에 포함된 버전 우선)
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

FAISS_PATH = os.getenv(
    "FAISS_DB_PATH",
    os.path.join(_REPO_ROOT, "pill_GPT/pill_vector_db/faiss_all_0722"),
)
CHROMA_PATH = os.getenv(
    "CHROMA_DB_PATH",
    os.path.join(_REPO_ROOT, "pill_GPT/pill_vector_db/chroma_all_0722"),
)

# pill_GPT v2.07 기준 14개 소스 (drug_info_2006.pdf 포함)
METADATA_LABELS = [
    "약물정보집 2009",
    "병용금기 DUR정보",
    "연령금기 DUR정보",
    "임부금기 DUR정보",
    "효능군중복주의 DUR정보",
    "용량주의 DUR정보",
    "투여기간 DUR정보",
    "노인주의 DUR정보",
    "수유부 주의 DUR정보",
    "간 질환 환자에 대한 의약품 적정사용 정보집",
    "신 질환 환자에 대한 의약품 적정사용 정보집",
    "노인에 대한 의약품 적정사용 정보집",
    "임산부에 대한 의약품 적정사용 정보집",
    "소아에 대한 의약품 적정사용 정보집",
]

_retriever = None  # 지연 로딩


def _get_retriever() -> EnsembleRetriever:
    global _retriever
    if _retriever is None:
        emb = OpenAIEmbeddings()
        faiss_vs = FAISS.load_local(
            FAISS_PATH, emb, allow_dangerous_deserialization=True
        )
        chroma_vs = Chroma(
            persist_directory=CHROMA_PATH, embedding_function=emb
        )
        _retriever = EnsembleRetriever(
            retrievers=[
                faiss_vs.as_retriever(search_kwargs={"k": 2}),
                chroma_vs.as_retriever(search_kwargs={"k": 2}),
            ],
            weights=[0.5, 0.5],
        )
    return _retriever


def _build_system_prompt(
    drug_name: str,
    patient_type: str,
    current_medications: List[str],
) -> str:
    med_section = ""
    if current_medications:
        meds = ", ".join(current_medications)
        med_section = f"""
현재 복용 중인 약물: {meds}
→ 위 약물들과의 병용금기, 중복투여(같은 효능군) 여부를 반드시 확인하여 알려주세요."""

    return f"""당신은 의약품 안전정보를 제공하는 전문 챗봇입니다.

━━━ 식별된 약물 정보 ━━━
약물명: {drug_name}
환자 유형: {patient_type if patient_type else "일반"}
{med_section}

아래 순서로 답해주세요:
1. 약물 효능 및 주요 부작용
2. {patient_type if patient_type else "일반"} 환자 특이 주의사항
3. 병용금기 및 중복투여 주의 (복용 중인 약물 있는 경우)
4. 권장 사항 및 복약 지도

규칙:
- 공감하되 공적인 존댓말 사용
- 출처는 괄호 안에 짧게 표기: [출처: 정보집명 p.XX]
- 마지막 문장은 반드시 "정확한 복약 정보는 담당 의사 또는 약사에게 확인하세요." 로 끝낼 것
- 참고 문서에 없는 내용은 "해당 내용은 참고자료에 없습니다"라고 한 번만 안내"""


def get_safety_info(
    drug_name: str,
    patient_type: Optional[str] = "",
    current_medications: Optional[List[str]] = None,
) -> str:
    """RAG 기반 안전정보 생성"""
    if current_medications is None:
        current_medications = []

    retriever = _get_retriever()
    query = f"{drug_name} 안전정보"
    if patient_type:
        query += f" {patient_type} 환자"

    docs = retriever.get_relevant_documents(query)

    docs_text = ""
    for i, doc in enumerate(docs):
        label = METADATA_LABELS[i] if i < len(METADATA_LABELS) else doc.metadata.get("source", "")
        docs_text += f"[{label}]\n{doc.page_content}\n\n"

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    messages = [
        SystemMessage(content=_build_system_prompt(drug_name, patient_type, current_medications)),
        HumanMessage(content=f"질문: {drug_name}의 안전정보를 알려주세요.\n\n참고 문서:\n{docs_text}"),
    ]

    response = llm.invoke(messages)
    return response.content


# ── LangGraph 노드 ─────────────────────────────────────────────────────────────

def agent_safety_provider(state: dict) -> dict:
    """Agent 3 노드: 안전정보 제공"""
    drug_name = state.get("drug_name") or "알 수 없음"
    try:
        safety_info = get_safety_info(
            drug_name=drug_name,
            patient_type=state.get("patient_type", ""),
            current_medications=state.get("current_medications", []),
        )
        return {**state, "safety_info": safety_info}
    except Exception as e:
        return {
            **state,
            "safety_info": f"{drug_name}에 대한 안전정보를 불러오지 못했습니다.",
            "error": f"[Agent3] 안전정보 조회 실패: {e}",
        }
