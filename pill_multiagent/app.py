"""
Streamlit UI
알약 이미지 업로드 → 멀티에이전트 분석 → 감별 결과 + 안전정보 출력
"""

import streamlit as st
from PIL import Image

from graph import pill_app

# ── 페이지 설정 ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="💊 AI 알약 감별 & 안전정보 시스템",
    page_icon="💊",
    layout="wide",
)

st.title("💊 AI 알약 감별 & 안전정보 시스템")
st.caption("여러 각도의 알약 사진을 업로드하면 약물을 자동 감별하고 안전정보를 제공합니다.")

# ── 세션 상태 초기화 ──────────────────────────────────────────────────────────

if "result" not in st.session_state:
    st.session_state.result = None
if "confirmed_drug" not in st.session_state:
    st.session_state.confirmed_drug = None

# ── 사이드바: 입력 ────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📥 입력 정보")

    uploaded_files = st.file_uploader(
        "알약 사진 업로드 (여러 각도 가능, 최대 4장)",
        type=["jpg", "jpeg", "png", "JPG"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        cols = st.columns(min(len(uploaded_files), 4))
        for i, f in enumerate(uploaded_files[:4]):
            cols[i].image(f, caption=f"이미지 {i + 1}", use_container_width=True)

    st.divider()

    patient_type = st.selectbox(
        "환자 유형",
        ["일반", "노인", "임산부", "간 질환", "신 질환", "소아"],
    )

    current_meds_input = st.text_area(
        "현재 복용 중인 약물 (줄 바꿈 또는 쉼표로 구분)",
        placeholder="예: 아스피린\n메트포르민\n암로디핀",
        height=100,
    )

    st.divider()
    st.caption("🔬 연구 설정 (Ablation)")
    ablation_mode = st.radio(
        "감별 모드",
        options=["auto", "oracle", "image_only"],
        format_func=lambda x: {
            "auto":       "AUTO — MLLM+OCR 자동 각인 (본 연구)",
            "oracle":     "ORACLE — 메타데이터 각인 직접 입력 (성능 상한선)",
            "image_only": "IMAGE ONLY — 이미지만 (baseline)",
        }[x],
        index=0,
    )
    oracle_imprint = ""
    if ablation_mode == "oracle":
        oracle_imprint = st.text_input("각인 직접 입력", placeholder="예: L544")

    analyze_btn = st.button(
        "🔍 알약 분석 시작",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

# ── 메인: 분석 실행 ───────────────────────────────────────────────────────────

if analyze_btn and uploaded_files:
    images = [f.read() for f in uploaded_files[:4]]

    current_medications = [
        m.strip()
        for line in current_meds_input.replace(",", "\n").splitlines()
        for m in [line.strip()]
        if m
    ]

    with st.spinner("🤖 멀티에이전트 분석 중..."):
        initial_state = {
            "images": images,
            "patient_type": "" if patient_type == "일반" else patient_type,
            "current_medications": current_medications,
            "ablation_mode": ablation_mode,
        }
        # ORACLE 모드: 사용자 입력 각인을 Agent1 결과로 직접 주입
        if ablation_mode == "oracle" and oracle_imprint:
            initial_state["imprint_text"] = oracle_imprint
            initial_state["imprint_confidence"] = 1.0
        result = pill_app.invoke(initial_state)
    st.session_state.result = result

# ── 메인: 결과 출력 ───────────────────────────────────────────────────────────

result = st.session_state.result

if result:
    st.divider()

    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader("🔬 감별 결과")

        # 각인 정보
        imprint = result.get("imprint_text", "")
        imprint_conf = result.get("imprint_confidence", 0.0)
        if imprint:
            st.info(f"**인식된 각인:** {imprint}  \n신뢰도: {imprint_conf:.0%}")
        else:
            st.warning("각인을 인식하지 못했습니다.")

        if result.get("imprint_notes"):
            st.caption(f"특이사항: {result['imprint_notes']}")

        # Confidence Gating 시각화 (논문 Figure용)
        gate = result.get("gate_weight")
        if gate is not None:
            mode_label = {
                "oracle":     "ORACLE",
                "auto":       "AUTO",
                "image_only": "IMAGE ONLY",
            }.get(result.get("ablation_mode", "auto"), "AUTO")
            st.caption(f"모드: **{mode_label}**")
            st.progress(
                gate,
                text=f"각인 Branch 가중치 (Confidence Gate): {gate:.0%}"
                     + (" ← 각인 신뢰도 낮음, 이미지 우선" if gate < 0.4 else
                        " ← 각인·이미지 균형" if gate < 0.8 else
                        " ← 각인 신뢰도 높음"),
            )

        st.divider()

        # 감별 약물
        drug = result.get("drug_name", "알 수 없음")
        conf = result.get("confidence", 0.0)

        color = "🟢" if conf >= 0.85 else "🟡" if conf >= 0.5 else "🔴"
        st.metric(
            label="감별 약물",
            value=drug,
            delta=f"{color} 신뢰도 {conf:.0%}",
        )

        # 사용자 확인 필요
        if result.get("needs_confirmation") and result.get("top3_candidates"):
            st.warning("신뢰도가 낮습니다. 아래에서 올바른 약물을 선택해주세요.")
            candidates = [c["drug"] for c in result["top3_candidates"]]
            selected = st.radio("약물 선택", candidates, key="drug_selection")
            if st.button("선택 확인 후 안전정보 조회"):
                st.session_state.confirmed_drug = selected
                st.rerun()

        # 상위 후보
        if result.get("top3_candidates"):
            st.subheader("상위 후보")
            for i, c in enumerate(result["top3_candidates"]):
                st.progress(
                    min(c["confidence"], 1.0),
                    text=f"{i + 1}. {c['drug']}  ({c['confidence']:.0%})",
                )

        # 오류 메시지
        if result.get("error"):
            st.error(result["error"])

    with col_right:
        st.subheader("💊 안전정보")
        safety_info = result.get("safety_info", "")

        if st.session_state.confirmed_drug:
            # 사용자가 직접 선택한 약물로 안전정보 재조회
            from agents.safety_agent import get_safety_info
            with st.spinner("안전정보 재조회 중..."):
                safety_info = get_safety_info(
                    drug_name=st.session_state.confirmed_drug,
                    patient_type=result.get("patient_type", ""),
                    current_medications=result.get("current_medications", []),
                )
            st.session_state.confirmed_drug = None

        if safety_info:
            st.markdown(safety_info)
        else:
            st.info("안전정보를 불러오는 중입니다.")

        st.divider()
        st.caption("⚠️ 본 정보는 참고용이며, 정확한 복약 정보는 담당 의사 또는 약사에게 확인하세요.")

# ── 채팅 인터페이스: 추가 질문 ───────────────────────────────────────────────

if result and result.get("drug_name"):
    st.divider()
    st.subheader("💬 추가 질문")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input(f"{result['drug_name']}에 대해 더 궁금한 점을 질문하세요"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        from agents.safety_agent import get_safety_info
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                answer = get_safety_info(
                    drug_name=result["drug_name"],
                    patient_type=result.get("patient_type", ""),
                    current_medications=result.get("current_medications", []),
                )
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
