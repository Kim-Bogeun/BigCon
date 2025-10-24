import os
import asyncio
import json
import pathlib
import time
import random
import pandas as pd
from typing import List, Dict, Optional

import streamlit as st
from PIL import Image
from langchain.agents import initialize_agent, AgentType
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from firebase_init import get_db_ref

API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY

MCP_CONFIG = {
    "firebase": {
        "url": "https://bigcon.onrender.com/sse",
        "transport": "sse",
        "headers": {"Accept": "text/event-stream"},
    }
}

LLM_CONFIG = {
    "model": "gemini-2.5-flash",
    "google_api_key": API_KEY,
    "temperature": 0.1
}

UI_CONFIG = {
    "page_title": "Is-This-Right?",
    "layout": "wide",
    "max_image_width": 30,
    "max_image_height": 20
}


def run_coro_sync(coro):
    """비동기 코루틴을 동기적으로 실행하는 헬퍼 함수"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

client = MultiServerMCPClient(MCP_CONFIG)
tools = run_coro_sync(client.get_tools())
chat = ChatGoogleGenerativeAI(**LLM_CONFIG)

agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

async def async_agent_run(prompt: str) -> str:
    """에이전트를 비동기적으로 실행"""
    return await agent.arun(prompt)


def run_multiple_instructions(instructions: List[str], mode: str = "sequential") -> List[str]:
    """
    여러 지시사항을 실행하는 함수
    
    Args:
        instructions: 실행할 지시사항 리스트
        mode: 실행 모드 ("sequential" 또는 "combined")
        
    Returns:
        실행 결과 리스트
    """
    outputs: List[str] = []
    
    if mode == "combined":
        combined = "\n\n".join(
            f"Instruction {i+1}:\n{ins}" 
            for i, ins in enumerate(instructions) 
            if ins.strip()
        )
        if not combined.strip():
            return []
        out = asyncio.run(async_agent_run(combined))
        return [out]
    
    context = ""
    for i, ins in enumerate(instructions):
        if not ins or not ins.strip():
            outputs.append("")
            continue
        prompt = f"Instruction {i+1}:\n{ins}\n\nContext so far:\n{context}"
        out = asyncio.run(async_agent_run(prompt))
        outputs.append(out)
        context += f"\n--- Output {i+1} ---\n{out}\n"
    
    return outputs


def load_instructions_file(fname: str = "instructions.json") -> Dict:
    """JSON 파일에서 지시사항을 로드하는 함수"""
    base = pathlib.Path(__file__).parent
    p = base / fname
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def get_franchise_data(franchise_name: str) -> Optional[Dict]:
    """Firebase에서 가맹점 데이터를 가져오는 함수"""
    try:
        ref = get_db_ref("/신한은행_데이터")
        return ref.child(franchise_name).get()
    except Exception as e:
        st.error(f"Firebase 조회 중 오류: {e}")
        return None


def get_instruction_by_business_type(biz: str, rare: int, instr_from_file: Dict) -> str:
    """업종과 재방문 고객 비중에 따라 적절한 지시사항을 선택하는 함수"""
    if biz == "카페":
        return instr_from_file.get("instr1", "")
    elif 0 <= rare <= 30:
        return instr_from_file.get("instr2", "")
    else:
        return instr_from_file.get("instr3", "")


def create_causal_instruction_from_data(cluster_causal_df: pd.DataFrame) -> str:
    """클러스터 인과관계 데이터를 기반으로 지시사항을 생성하는 함수"""
    if cluster_causal_df.empty:
        return ""
    
    causal_instructions = []
    
    for _, row in cluster_causal_df.iterrows():
        causal_path = row.get("인과경로", "")
        causal_interpretation = row.get("인과적 해석", "")
        
        if causal_path and causal_interpretation:
            instruction = f"""
            인과경로: {causal_path}
            인과적 해석: {causal_interpretation}
            """
            causal_instructions.append(instruction.strip())
    
    return "\n\n".join(causal_instructions)


def get_random_image_from_folder(folder_path: str) -> str:
    """폴더에서 랜덤으로 PNG 이미지를 선택하는 함수"""
    if not os.path.exists(folder_path):
        return "image.png"  
    
    png_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.png')]
    
    if not png_files:
        return "image.png"  
    
    selected_file = random.choice(png_files)
    return os.path.join(folder_path, selected_file)


INSTR_FROM_FILE = load_instructions_file()

st.set_page_config(
    page_title=UI_CONFIG["page_title"], 
    layout=UI_CONFIG["layout"]
)
st.markdown(
    f"""
    <style>
    .app-title {{ 
        font-size: 28px; 
        font-weight: 700; 
        margin-bottom: 6px; 
        color: #1f2937;
    }}
    .muted {{ 
        color: #6c757d; 
        margin-bottom: 16px; 
        font-size: 14px;
    }}
    .card {{ 
        background: #ffffff; 
        border-radius: 8px; 
        padding: 16px; 
        box-shadow: 0 4px 14px rgba(31, 41, 55, 0.06);
        border: 1px solid #e5e7eb;
    }}
    .divider {{ 
        border-left: 1px solid #e6e6e6; 
        height: 100%; 
        margin: 0 20px; 
    }}
    .image-container {{
        max-width: {UI_CONFIG['max_image_width']}px;
        max-height: {UI_CONFIG['max_image_height']}px;
        margin: 0 auto;
        text-align: center;
    }}
    .image-container img {{
        max-width: 100%;
        max-height: 100%;
        object-fit: contain;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }}
    .run-btn button {{
        background-color: #4CAF50;
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 6px;
        padding: 0.6em 1em;
        transition: background-color 0.2s ease;
        width: 100%;
    }}
    .run-btn button:hover {{
        background-color: #45a049;
        color: white;
    }}
    .result-container {{
        background: #f8fafc;
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
        border-left: 4px solid #4CAF50;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


DEFAULT_INSTR1 = INSTR_FROM_FILE.get("common_instr", "")
DEFAULT_INSTR2 = INSTR_FROM_FILE.get("instr3", "")

left_col, right_col = st.columns([0.5, 1.5])

with left_col:
    image_placeholder = st.empty() 
    
    default_img = Image.open("image.png")
    image_placeholder.image(
        default_img,
        use_container_width=True,
        caption="지니야 도와줘!!!!"
    )

    col_input, col_btn = st.columns([4, 1])
    with col_input:
        franchise_name = st.text_input(
            "가맹점명 (MCT_NM)", 
            value="", 
            help="검색할 가맹점명을 입력하세요.",
            placeholder="예: 스타**"
        )
    with col_btn:
        st.markdown("<div class='run-btn'>", unsafe_allow_html=True)
        def _on_click_run():
            st.session_state["run_btn"] = True
        run_btn = st.button(
            "🚀 실행",
            key="run_combined",
            help="입력한 instruction으로 LLM을 실행합니다.",
            on_click=_on_click_run
        )
        st.markdown("</div>", unsafe_allow_html=True)


with right_col:
    info_container2 = st.empty()
    INFO_DEFAULT = ""
    info_container2.subheader(INFO_DEFAULT)
    
    result_container = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)

if run_btn:
    st.session_state["run_btn"] = True
    if not franchise_name.strip():
        st.warning("가맹점명을 입력하세요.")
    else:
        record = get_franchise_data(franchise_name)
        
        if record is None:
            st.info("Firebase에서 해당 가맹점 정보를 찾지 못했습니다. 기본 instruction으로 실행합니다.")
            selected_instr1 = DEFAULT_INSTR1
            selected_instr2 = DEFAULT_INSTR2
            cluster_causal_df = pd.read_excel(f"dataset/cluster_causal/0.xlsx")
            selected_instr_causal = create_causal_instruction_from_data(cluster_causal_df)
            selected_instr3 = INSTR_FROM_FILE.get("instr1-2")
        else:
            biz = record.get("업종", "")
            rare = record.get("재방문 고객 비중", 0)
            business = record.get("업종", "")
            delivery_rate = record.get("배달매출_비율", 0)
            cluster = int(record.get("cluster", 0))
            
            
            selected_instr1 = INSTR_FROM_FILE.get("common_instr", DEFAULT_INSTR1)
            selected_instr3 = INSTR_FROM_FILE.get("default_instr-2")
            if biz == "카페":
                selected_instr2 = INSTR_FROM_FILE.get("instr1") or instr2
                selected_instr3 = INSTR_FROM_FILE.get("instr1-2")
            elif business == "건강식품":
                selected_instr2 = INSTR_FROM_FILE.get("instr5") or instr2
                selected_instr3 = INSTR_FROM_FILE.get("instr5-2")
            elif rare <= 30 and rare >= 0:
                selected_instr2 = INSTR_FROM_FILE.get("instr2") or instr2
                selected_instr3 = INSTR_FROM_FILE.get("instr2-2")
            else:
                selected_instr2 = INSTR_FROM_FILE.get("instr3") or instr2
                selected_instr3 = INSTR_FROM_FILE.get("instr3-2")
            
            if not selected_instr2:
                selected_instr2 = DEFAULT_INSTR2
                
            st.session_state["current_cluster"] = cluster
            
            graph_folder = f"images/cluster{cluster}"
            image_file = get_random_image_from_folder(graph_folder)
            
            cluster_dict = {
                0: "기초, 전통 식자재형\n\n특징: 고령층 및 여성 중심 방문, 생필·식자재 중심 업종 \n\n업종: 건어물, 건강원, 농산물, 미곡상, 수산물, 식품 제조, 축산물",
                1: "여가, 미식 소비형\n\n특징: 20–30대 주요 방문 및 성별 균형, 카페·베이커리·와인바 등 감성소비 업종\n\n업종: 카페, 커피전문점, 베이커리, 와인바, 일식당, 양식, 마카롱 등",
                2: "실속 외식형\n\n특징: 전연령대 고루 방문, 남성 비중 다소 높음, 치킨, 맥주, 한식, 육류 등 회식, 외식 중심\n\n업종: 치킨, 호프/맥주, 한식-육류, 피자, 중식당, 분식, 포장마차 등",
                3: "건강 프리미엄형\n\n특징: 중장년 및 여성 중심 방문, 건강식·반찬·죽 등 웰빙 중심 업종\n\n업종: 건강식품, 반찬, 인삼제품, 청과물, 떡/한과 제조, 유제품, 한식-죽"
            }
            
            try:
                img = Image.open(image_file)
                image_placeholder.image(
                    img,
                    use_container_width=True,
                    caption=cluster_dict[cluster]
                )
            except Exception as e:
                st.warning(f"이미지를 불러올 수 없습니다: {e}")
            
            try:
                cluster_causal_df = pd.read_excel(f"dataset/cluster_causal/{cluster}.xlsx")
                selected_instr_causal = create_causal_instruction_from_data(cluster_causal_df)
            except Exception as e:
                st.warning(f"클러스터 인과관계 데이터를 불러올 수 없습니다: {e}")
                selected_instr_causal = ""
            
        instructions = [selected_instr1, selected_instr_causal, selected_instr2, selected_instr3]
        
        combined = "\n\n".join(
            f"Instruction {i+1}:\n{ins}" 
            for i, ins in enumerate(instructions) 
            if ins.strip()
        )
        combined = f"Target franchise (MCT_NM): {franchise_name}\n\n" + combined
        
        try:
            info_text = selected_instr2[:-3] if len(selected_instr2) > 3 else selected_instr2
            info_container2.markdown(f'<div class="muted">{info_text}</div>', unsafe_allow_html=True)
        except Exception:
            info_container2.text(selected_instr2)

        with st.spinner("처리 중..."):
            try:
                output = run_coro_sync(async_agent_run(combined))
                result_container.markdown(
                    f"<div style='font-size:17px; line-height:1.6; color:#111;'>{output}</div>",
                    unsafe_allow_html=True
                )
            except Exception as e:
                result_container.error(f"Agent 실행 중 오류: {e}")
                st.error(f"상세 오류: {str(e)}")