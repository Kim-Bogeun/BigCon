import os
import subprocess
import streamlit as st
import time
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio

API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY

client = MultiServerMCPClient(
    {
        "firebase": {
            "url": "http://0.0.0.0:8000/sse",
            "transport": "sse",
        },
    }
)

# tools 가져오기
tools = asyncio.run(client.get_tools())  # list of all tools


# 에이전트 생성
chat = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    temperature=0.1
)

# 에이전트 생성
agent = initialize_agent(
    tools=tools,
    llm=chat,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True
)

async def async_agent_run(prompt):
    return await agent.arun(prompt)


def run_multiple_instructions(instructions: list[str], mode: str = "sequential") -> list[str]:
    """
    mode:
      - "sequential": run agent on each instruction in order, each run gets previous outputs as context
      - "combined": run agent once with all instructions concatenated (sections)
    Returns list of outputs (one per instruction in sequential, single-item list for combined).
    """
    outputs: list[str] = []
    if mode == "combined":
        combined = "\n\n".join(f"Instruction {i+1}:\n{ins}" for i, ins in enumerate(instructions) if ins.strip())
        if not combined.strip():
            return []
        out = asyncio.run(async_agent_run(combined))
        return [out]
    # sequential
    context = ""
    for i, ins in enumerate(instructions):
        if not ins or not ins.strip():
            outputs.append("")  # keep positional mapping
            continue
        prompt = f"Instruction {i+1}:\n{ins}\n\nContext so far:\n{context}"
        out = asyncio.run(async_agent_run(prompt))
        outputs.append(out)
        # append latest output to context for next iteration
        context += f"\n--- Output {i+1} ---\n{out}\n"
    return outputs


# Streamlit UI
st.title("BigCon Demo")

st.markdown("가맹점명을 입력하고, 세 개의 instruction을 확인/수정한 뒤 실행하세요.")
franchise_name = st.text_input("가맹점명", value="")

DEFAULT_INSTR1 = """너는 최고의 마케팅 방법을 자동으로 추천하는 AI비서야.
실제 점주가 바로 쓸 수 있는 서비스 아이디어 제안해야해.

[목표]
- 가맹점별 특징/고객층/상권에 맞는 홍보, 이벤트, 할인, SNS 활용 꿀팁 제공
- 전략의 근거와 함께, 어떻게 실행할 수 있는지도 구체적으로 설명

[예시1]
문의하신 가맹점은 유동인구가 많은 성수에 위치해 있습니다. 제시된 표처럼 이미 20대 남녀 고객의 비중이 높지만, 동일 지역에서의 동일 업종에 비해 낮은 수준입니다. 매출을 늘리기 위해서는 매출에 직접적인 원인이 되는 변수인 객단가를 높이는 것이 좋습니다. 따라서 첫 방문 고객이 많은 특성상 가성비보다는 객단가를 높이는 것이 매출에 도움이 될 수 있습니다. 
이런 전략은 어떨까요? 유니크하고 술을 곁들일 수 있는 메뉴(마라양꼬치)를 개발하여 SNS에 바이럴함으로써, 처음 오는 고객들의 눈길을 끌고 객단가를 높일 수 있습니다.
"""
DEFAULT_INSTR2 = """[사용할 수 있는 도구]
1. search_franchise_by_name: 이름으로 가맹점 검색
"""
DEFAULT_INSTR3 = f"""매장에서 현재 재방문률을 높일 수 있는 마케팅 아이디어와 근거를 제시해줘.
가맹점명 : {franchise_name}
"""


instr1 = st.text_area("Instruction 1", value=DEFAULT_INSTR1, height=120)
instr2 = st.text_area("Instruction 2", value=DEFAULT_INSTR2, height=120)
instr3 = st.text_area("Instruction 3", value=DEFAULT_INSTR3, height=120)

if st.button("실행"):
    instrs = [instr1, instr2, instr3]
    if not franchise_name.strip():
        st.warning("가맹점명을 입력하세요.")
    elif not any(i.strip() for i in instrs):
        st.warning("최소 하나의 instruction을 입력하세요.")
    else:
        # prepare combined prompt, inject franchise name where useful
        combined = "\n\n".join(
            f"Instruction {i+1}:\n{ins}" for i, ins in enumerate(instrs) if ins.strip()
        )
        combined = f"Target franchise (MCT_NM): {franchise_name}\n\n" + combined
        with st.spinner("처리 중..."):
            result = asyncio.run(async_agent_run(combined))
        st.subheader("Combined result")
        st.write(result)