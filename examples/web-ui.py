import torch
import streamlit as st
from chatglm_q.decoder import ChatGLMDecoder, chat_template
from torch.utils.mobile_optimizer import optimize_for_mobile
from typing import Tuple, List
# page state

@st.cache_resource
def create_model():
    device = torch.device("cpu")
    torch_dtype = torch.float
    decoder = ChatGLMDecoder.from_pretrained("/home/shanlin/pytorch-vulkan/weight_chatglm-q", torch_dtype=torch.float)
    model = decoder.model
    model.eval()
    script_model = torch.jit.script(model)
    cpu_script_model = optimize_for_mobile(script_model, backend='cpu')
    decoder.model = cpu_script_model
    # decoder.time_log = True # log generation performance
    return decoder

with st.spinner("加载模型中..."):
    model = create_model()


if "history" not in st.session_state:
    st.session_state["history"] = []


# parameters

with st.sidebar:
    st.markdown("## 采样参数")

    max_tokens = st.number_input("max_tokens", min_value=1, max_value=2000, value=800)
    temperature = st.number_input("temperature", min_value=0.1, max_value=4.0, value=1.0)
    top_p = st.number_input("top_p", min_value=0.1, max_value=1.0, value=0.8)
    top_k = st.number_input("top_k", min_value=1, max_value=100, value=50)

    if st.button("清空上下文"):
        st.session_state.history = []

    st.markdown("""
    [ChatGLM2](https://huggingface.co/THUDM/chatglm2-6b)

    [chatglm-q](https://github.com/K024/chatglm-q)
    """)


# main body

st.markdown("## ChatGLM2")

history: List[Tuple[str, str]] = st.session_state.history

if len(history) == 0:
    st.caption("请在下方输入消息开始会话")


for idx, (question, answer) in enumerate(history):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        st.write(answer)

question = st.chat_input("消息", key="message")

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        empty = st.empty()
        with st.spinner("正在回复中"):
            prompt = chat_template(history, question)
            for answer in model.generate(
                prompt,
                max_generated_tokens=max_tokens,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            ):
                empty.write(answer)

    st.session_state.history = history + [(question, answer)]
