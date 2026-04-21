import streamlit as st
import requests

# =========================
# CONFIG
# =========================
BASE_URL_CHAT = "http://localhost:7999/api/v1"
BASE_URL = "http://localhost:8000/api/v1"

# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="Chatbot UI", layout="wide")

st.title("🤖 Chatbot Document QA")

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# SIDEBAR - UPLOAD FILE
# =========================
st.sidebar.header("Upload tài liệu")

uploaded_file = st.sidebar.file_uploader(
    "Chọn file (PDF, DOCX, Excel...)",
    type=["pdf", "docx", "xlsx", "doc", "xls", "msg", "pptx", "ppt"]
)

if st.sidebar.button("Upload & Extract"):
    if uploaded_file is not None:
        with st.spinner("Đang xử lý file..."):
            files = {
                "file": (uploaded_file.name, uploaded_file, uploaded_file.type)
            }

            try:
                res = requests.post(f"{BASE_URL}/extract", files=files)

                if res.status_code == 200:
                    st.sidebar.success("Extract thành công")
                else:
                    st.sidebar.error(f"Lỗi: {res.text}")

            except Exception as e:
                st.sidebar.error(f"Exception: {e}")
    else:
        st.sidebar.warning("Chưa chọn file")

# =========================
# MAIN - CHAT
# =========================
st.subheader("Chat")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input chat
if prompt := st.chat_input("Nhập câu hỏi..."):
    # add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # call API
    with st.chat_message("assistant"):
        with st.spinner("Đang trả lời..."):
            try:
                res = requests.post(
                    f"{BASE_URL_CHAT}/chat",
                    json={
                        "query": prompt,
                        "chat_history": st.session_state.messages
                    }
                )

                if res.status_code == 200:
                    answer = res.json().get("answer", "Không có câu trả lời")
                else:
                    answer = f"Lỗi API: {res.text}"

            except Exception as e:
                answer = f"Lỗi hệ thống: {e}"

            st.markdown(answer)

    # save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })