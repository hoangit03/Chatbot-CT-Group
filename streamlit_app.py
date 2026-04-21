import streamlit as st
import requests
from datetime import datetime

# =========================
# CONFIG
# =========================
BASE_URL_CHAT = "http://localhost:7999/api/v1"
BASE_URL = "http://localhost:8000/api/v1"

MAX_FILES = 10

st.set_page_config(page_title="Chatbot Document QA", layout="wide")

# =========================
# INIT SESSION STATE
# =========================
def init_session():
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("upload_logs", [])
    st.session_state.setdefault("is_uploading", False)
    st.session_state.setdefault("confirm_delete", False)
    st.session_state.setdefault("delete_notice", None)

init_session()

# =========================
# UI HEADER
# =========================
st.title("🤖 Chatbot Document QA")

# =========================
# LOG HELPER
# =========================
def log_upload(file_name, status, message=""):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.upload_logs.insert(
        0, f"[{timestamp}] {file_name} - {status} {message}"
    )

# =========================
# API FUNCTIONS
# =========================
def upload_and_extract(files):
    progress = st.sidebar.progress(0)
    total = len(files)

    for i, file in enumerate(files):
        log_upload(file.name, "⏳ Uploading")

        try:
            res = requests.post(
                f"{BASE_URL}/extract",
                files={"file": (file.name, file, file.type)},  # ✅ FIX KEY
                timeout=300
            )

            if res.status_code == 200:
                data = res.json()

                for result in data.get("results", []):
                    status = result["status"]
                    message = result.get("message", "")

                    if status == "queued":
                        log_upload(file.name, "📥 Queued", message)
                    else:
                        log_upload(file.name, "❌ Failed", message)

            else:
                log_upload(file.name, "❌ API Error", res.text)

        except Exception as e:
            log_upload(file.name, "❌ Exception", str(e))

        progress.progress((i + 1) / total)


def delete_vectordb():
    try:
        res = requests.delete(f"{BASE_URL}/vectordb/all", timeout=60)

        if res.status_code == 200:
            return True, "Đã xóa toàn bộ VectorDB"
        else:
            return False, res.text

    except Exception as e:
        return False, str(e)


def call_chat_api(prompt):
    try:
        res = requests.post(
            f"{BASE_URL_CHAT}/chat",
            json={
                "query": prompt,
                "chat_history": st.session_state.messages
            },
            timeout=120
        )

        if res.status_code == 200:
            return res.json().get("answer", "Không có câu trả lời")
        return f"Lỗi API: {res.text}"

    except Exception as e:
        return f"Lỗi hệ thống: {e}"

# =========================
# SIDEBAR - UPLOAD
# =========================
st.sidebar.header("📂 Upload tài liệu")

uploaded_files = st.sidebar.file_uploader(
    "Chọn nhiều file",
    type=["pdf", "docx", "xlsx", "doc", "xls", "msg", "pptx", "ppt"],
    accept_multiple_files=True
)

if st.sidebar.button("🚀 Upload & Extract", disabled=st.session_state.is_uploading):
    if not uploaded_files:
        st.sidebar.warning("Chưa chọn file")

    elif len(uploaded_files) > MAX_FILES:
        st.sidebar.error(f"Tối đa {MAX_FILES} file")

    else:
        st.session_state.is_uploading = True
        upload_and_extract(uploaded_files)
        st.session_state.is_uploading = False

        st.sidebar.info("📌 File đã vào queue. Vui lòng đợi xử lý trước khi hỏi.")

# =========================
# SIDEBAR - LOG
# =========================
st.sidebar.subheader("📊 Trạng thái")

if st.session_state.upload_logs:
    for log in st.session_state.upload_logs:
        st.sidebar.write(log)
else:
    st.sidebar.caption("Chưa có dữ liệu")

# =========================
# DELETE (DANGER ZONE)
# =========================
st.sidebar.divider()
st.sidebar.subheader("⚠️ Danger Zone")

if not st.session_state.confirm_delete:
    if st.sidebar.button("🗑️ Xóa toàn bộ dữ liệu"):
        st.session_state.confirm_delete = True
        st.rerun()
else:
    st.sidebar.error("Hành động này không thể hoàn tác")

    col1, col2 = st.sidebar.columns(2)

    if col1.button("✅ Xác nhận"):
        with st.spinner("Đang xóa..."):
            success, msg = delete_vectordb()

        if success:
            st.session_state.delete_notice = ("success", msg)
            st.session_state.messages = []
            st.session_state.upload_logs = []
        else:
            st.session_state.delete_notice = ("error", msg)

        st.session_state.confirm_delete = False
        st.rerun()

    if col2.button("❌ Hủy"):
        st.session_state.confirm_delete = False
        st.rerun()

# =========================
# GLOBAL NOTICE
# =========================
if st.session_state.delete_notice:
    status, msg = st.session_state.delete_notice

    if status == "success":
        st.success(f"🗑️ {msg}")
    else:
        st.error(f"❌ {msg}")

    st.session_state.delete_notice = None

# =========================
# CHAT UI
# =========================
st.subheader("💬 Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# CHAT INPUT
# =========================
if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang trả lời..."):
            answer = call_chat_api(prompt)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})