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
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "upload_logs" not in st.session_state:
        st.session_state.upload_logs = []

    if "is_uploading" not in st.session_state:
        st.session_state.is_uploading = False
        
    if "confirm_delete" not in st.session_state:
        st.session_state.confirm_delete = False
    if "delete_notice" not in st.session_state:
        st.session_state.delete_notice = None

init_session()

# =========================
# UI HEADER
# =========================
st.title("🤖 Chatbot Document QA")

# =========================
# SIDEBAR - FILE UPLOAD
# =========================
st.sidebar.header("📂 Upload tài liệu")

uploaded_files = st.sidebar.file_uploader(
    "Chọn nhiều file (PDF, DOCX, Excel...)",
    type=["pdf", "docx", "xlsx", "doc", "xls", "msg", "pptx", "ppt"],
    accept_multiple_files=True
)

def log_upload(file_name, status, message=""):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.upload_logs.insert(
        0, f"[{timestamp}] {file_name} - {status} {message}"
    )

def upload_and_extract(files):
    for file in files:
        log_upload(file.name, "⏳ Processing")

        try:
            res = requests.post(
                f"{BASE_URL}/extract",
                files={"file": (file.name, file, file.type)},
                timeout=300
            )

            if res.status_code == 200:
                log_upload(file.name, "✅ Success")
            else:
                log_upload(file.name, "❌ Failed", res.text)

        except Exception as e:
            log_upload(file.name, "❌ Exception", str(e))


def delete_vectordb():
    try:
        res = requests.delete(
            f"{BASE_URL}/vectordb/all",
            timeout=60
        )

        if res.status_code == 200:
            return True, "Đã xóa toàn bộ VectorDB"
        else:
            return False, res.text

    except Exception as e:
        return False, str(e)


# Upload button
if st.sidebar.button("🚀 Upload & Extract", disabled=st.session_state.is_uploading):
    if not uploaded_files:
        st.sidebar.warning("Chưa chọn file")

    elif len(uploaded_files) > MAX_FILES:
        st.sidebar.error(f"Chỉ được upload tối đa {MAX_FILES} file mỗi lần")

    else:
        st.session_state.is_uploading = True
        upload_and_extract(uploaded_files)
        st.session_state.is_uploading = False

# =========================
# SIDEBAR - UPLOAD LOGS
# =========================
st.sidebar.subheader("📊 Trạng thái xử lý")

if st.session_state.upload_logs:
    for log in st.session_state.upload_logs:
        st.sidebar.write(log)
else:
    st.sidebar.caption("Chưa có file nào được xử lý")

# =========================
# DELETE VECTORDB (DANGER ZONE)
# =========================
st.sidebar.divider()
st.sidebar.subheader("⚠️ Danger Zone")

# STEP 1: click delete
if not st.session_state.confirm_delete:
    if st.sidebar.button("🗑️ Xóa toàn bộ dữ liệu"):
        st.session_state.confirm_delete = True
        st.rerun()   # 🔥 FIX delay

# STEP 2: confirm UI
else:
    st.sidebar.warning("Bạn chắc chắn muốn xóa toàn bộ dữ liệu?")

    col1, col2 = st.sidebar.columns(2)

    if col1.button("✅ Xác nhận"):
        with st.spinner("Đang xóa dữ liệu..."):
            success, msg = delete_vectordb()

        if success:
            # lưu notice thay vì show trực tiếp
            st.session_state.delete_notice = ("success", msg)

            # reset data
            st.session_state.messages = []
            st.session_state.upload_logs = []
        else:
            st.session_state.delete_notice = ("error", msg)

        st.session_state.confirm_delete = False
        st.rerun()

    if col2.button("❌ Hủy"):
        st.session_state.confirm_delete = False
        st.rerun()   # 🔥 đảm bảo UI reset ngay
        
# =========================
# GLOBAL NOTIFICATION
# =========================
if st.session_state.delete_notice:
    status, msg = st.session_state.delete_notice

    if status == "success":
        st.success(f"🗑️ {msg}")
    else:
        st.error(f"❌ {msg}")

    # clear sau khi hiển thị 1 lần
    st.session_state.delete_notice = None

# =========================
# MAIN - CHAT UI
# =========================
st.subheader("💬 Chat")

# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# CHAT FUNCTION
# =========================
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
        else:
            return f"Lỗi API: {res.text}"

    except Exception as e:
        return f"Lỗi hệ thống: {e}"
    
# =========================
# CHAT INPUT
# =========================
if prompt := st.chat_input("Nhập câu hỏi..."):
    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Đang trả lời..."):
            answer = call_chat_api(prompt)
            st.markdown(answer)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })