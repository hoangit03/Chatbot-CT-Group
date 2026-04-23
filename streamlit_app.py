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
    try:
        # build multipart đúng chuẩn cho FastAPI List[UploadFile]
        multipart_files = [
            ("files", (file.name, file, file.type))
            for file in files
        ]

        # log trước khi gửi
        for file in files:
            log_upload(file.name, "⏳ Uploading")

        res = requests.post(
            f"{BASE_URL}/extract",
            files=multipart_files,
            timeout=300
        )

        if res.status_code == 200:
            data = res.json()

            for result in data.get("results", []):
                file_name = result["file_name"]
                status = result["status"]
                message = result.get("message", "")

                if status == "queued":
                    log_upload(file_name, "📥 Queued", message)
                else:
                    log_upload(file_name, "❌ Failed", message)
        else:
            st.sidebar.error(res.text)

    except Exception as e:
        st.sidebar.error(str(e))


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
    """Gọi API chat (blocking - backup)"""
    try:
        res = requests.post(
            f"{BASE_URL_CHAT}/chat",
            json={
                "query": prompt,
                "chat_history": st.session_state.messages
            },
            timeout=12000
        )

        if res.status_code == 200:
            return res.json().get("answer", "Không có câu trả lời")
        return f"Lỗi API: {res.text}"

    except Exception as e:
        return f"Lỗi hệ thống: {e}"


def stream_chat_api(prompt):
    """Generator: Gọi API chat/stream, yield từng chunk text cho Streamlit."""
    try:
        res = requests.post(
            f"{BASE_URL_CHAT}/chat/stream",
            json={
                "query": prompt,
                "chat_history": st.session_state.messages
            },
            stream=True,
            timeout=12000
        )

        if res.status_code == 200:
            for chunk in res.iter_content(chunk_size=None, decode_unicode=True):
                if chunk:
                    yield chunk
        else:
            yield f"Lỗi API: {res.text}"

    except Exception as e:
        yield f"Lỗi hệ thống: {e}"

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
# CHAT INPUT (STREAMING)
# =========================
if prompt := st.chat_input("Nhập câu hỏi..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # 2 vùng hiển thị: thinking (tạm) + content (chính)
        thinking_container = st.empty()
        content_container = st.empty()

        thinking_lines = []
        full_answer = ""
        is_thinking = True

        for chunk in stream_chat_api(prompt):
            if chunk == "<!-- clear_thinking -->":
                # Xóa thinking, chuyển sang hiển thị content
                is_thinking = False
                thinking_container.empty()
                continue

            if chunk.startswith("<!-- thinking -->"):
                # Hiện thinking step
                step_text = chunk.replace("<!-- thinking -->", "").strip()
                thinking_lines.append(step_text)
                thinking_md = "\n\n".join(f"*{line}*" for line in thinking_lines)
                thinking_container.markdown(thinking_md)
            else:
                # Stream content thật
                full_answer += chunk
                content_container.markdown(full_answer + "▌")

        # Render final (bỏ cursor ▌)
        if full_answer:
            content_container.markdown(full_answer)
        
        answer = full_answer if full_answer else "Không có câu trả lời"

    st.session_state.messages.append({"role": "assistant", "content": answer})