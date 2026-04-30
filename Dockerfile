# ============================================================
# Dockerfile — CT-Group Chatbot (3-Tier: App Layer ONLY)
# FROM ctgroup/python-base:1.0 — shared base image
# Chỉ COPY code, KHÔNG cài pip lại (~30s build)
# ============================================================

FROM ctgroup/python-base:1.0

WORKDIR /app

# --- Project-specific deps (chỉ lib RIÊNG của project này) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

# --- Copy application source ---
COPY app/ ./app/
COPY pipeline/ ./pipeline/
COPY streamlit_app.py .

ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# --- Shared data directories ---
RUN mkdir -p /app/shared_data/data_input /app/shared_data/data_output

# --- Supervisor config ---
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# --- Health check ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:7999/api/v1/health || exit 1

EXPOSE 7999 8001 8501 8003

CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
