# ============================================================
# Dockerfile — CT-Group Chatbot (3-Tier: App Layer ONLY)
# NO GPU — gọi Core AI services qua HTTP API
# Base: python:3.11-slim (~150MB thay vì ~15GB)
# ============================================================

FROM python:3.11-slim

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl git supervisor libreoffice \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (CPU-only, nhẹ) ---
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

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
