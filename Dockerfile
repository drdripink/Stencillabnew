# StencilLab — Python backend + static frontend in one image.
# Runs on Fly.io, Railway, Render, or any standard container host.

FROM python:3.12-slim

# System deps:
#   potrace    — Stage 5 vectorization
#   libgl1     — opencv runtime dep (even headless needs some libs)
#   libglib2.0-0 — opencv runtime dep
#   ca-certs   — HTTPS to model providers
RUN apt-get update && apt-get install -y --no-install-recommends \
    potrace \
    libgl1 \
    libglib2.0-0 \
    ca-certificates \
    curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY static/ ./static/

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD curl -fsS http://127.0.0.1:8080/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
