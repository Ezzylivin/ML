# NeoV6 engine — containerized FastAPI/uvicorn service (replaces the systemd unit).
# Build context is /root/Project/ML. Code + state are bind-mounted at runtime
# (see docker-compose.yml) so you still edit files in place and just restart.
FROM python:3.12-slim

# Build tools some wheels (numpy/pandas/pandas_ta/ccxt) may need on slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (layer-cached unless requirements.txt changes).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir "uvicorn[standard]"

# Copy the source (overridden by the bind mount in compose, but lets the image
# run standalone too).
COPY . .

EXPOSE 8000

# Lightweight healthcheck hits the engine's own /api/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

# FastAPI app object is `app` in main4.py.
CMD ["python", "-m", "uvicorn", "main4:app", "--host", "0.0.0.0", "--port", "8000"]
