FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user (uid 1000). Hugging Face Spaces recommends this and
# it is safer everywhere; all writable paths live under /app.
RUN useradd -m -u 1000 app

WORKDIR /app

# Install dependencies first so Docker can cache this layer independently
# of application code changes.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the rembg U2Net model into the image so the first request
# doesn't stall while the container fetches ~170 MB at runtime.
# U2NET_HOME is set to a stable app-owned path rather than ~/.u2net so the
# model works correctly regardless of which user the container runs as.
ENV U2NET_HOME=/app/models
RUN python -c "from rembg.sessions.u2net import U2netSession; U2netSession.download_models()"

COPY . /app

RUN mkdir -p /app/output && chown -R app:app /app

USER app

EXPOSE 8000
# Allow configurable worker count via UVICORN_WORKERS (default 2).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
