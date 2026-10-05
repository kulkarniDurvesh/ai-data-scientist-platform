# AI Data Scientist Platform: one image for the API and the dashboard.
#   docker build -t ai-data-scientist .
#   docker run -p 8000:8000 ai-data-scientist                          # API (default)
#   docker run -p 8050:8050 ai-data-scientist python app.py --host 0.0.0.0 --port 8050

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY core ./core
COPY visualization ./visualization
COPY service ./service
COPY ui ./ui
COPY api ./api
COPY domains ./domains
COPY app.py .

# Run as a non-root user; saved models go to /app/models (mount a volume to keep them).
RUN useradd --create-home appuser && mkdir -p /app/models && chown -R appuser /app
USER appuser

EXPOSE 8000 8050

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
