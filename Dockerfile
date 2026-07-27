FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

COPY requirements.runtime.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.runtime.txt

COPY api.py claims.py ./
COPY index ./index

EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn api:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
