#########################################################
FROM python:3.11-slim as builder

RUN python -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

#########################################################
FROM python:3.11-slim as runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

WORKDIR /app

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

COPY --from=builder /opt/venv /opt/venv

RUN chown -R appuser:appgroup /app

ENV PATH="/opt/venv/bin:$PATH"

COPY . .

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

ENTRYPOINT ["python", "-m"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "$PORT"]