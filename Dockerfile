#########################################################
FROM python:3.11-slim as builder

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock* ./

# Create virtual environment and install dependencies
RUN uv sync --frozen --no-dev

#########################################################
FROM python:3.11-slim as runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT=8000

WORKDIR /app

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

COPY --from=builder /.venv /.venv

RUN chown -R appuser:appgroup /app

ENV PATH="/.venv/bin:$PATH"

COPY . .

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

EXPOSE 8000

ENTRYPOINT ["python", "-m"]

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "$PORT"]