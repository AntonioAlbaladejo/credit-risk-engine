#########################################################
FROM python:3.11.10-slim AS builder

WORKDIR /build

# Install uv. Pinned: :latest let two builds of the same commit resolve
# different uv versions, which is the kind of drift uv.lock cannot cover.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies. The genai group is not
# optional here: the service searches the regulatory corpus, so fastembed ships.
RUN uv sync --frozen --no-dev --group genai

# Bake the embedding model into the image. fastembed fetches it from HuggingFace
# the first time it is instantiated, and a fetch on the startup path would leave
# the task down whenever that host is unreachable. Only config.py is copied, so
# this layer survives ordinary source changes and rebuilds exactly when the
# model name moves -- which is when the weights have to move with it.
COPY src/__init__.py src/config.py src/
ENV FASTEMBED_CACHE_PATH=/build/.fastembed
RUN .venv/bin/python -c "from src.config import EMBEDDING_MODEL; \
    from fastembed import TextEmbedding; TextEmbedding(model_name=EMBEDDING_MODEL)"

#########################################################
FROM python:3.11.10-slim AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

COPY --from=builder --chown=appuser:appgroup /build/.venv /app/.venv
COPY --from=builder --chown=appuser:appgroup /build/.fastembed /app/.fastembed

# Points fastembed at the baked weights. Without it the default is a temp
# directory, empty in a fresh container, and the download would happen anyway.
ENV FASTEMBED_CACHE_PATH=/app/.fastembed

ENV PATH="/app/.venv/bin:$PATH"

# Ownership is set by each COPY. A `RUN chown -R /app` afterwards would work too,
# but it rewrites every file into a new layer -- 300 MB of venv duplicated.
COPY --chown=appuser:appgroup . .

USER appuser

# start-period covers the slowest observed cold start: 13 s on 0.5 vCPU, which is
# how the Fargate task is sized. This governs `docker run`; ECS reads the
# healthCheck block of the task definition instead, and that lives in AWS.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]