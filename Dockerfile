#########################################################
FROM python:3.11.10-slim AS builder

WORKDIR /build

# Install uv. Pinned: :latest let two builds of the same commit resolve
# different uv versions, which is the kind of drift uv.lock cannot cover.
COPY --from=ghcr.io/astral-sh/uv:0.12.1 /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Create virtual environment and install dependencies
RUN uv sync --frozen --no-dev

#########################################################
FROM python:3.11.10-slim AS runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appgroup && useradd -r -g appgroup appuser

COPY --from=builder --chown=appuser:appgroup /build/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"

# Ownership is set by each COPY. A `RUN chown -R /app` afterwards would work too,
# but it rewrites every file into a new layer -- 300 MB of venv duplicated.
COPY --chown=appuser:appgroup . .

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]