#########################################################
FROM python:3.11.10-slim AS builder

WORKDIR /build

# Install uv CLI for managing virtual environments and dependency sync
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency manifests before installing packages so the layer can be cached
COPY pyproject.toml uv.lock ./

# Install compile-time dependencies and clear apt lists
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create the virtual environment and install runtime dependencies only
RUN uv sync --frozen --no-dev

# Strip unnecessary Python bytecode and library symbols to reduce image size
RUN find /build/.venv -type d -name "__pycache__" -exec rm -rf {} + \
    && find /build/.venv -type f -name "*.pyc" -delete \
    && find /build/.venv -type f -name "*.pyo" -delete \
    && find /build/.venv -type f -name "*.so" -exec strip --strip-unneeded {} \; 2>/dev/null || true

# --- Runtime Stage: Use the same libc-compatible base image as the builder stage ---
FROM python:3.11.10-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install minimal runtime packages, including curl for health checks and passwd for useradd/groupadd
RUN apt-get update && apt-get install -y --no-install-recommends curl passwd \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and group for running the app
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Copy the prepared virtual environment from the builder stage
COPY --from=builder /build/.venv /app/.venv

# Copy only application source and model artifacts needed at runtime
COPY src ./src
COPY models ./models

# Set ownership of application files before switching to non-root user
RUN chown -R appuser:appgroup /app

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]