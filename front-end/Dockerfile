# syntax=docker/dockerfile:1

# ===========================================================================
# BUILD STAGE
# Installs all dependencies. Build tools stay here and never reach production.
# ===========================================================================
FROM python:3.13-slim AS builder

# Copy uv binary from the official image — faster and cleaner than pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# uv settings:
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# System deps needed only at build time (compiling C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# ===========================================================================
# RUNTIME STAGE
# Clean image with only what is needed to run the app.
# No uv, no compiler, no build tools.
# ===========================================================================
FROM python:3.13-slim AS runtime

# Runtime environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    # Put the venv on PATH so uvicorn and all installed packages are found
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy the installed venv and app code from the build stage
COPY --from=builder --chown=appuser:appgroup /app /app

USER appuser

EXPOSE 8501

HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app/app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]