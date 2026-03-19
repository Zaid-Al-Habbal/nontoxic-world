# syntax=docker/dockerfile:1

# ===========================================================================
# BUILD STAGE
# Installs all dependencies. Build tools stay here and never reach production.
# ===========================================================================
FROM python:3.13-slim AS builder

# Copy uv binary from the official image — faster and cleaner than pip install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# uv settings:
#   UV_COMPILE_BYTECODE=1  : compile .py -> .pyc at install time for faster startup
#   UV_LINK_MODE=copy      : copy files instead of symlinking (required in Docker)
#   UV_PYTHON_DOWNLOADS=never : use the base image's Python, don't let uv download its own
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# System deps needed only at build time (compiling C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Option A — if you use pyproject.toml + uv.lock (recommended)
# Copy lock files first so this layer is cached independently of app code.
# Dependencies are only re-installed when pyproject.toml or uv.lock changes.
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

# ---------------------------------------------------------------------------
# Option B — if you prefer requirements.txt
# Comment out Option A above and uncomment these two lines instead:
# ---------------------------------------------------------------------------
# COPY requirements.txt .
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv pip install --system -r requirements.txt

# Copy app code and do final project install
COPY app ./app
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
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    HF_HOME=/app/.cache/huggingface \
    # Put the venv on PATH so uvicorn and all installed packages are found
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Runtime-only system deps:
#   curl     : HEALTHCHECK + huggingface_hub downloads
#   libgomp1 : OpenMP runtime — mandatory for PyTorch on Linux
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create cache directory for HuggingFace Hub before switching to non-root user
RUN mkdir -p /app/.cache/huggingface

# Non-root user for security
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

# Copy the installed venv and app code from the build stage
COPY --from=builder --chown=appuser:appgroup /app /app

# Set correct ownership for cache directory
RUN chown -R appuser:appgroup /app/.cache

USER appuser

EXPOSE 7860

# 60s start-period: the app downloads model checkpoints from HuggingFace on boot
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Single worker is intentional: multiple workers would load duplicate model
# copies into memory and compete for the GPU
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]