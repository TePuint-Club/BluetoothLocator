# syntax=docker/dockerfile:1.6

# ============== Builder: resolve and install deps with uv ==============
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS builder
WORKDIR /app

# Ensure deterministic copies and no symlinks inside venv for easier relocation
ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy project (including pyproject & sources) so uv can locate the root package
COPY . .

# Sanity check: ensure root manifests and package directory exist in context
RUN test -f pyproject.toml && test -f uv.lock && test -d ble_locator_server && echo "Project files present"

# Create project venv and install dependencies (and project in editable mode)
RUN --mount=type=cache,target=/root/.cache \
    uv sync --frozen --no-dev


# ============== Runtime: offline run with prebuilt venv ==============
FROM ghcr.io/astral-sh/uv:python3.12-bookworm AS runtime
WORKDIR /app

ENV UV_LINK_MODE=copy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy prebuilt virtual environment and source tree from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app /app

# Make venv default in PATH (uv will also auto-detect .venv in project root)
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Expose nothing by default; configure at run-time if needed

# Default command: use uv run to invoke CLI
CMD ["uv", "run", "ble-locator-server", "run"]
