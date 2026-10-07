# syntax=docker/dockerfile:1

# ---- build: wheels for the app and every dependency -------------------------
FROM python:3.12-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /src
COPY pyproject.toml ./
COPY src ./src
RUN pip wheel --wheel-dir /wheels .

# ---- runtime ----------------------------------------------------------------
# Installed from wheels, not from the source tree, so the image exercises the
# same path as `pip install`: prompts, templates and static files must ship as
# package data or the container cannot run a single brief.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ANALYST_DATA_DIR=/data

RUN useradd --create-home --uid 10001 analyst \
    && mkdir -p /data \
    && chown analyst:analyst /data

COPY --from=build /wheels /wheels
RUN pip install --no-index --find-links /wheels autonomous-market-analyst \
    && rm -rf /wheels

USER analyst
WORKDIR /home/analyst
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["analyst", "serve", "--host", "0.0.0.0", "--port", "8000"]
