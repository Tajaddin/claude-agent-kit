FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
COPY benchmarks ./benchmarks
COPY README.md LICENSE ./

RUN pip install --no-cache-dir -e . \
    && useradd --create-home --uid 10001 app \
    && chown -R app benchmarks/results

USER app

HEALTHCHECK --interval=30s --timeout=15s --start-period=5s --retries=2 \
    CMD python -c "from claude_agent_kit import Agent; print('ok')" || exit 1

ENTRYPOINT ["python", "-m", "claude_agent_kit.cli"]
CMD ["--help"]
