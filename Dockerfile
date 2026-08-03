FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir .

COPY configs ./configs
COPY data ./data
COPY docs ./docs
COPY reports ./reports
COPY sql ./sql
COPY Makefile ./Makefile

ENTRYPOINT ["vol-platform"]
CMD ["--help"]
