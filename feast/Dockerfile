FROM python:3.9-slim

RUN apt-get update && apt-get install -y curl gcc g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir \
    feast==0.34.1 \
    redis \
    minio \
    pandas \
    numpy \
    pyarrow

RUN useradd -m -u 1000 feast && chown -R feast:feast /app

USER feast

RUN mkdir -p /app/feast /app/data

COPY --chown=feast:feast entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

EXPOSE 6565 6566

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:6565/health || exit 1

ENTRYPOINT ["/app/entrypoint.sh"]
