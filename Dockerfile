# AETHOS RAG API — slim, CPU-only, no-GPU-at-serve. Builds on both x86_64 and ARM64
# (Oracle Always-Free Ampere A1). No CUDA base image; the serve path is pure numpy.
FROM python:3.11-slim

# ---- hard no-GPU boundary at the process level (belt-and-suspenders with the app's own env guard) ----
ENV CUDA_VISIBLE_DEVICES="" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUTF8=1 \
    PIP_NO_CACHE_DIR=1 \
    AETHOS_INDEX_DIR=/data/aethos_index \
    PORT=8000

# numba (glass-box tier) needs a C toolchain for llvmlite wheels on some arches; slim gcc covers it.
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app + the measured modules it reuses (_dense_compress_*.py, aethos_rag_pipeline.py + its engine stack)
COPY . .

# persisted index lives on a mounted volume so a restart serves with no re-ingest
RUN mkdir -p /data/aethos_index
VOLUME ["/data"]

# non-root
RUN useradd -m -u 10001 aethos && chown -R aethos:aethos /app /data
USER aethos

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

# uvicorn entrypoint (single worker keeps the in-process index consistent; scale with a shared store if needed)
CMD ["uvicorn", "aethos_rag_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
