# Deploying the AETHOS RAG API on Oracle Cloud Always-Free (no GPU)

This service runs entirely on CPU. The **only** place a model is ever touched is embedding the *query*,
and even that is optional (send precomputed query vectors). The index build, storage, and every search are
pure-numpy — nothing at serve can allocate a GPU. That makes it a clean fit for Oracle's **Always-Free** tier.

Two free-tier options, both covered below:

| Option | Shape | Free allowance | Best for |
|---|---|---|---|
| **A. Ampere A1 VM** (recommended) | `VM.Standard.A1.Flex` (ARM64) | up to **4 OCPU / 24 GB RAM**, always free | a persistent demo + API you SSH into |
| **B. Container Instances** | 1 OCPU / a few GB | free monthly allotment | fire-and-forget container, no VM to manage |

Both are ARM64. The `Dockerfile` builds on ARM64 and x86_64 (uses `python:3.11-slim`, no CUDA base).

---

## 0. Build and run locally first (sanity)

```bash
# from the repo root (C:/Users/wynos/New folder (3))
docker build -t aethos-rag-api .
docker run --rm -p 8000:8000 -v aethos_data:/data aethos-rag-api
# -> serves on http://localhost:8000 ; GET /healthz should return {"status":"ok","gpu_at_serve":false}
```

Smoke the endpoints (precomputed-embedding path, zero model at serve):

```bash
# ingest 2 docs with 4-dim embeddings (use your real 1024-d bge vectors in production)
curl -s localhost:8000/ingest -H 'content-type: application/json' -d '{
  "docs":[{"doc_id":"a","text":"neural network training","embedding":[0.1,0.2,0.3,0.4]},
          {"doc_id":"b","text":"cardiac disease treatment","embedding":[0.4,0.3,0.2,0.1]}],
  "M":4,"Kp":16}'
curl -s localhost:8000/search  -H 'content-type: application/json' -d '{"query_embedding":[0.1,0.2,0.3,0.4],"k":2}'
curl -s localhost:8000/explain -H 'content-type: application/json' -d '{"query":"neural network training","doc_id":"a"}'
curl -s localhost:8000/stats
```

`aethos_rag_api.py` also has a no-network self-test: `python aethos_rag_api.py` (synthetic embeddings) and
`python aethos_rag_api.py serve` (starts uvicorn).

---

## A. Ampere A1 VM (recommended)

1. **Create the instance.** OCI Console → Compute → Instances → Create.
   - Image: **Oracle Linux 8/9** or **Ubuntu 22.04** (ARM64).
   - Shape: **VM.Standard.A1.Flex**, e.g. 2 OCPU / 12 GB (all within the 4 OCPU / 24 GB always-free cap).
   - Add your SSH key. Open port 8000 (or put it behind a reverse proxy — see §D).

2. **Open the port.** VCN → the instance's subnet → Security List → add an **ingress** rule:
   `Source 0.0.0.0/0, TCP, dest port 8000` (tighten to your IP for a private demo).

3. **Install Docker and run.**
   ```bash
   ssh opc@<public-ip>            # 'ubuntu@' on Ubuntu images
   sudo dnf install -y docker || sudo apt-get update && sudo apt-get install -y docker.io
   sudo systemctl enable --now docker

   # get the code onto the box (git clone, scp, or rsync this repo), then:
   sudo docker build -t aethos-rag-api .
   sudo docker run -d --restart=always --name aethos \
        -p 8000:8000 -v /opt/aethos_data:/data aethos-rag-api
   curl -s localhost:8000/healthz
   ```

4. **Persist the index across restarts.** The `-v /opt/aethos_data:/data` mount holds
   `/data/aethos_index/` (the codec) and `/data/aethos_index/texts.json`. On container start the app calls
   `AethosRAG._maybe_load_persisted()` and reloads the codec — **no re-ingest**. Back it up with a simple
   `tar czf aethos_index.tgz /opt/aethos_data/aethos_index`.

---

## B. Container Instances (no VM)

1. Push the image to **OCI Registry (OCIR)**:
   ```bash
   docker tag aethos-rag-api <region>.ocir.io/<tenancy-namespace>/aethos-rag-api:latest
   docker login <region>.ocir.io      # user: <tenancy-namespace>/<oci-username>, pass: an Auth Token
   docker push <region>.ocir.io/<tenancy-namespace>/aethos-rag-api:latest
   ```
2. Console → **Container Instances** → Create. Pull the OCIR image, 1 OCPU / 2–4 GB, expose port 8000,
   set env `AETHOS_INDEX_DIR=/data/aethos_index`.
3. For persistence, attach a **File Storage (FSS)** mount at `/data`, or accept that a restart requires a
   re-`/ingest` (the codec is small and rebuilds in seconds for a demo corpus).

---

## C. RAM footprint for a 100k-doc index (dim = 1024, M = 32, Kp = 256)

Everything below is resident RAM; it fits the 1 GB Container Instances shape and is trivial on A1 (24 GB).

| Component | Size | Note |
|---|---|---|
| PQ codes | `100k × 32 × 1 B` = **3.2 MB** | `uint8` (Kp=256); the whole vector index |
| Codebook | `256 × 1024 × 4 B` = **1.0 MB** | shared across the corpus |
| Rotation | **0 MB** | regenerated from a seed (invertible/regenerable) |
| Doc texts (returned in results) | ~**60 MB** | ~600 B/doc abstracts; drop if you only return ids |
| doc_ids + bookkeeping | ~**5 MB** | |
| Glass-box lexical lattice *(optional)* | ~**20 MB** | ~200 B/doc; enables named bridges in `/explain` |
| Python + numpy + fastapi + numba runtime | **250–400 MB** | fixed base |
| **Total** | **≈ 350–500 MB** | comfortably under 1 GB |

**Compression vs fp32, at scale:** a full-precision dense index would be `100k × 4096 B = 409.6 MB` of
vectors. The PQ codes are `100k × 32 B ≈ 3.2 MB` (+1 MB shared codebook) → **~100–128× smaller vectors**,
holding 96–97% of accuracy (MEASURED: scifact 0.7463→0.7223, nfcorpus 0.3814→0.3664). `/stats` reports the
live `bytes_per_doc` and `compression_vs_fp32` for your actual corpus (small corpora look worse because the
1 MB codebook is amortized over few docs — it approaches ~M bytes/doc as N grows).

---

## D. Notes

- **Reverse proxy / TLS:** for a public demo put nginx or Caddy in front of port 8000 for HTTPS.
- **Query embedding modes** (the one model touch): default is **precomputed** — the caller sends
  `query_embedding`. To accept raw text queries, set `AETHOS_ENCODER=<sentence-transformers-model>` and
  uncomment `torch` + `sentence-transformers` in `requirements.txt`; the encoder is pinned to `device="cpu"`
  and `CUDA_VISIBLE_DEVICES=""` is set in both the app and the Dockerfile, so it can never reach a GPU.
- **Scaling:** a single uvicorn worker keeps the in-process index consistent. To run multiple workers or
  replicas, move the codec to a shared volume/object store and load it read-only per worker.
- **Ingest embeddings** may be produced on any machine (even a GPU box, offline) — they are frozen into PQ
  codes at ingest. The deployed service never re-embeds a document and never needs a GPU.
