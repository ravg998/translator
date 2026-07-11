# Translator — Transformer-based Machine Translation (EN → FR)

> **Proof of concept.** This project is a proof of concept carried out for
> educational purposes: to **break down and implement a Transformer end to end**,
> and then bring it to life within a real production pipeline (CI/CD + cloud
> deployment). The goal is not to reach state-of-the-art translation quality — the
> model is intentionally small — but to understand **every building block**, from
> the neural architecture all the way to the server that hosts it.

---

## Objective

Build a machine translation model (English → French) by implementing a
**Transformer from scratch in PyTorch**, without relying on high-level pre-built
layers. Every component of the architecture described in *Attention Is All You Need*
was reimplemented and studied individually:

- Input embeddings and positional encoding
- Multi-head attention (with causal and padding masking)
- Position-wise feed-forward network
- Residual connections and layer normalization
- Encoder / decoder blocks assembled by hand
- Final projection to the target vocabulary
- Training loop (teacher forcing) and greedy decoding at inference time

The model is trained on the `Helsinki-NLP/opus_books` dataset.

---

## What this project demonstrates

Beyond the model itself, this repository serves as a **guided thread for learning a
complete software delivery pipeline**, from building the artifact to automatically
deploying it on a remote server:

1. **Containerization** of the application with Docker (lightweight image, CPU-only
   torch).
2. **Continuous Integration (CI)**: image build and test execution on every change.
3. **Continuous Delivery**: publishing the image to a registry.
4. **Continuous Deployment (CD)**: automatic update of the service on a cloud virtual
   machine.

---

## Tech stack

| Area | Technologies |
| --- | --- |
| Model / ML | Python, PyTorch, Transformer implemented from scratch |
| Python project management | `uv`, `pyproject.toml`, configuration via Pydantic |
| Containerization | Docker, Docker Compose |
| CI/CD | Jenkins (configured via *Configuration as Code* / JCasC) |
| Image registry | GitHub Container Registry (GHCR) |
| Model weights hosting | Hugging Face Hub |
| Deployment | Google Cloud Platform (Compute Engine — VM) |
| Testing | pytest |

---

## CI/CD pipeline

The pipeline is orchestrated by **Jenkins**, set up as reproducible, version-controlled
infrastructure (custom Jenkins image, pinned plugins, configuration described in YAML
via JCasC, secrets isolated in the *Credentials Store*).

On every iteration pushed to the `main` branch, the pipeline automatically runs:

```
 Code (GitHub)
      |
      v
 +--------------------------------------------------------------+
 |  JENKINS                                                     |
 |   1. Build     -> builds the application's Docker image      |
 |   2. Test      -> runs the pytest suite inside the container |
 |   3. Push      -> publishes the image to GHCR                |
 |   4. Deploy    -> SSH into the GCP VM, pull image, restart   |
 +--------------------------------------------------------------+
      |
      v
 GCP server (Compute Engine) — the service runs
```

Implementation highlights:

- **Jenkins drives the host's Docker engine** (Docker client + mounted socket).
- **Optimized multi-stage build**: heavy dependencies cached, image slimmed down by
  installing the **CPU-only** version of PyTorch (~8 GB -> ~1 GB).
- **Weights and tokenizer** downloaded from **Hugging Face Hub** at build time, so the
  image is a self-contained artifact (large binary files are not versioned in Git).
- **Deployment over SSH** from Jenkins to the VM, using `docker compose` on the server
  side (`base` + production `override` configuration).
- **Strict secret management**: no secrets in plain text in the code — tokens and keys
  stored in the Jenkins credentials store or injected via environment variables.

---

## Repository structure

```
translator/
├── config/            # Model and training configuration (YAML)
├── scripts/           # Training and evaluation scripts
├── src/translator/    # Source code: model, dataset, tokenizer, API
├── tests/             # Unit tests (pytest)
├── Dockerfile         # Application image build
├── docker-compose*.yml# Deployment configuration (local / production)
├── Jenkinsfile        # CI/CD pipeline definition
├── pyproject.toml     # Dependencies and project metadata (uv)
└── uv.lock
```

> The Jenkins infrastructure itself (custom image, JCasC, plugins) is managed in a
> separate dedicated repository, following the *application vs infrastructure*
> separation principle.

---

## Model configuration

The model is intentionally compact (suited to CPU inference):

| Parameter | Value |
| --- | --- |
| `d_model` | 128 |
| Attention heads | 4 |
| Encoder / decoder layers | 2 / 2 |
| Feed-forward dimension | 512 |
| Sequence length | 120 |
| Languages | EN → FR |

---

## Status & limitations

This repository is an **educational proof of concept**. Because of its small size,
the model does not produce professional-grade translations: its value lies in
**understanding the Transformer architecture** and in **mastering an end-to-end CI/CD
pipeline**, from code to production server.

Possible next steps: scaling up the model, exposing a public inference API (with
authentication and rate limiting), image versioning, automatic pipeline triggering via
webhook, and health check / rollback on deployment.
