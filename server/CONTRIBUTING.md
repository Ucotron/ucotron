# Contributing to Ucotron

## Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Rust | stable (1.93+) | Core workspace compilation |
| Node.js | 20+ | TypeScript SDK, landing page, dashboard |
| Python | 3.12+ | Python SDK, fine-tuning scripts |
| Go | 1.23+ | Go SDK |
| JDK | 11+ (17 recommended) | Java SDK |
| PHP | 8.1+ | PHP SDK |
| Docker | 24+ | Container builds, security scanning |

### ONNX Models

Download ONNX models before running extraction or multimodal tests:

```bash
./scripts/download_models.sh
```

This places models in `models/` (gitignored):
- `all-MiniLM-L6-v2/` — embedding model (384-dim)
- `gliner_small-v2.1/` — NER model (zero-shot)
- `whisper-tiny/` — audio transcription
- `clip-vit-base-patch32/` — image embeddings (512-dim)

## Building

```bash
# Debug build
cargo build --workspace

# Release build (slow — LTO enabled)
cargo build --workspace --release

# Clippy (zero warnings policy)
cargo clippy --workspace --all-targets -- -D warnings

# Format check
cargo fmt --all -- --check
```

The workspace produces 3 binaries:
- `ucotron_server` — REST API (port 8420)
- `ucotron_mcp` — MCP server (stdio)
- `bench_runner` — Phase 1 benchmarks

## Testing

### Rust

```bash
# All workspace tests
cargo test --workspace

# Specific crate
cargo test -p ucotron-core
cargo test -p ucotron-helix
cargo test -p ucotron-extraction
cargo test -p ucotron-server
cargo test -p ucotron-sdk
```

### SDKs

```bash
# TypeScript
cd sdks/typescript && npm ci && npm test

# Python
cd sdks/python && pip install -e ".[dev]" && pytest

# Go
cd sdks/go && go test ./...

# Java
cd sdks/java && ./gradlew test

# PHP
cd sdks/php && composer install && ./vendor/bin/phpunit
```

### Integration Tests

Start the server, then run cross-language integration:

```bash
cargo run --bin ucotron_server &
./scripts/cross_language_tests.sh
```

## Commit Convention

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new endpoint for batch search
fix: correct HNSW index rebuild on empty dataset
docs: update API reference for multimodal endpoints
test: add integration tests for connector sync
refactor: extract common middleware into shared module
```

The release workflow generates changelogs from these commit prefixes.

---

## CI/CD Workflow Reference

### Workflow Overview

| Workflow | File | Trigger | Purpose |
|----------|------|---------|---------|
| CI | `ci.yml` | PR, push to main | Lint, test, build across all languages |
| Security | `security.yml` | PR, push, weekly (Mon 06:00 UTC), manual | Trivy vulnerability scanning |
| Benchmarks | `benchmarks.yml` | Weekly (Sun 02:00 UTC), manual | Performance regression tracking |
| Coverage | `coverage.yml` | PR, push | Multi-language coverage to Codecov |
| Release | `release.yml` | Git tag `v*` | Multi-registry publishing |
| Release Dry Run | `release-dry-run.yml` | Manual | Validate release pipeline |
| Fine-Tuning | `fine-tuning.yml` | Manual | LLM dataset generation and training |

### CI Workflow (`ci.yml`)

Triggered on PRs to `main` or `ralph/ucotron`, and pushes to `main`.

**Jobs:**

| Job | Runner | What it does |
|-----|--------|-------------|
| `rust-lint` | ubuntu-latest | `cargo clippy` + `cargo fmt --check` |
| `rust-test` | Matrix: linux-x64, linux-arm64, macos-arm64 | `cargo test --workspace` |
| `multimodal-tests` | ubuntu-latest | Audio, image, video pipeline tests (sequential) |
| `typescript-sdk` | ubuntu-latest | Node 20, `npm ci && npm test` |
| `python-sdk` | ubuntu-latest | Python 3.12, `pip install && pytest` |
| `go-sdk` | ubuntu-latest | Go 1.23, `go vet && go test` |
| `java-sdk` | Matrix: JDK 11, 17, 21 | `./gradlew test` |
| `php-sdk` | Matrix: PHP 8.1, 8.2, 8.3 | `phpunit` |
| `docs-build` | ubuntu-latest | Next.js docs build verification |
| `integration` | ubuntu-latest | Cross-language tests against running server |

**Model caching:** ONNX models are cached in two groups:
- Base models (MiniLM, GLiNER) — keyed by `scripts/download_models.sh` SHA256
- Multimodal models (Whisper, CLIP) — separate cache key

### Security Workflow (`security.yml`)

Runs Trivy scanner on:
- **Filesystem**: Scans `Cargo.lock` and lockfiles for dependency vulnerabilities
- **Docker image**: Scans built container image

Fails on CRITICAL or HIGH severity findings. Results uploaded as SARIF to GitHub Advanced Security.

### Benchmarks Workflow (`benchmarks.yml`)

Manual dispatch parameters:
- `node_count` (default: 10000)
- `edge_count` (default: 50000)
- `query_count` (default: 500)

**Jobs:** `build` → `bench-ingest` → `bench-search` → `summary`

Artifacts retained for 90 days with metadata (run number, trigger, commit, branch, date).

### Coverage Workflow (`coverage.yml`)

Each language uploads coverage separately with a flag to Codecov:

| Language | Tool | Format | Flag |
|----------|------|--------|------|
| Rust | cargo-llvm-cov | LCOV | `rust` |
| TypeScript | Jest | LCOV | `typescript` |
| Python | pytest-cov | XML | `python` |
| Go | go test -coverprofile | Go | `go` |
| Java | JaCoCo (JDK 17) | XML | `java` |
| PHP | PHPUnit + Xdebug (8.3) | XML | `php` |

### Release Workflow (`release.yml`)

Triggered by pushing a tag matching `v*` (e.g., `v0.1.0`, `v2.0.0`).

**Pipeline stages:**

1. **validate** — Lint + test on linux-x64 and macos-arm64
2. **docker** — Multi-arch build (linux/amd64 + linux/arm64), push to GHCR (+ optional Docker Hub)
3. **publish-rust** — Publishes `ucotron-core`, `ucotron-config`, `ucotron-sdk` to crates.io (in dependency order, 30s waits between publishes)
4. **publish-typescript** — `@ucotron/sdk` to npm
5. **publish-python** — `ucotron-sdk` to PyPI via twine
6. **publish-go** — Tag-based, auto-discovered by proxy.golang.org
7. **publish-java** — Maven Central via Sonatype OSSRH (GPG signed)
8. **publish-php** — Packagist via webhook
9. **docs** — Build docs site
10. **release** — Create GitHub Release with conventional-commit changelog

Semantic versioning tags: `vX.Y.Z`, `vX.Y`, `vX`, `latest` (skips `latest` for pre-releases).

### Release Dry Run (`release-dry-run.yml`)

Manual dispatch with `version` parameter (default: `0.0.0-dryrun`). Runs the full release pipeline with `--dry-run` flags — Docker build without push, `cargo publish --dry-run`, `npm pack` without publish, etc.

### Fine-Tuning Workflow (`fine-tuning.yml`)

Manual dispatch with parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_tier` | — | `slm` (0.5B) / `small` (1.5B) / `medium` (7B) |
| `training_type` | — | `sft` / `dpo` / `sft+dpo` |
| `generate_datasets` | `true` | Generate training datasets |
| `train_projection` | `false` | Train CLIP→MiniLM projection layer |
| `dataset_samples` | `0` | Override sample count (0 = config defaults) |
| `dry_run` | `false` | Validate without calling Fireworks API |

**Pipeline:** `validate` → `generate-datasets` (conditional) → `sft-training` (conditional) → `dpo-training` (conditional, uses SFT model as base) → `projection-training` (conditional) → `summary`

---

## GitHub Secrets

All secrets are configured in **Settings → Secrets and variables → Actions**.

### Required for Releases

| Secret | Used By | Description |
|--------|---------|-------------|
| `CARGO_REGISTRY_TOKEN` | release.yml | crates.io API token for publishing Rust crates |
| `NPM_TOKEN` | release.yml | npm access token for publishing `@ucotron/sdk` |
| `PYPI_TOKEN` | release.yml | PyPI API token for publishing `ucotron-sdk` via twine |
| `GPG_PRIVATE_KEY` | release.yml | ASCII-armored GPG key for signing Maven artifacts |
| `GPG_PASSPHRASE` | release.yml | Passphrase for the GPG signing key |
| `OSSRH_USERNAME` | release.yml | Sonatype OSSRH username for Maven Central publishing |
| `OSSRH_PASSWORD` | release.yml | Sonatype OSSRH password for Maven Central publishing |
| `PACKAGIST_TOKEN` | release.yml | Packagist API token for PHP SDK webhook trigger |
| `PACKAGIST_USERNAME` | release.yml | Packagist username for PHP SDK publishing |

### Optional for Releases

| Secret | Used By | Description |
|--------|---------|-------------|
| `DOCKERHUB_USERNAME` | release.yml | Docker Hub username (optional mirror alongside GHCR) |
| `DOCKERHUB_TOKEN` | release.yml | Docker Hub access token (optional mirror alongside GHCR) |

### Required for Coverage

| Secret | Used By | Description |
|--------|---------|-------------|
| `CODECOV_TOKEN` | coverage.yml | Codecov.io upload token (used by all 6 language jobs) |

### Required for Fine-Tuning

| Secret | Used By | Description |
|--------|---------|-------------|
| `FIREWORKS_API_KEY` | fine-tuning.yml | Fireworks.ai API key for model training |
| `FIREWORKS_ACCOUNT_ID` | fine-tuning.yml | Fireworks.ai account identifier |

### Optional for Fine-Tuning

| Secret | Used By | Description |
|--------|---------|-------------|
| `WANDB_API_KEY` | fine-tuning.yml | Weights & Biases API key for experiment tracking |

### Built-in

| Secret | Used By | Description |
|--------|---------|-------------|
| `GITHUB_TOKEN` | release.yml, security.yml | Auto-provided by GitHub Actions (GHCR login, release creation, SARIF upload) |

### Setting Up Secrets

1. Go to the repository **Settings → Secrets and variables → Actions**
2. Click **New repository secret**
3. Enter the secret name exactly as listed above and paste the value

**Obtaining tokens:**

| Token | How to get it |
|-------|---------------|
| `CARGO_REGISTRY_TOKEN` | [crates.io/settings/tokens](https://crates.io/settings/tokens) — create with `publish-new` and `publish-update` scopes |
| `NPM_TOKEN` | `npm token create --read-only=false` or [npmjs.com/settings/tokens](https://www.npmjs.com/settings/tokens) |
| `PYPI_TOKEN` | [pypi.org/manage/account/token](https://pypi.org/manage/account/token/) — scope to `ucotron-sdk` project |
| `GPG_PRIVATE_KEY` | `gpg --armor --export-secret-keys YOUR_KEY_ID` |
| `GPG_PASSPHRASE` | The passphrase you set when generating the GPG key |
| `OSSRH_USERNAME` / `OSSRH_PASSWORD` | Register at [issues.sonatype.org](https://issues.sonatype.org/) and create a Jira ticket for `com.ucotron` group ID |
| `PACKAGIST_TOKEN` | [packagist.org/profile](https://packagist.org/profile/) → API Tokens |
| `CODECOV_TOKEN` | [app.codecov.io](https://app.codecov.io/) → Add repository → copy upload token |
| `FIREWORKS_API_KEY` | [fireworks.ai/account/api-keys](https://fireworks.ai/account/api-keys) |
| `FIREWORKS_ACCOUNT_ID` | Fireworks dashboard → Account Settings |
| `WANDB_API_KEY` | [wandb.ai/authorize](https://wandb.ai/authorize) |

---

## New Contributor Setup

### 1. Clone and Build

```bash
git clone <repo-url>
cd Ucotron/memory_arena

# Download ONNX models
./scripts/download_models.sh

# Build workspace
cargo build --workspace

# Run tests
cargo test --workspace
```

### 2. Install SDK Dependencies

```bash
# TypeScript
cd sdks/typescript && npm ci && cd ../..

# Python
cd sdks/python && pip install -e ".[dev]" && cd ../..

# Go (no install step needed)

# Java
cd sdks/java && ./gradlew build && cd ../..

# PHP
cd sdks/php && composer install && cd ../..
```

### 3. Run the Server Locally

```bash
# Create a config file (or use defaults)
cargo run --bin ucotron_server

# Server starts on http://localhost:8420
# Health check: GET http://localhost:8420/api/v1/health
```

### 4. Project Structure

```
memory_arena/
├── core/               # Shared traits, types, algorithms
├── helix_impl/         # HelixDB (LMDB) backend
├── ucotron_config/     # TOML config, env overrides
├── ucotron_extraction/ # Embeddings, NER, relations, orchestrators
├── ucotron_server/     # Axum REST API + MCP server
├── ucotron_sdk/        # Rust SDK client
├── ucotron_connectors/ # External connectors (Slack, GitHub, etc.)
├── bench_runner/       # Phase 1 benchmark CLI
├── sdks/               # TypeScript, Python, Go, Java, PHP SDKs
├── landing/            # Next.js landing page
├── dashboard/          # Next.js admin dashboard
├── templates/          # Framework integration templates
├── scripts/            # Build, training, and utility scripts
├── deploy/             # Helm charts, Grafana dashboards, Terraform
├── models/             # ONNX models (gitignored)
└── docs/               # ADRs, runbooks
```

### 5. Key Conventions

- **Serialization**: bincode v1 (positional — never use `skip_serializing_if` on Node fields)
- **Embeddings**: 384-dim (text/MiniLM), 512-dim (visual/CLIP)
- **IDs**: `NodeId = u64`, `FactId = u64`
- **Multi-tenancy**: `X-Ucotron-Namespace` header on all API requests
- **Auth**: `X-Api-Key` header with RBAC (admin, writer, reader, viewer)
- **Config**: TOML files with environment variable overrides (`UCOTRON_` prefix)

### 6. Running Benchmarks

```bash
# Build release binary
cargo build --release -p bench_runner

# Ingestion benchmark
./target/release/bench_runner ingest --count 10000 --edges 50000

# Search benchmark
./target/release/bench_runner search --count 1000 --queries 100

# Recursion benchmark
./target/release/bench_runner recursion --depths 10,20,50,100
```
