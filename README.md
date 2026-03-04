
# vLLM Docker Build System

Build vLLM + FlashInfer from source for any NVIDIA GPU architecture. Auto-detects your GPU, tracks upstream changes, and rebuilds only when needed.

Forked from [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker).

## Quick Start

```bash
git clone https://github.com/tylerwagler/spark-vllm-docker.git
cd spark-vllm-docker

# Build (auto-detects GPU arch, checks for upstream changes)
./build.sh

# Check what's stale without building
./build.sh --check

# Run
docker run --privileged --gpus all -it --rm \
  --network host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node \
  vllm serve <model-name> --port 8000 --host 0.0.0.0
```

## Build Script

`build.sh` runs a 3-phase Docker build:

1. **FlashInfer wheels** — compiled from source with cubin caching
2. **vLLM wheels** — compiled from source with ccache
3. **Runner image** — clean NGC PyTorch base + wheels installed

Wheels are cached in `./wheels/` and reused across builds. SHA tracking via `git ls-remote` auto-detects upstream changes — no manual `--rebuild-*` flags needed for routine updates.

### Options

| Flag | Description |
| :--- | :--- |
| `-t, --tag <tag>` | Image tag (default: `vllm-node`) |
| `--gpu-arch <arch>` | GPU architecture (default: auto-detect via `nvidia-smi`) |
| `--rebuild-flashinfer` | Force FlashInfer rebuild (ignore cached wheels) |
| `--rebuild-vllm` | Force vLLM rebuild (ignore cached wheels) |
| `--vllm-ref <ref>` | vLLM commit SHA, branch or tag (default: `main`) |
| `--apply-vllm-pr <pr-num>` | Apply a vLLM PR patch during build (repeatable) |
| `-j, --build-jobs <jobs>` | Parallel build jobs (default: `nproc - 4`) |
| `--check` | Dry-run: report staleness of all components and exit |
| `--full-log` | Full Docker build output (`--progress=plain`) |
| `-h, --help` | Show help |

### Staleness Check

`./build.sh --check` reports the status of:

- **FlashInfer** and **vLLM** — compares local wheel SHA against upstream `main`
- **CUTLASS** — compares pinned version against latest GitHub release
- **NGC base image** — compares current tag against latest on nvcr.io

```
=========================================
         STALENESS CHECK
=========================================
  ✓ FlashInfer  — up to date (cb593c82)
  ✗ vLLM        — stale (local: 70c73df6, remote: a1b2c3d4)
  ✓ CUTLASS     — v4.4.1 is latest
  ✓ NGC Image   — 26.02-py3 is latest
=========================================
1 component(s) are stale.
```

### Build Examples

```bash
# Standard build (auto-detect everything)
./build.sh

# Build for specific GPU arch
./build.sh --gpu-arch "7.5"

# Build with a specific vLLM commit
./build.sh --vllm-ref v0.9.1

# Apply a vLLM PR patch
./build.sh --apply-vllm-pr 34695

# Force rebuild everything
./build.sh --rebuild-flashinfer --rebuild-vllm
```

## Model Management

### Check Model Freshness

```bash
# Check if cached HuggingFace models are up to date
./check-models.sh

# Check and download any outdated models
./check-models.sh --update
```

## Running Models

### Basic Docker Run

```bash
docker run --privileged --gpus all -it --rm \
  --network host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm-node \
  vllm serve <model> \
    --port 8000 --host 0.0.0.0 \
    --gpu-memory-utilization 0.7 \
    --load-format fastsafetensors
```

### Docker Compose

The intended workflow is to define services in a `docker-compose.yml`:

```yaml
services:
  vllm:
    image: vllm-node
    privileged: true
    network_mode: host
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    volumes:
      - ~/.cache/huggingface:/root/.cache/huggingface
    command: >
      vllm serve <model>
        --port 8000 --host 0.0.0.0
        --gpu-memory-utilization 0.7
        --load-format fastsafetensors
```

### Fastsafetensors

This build includes [fastsafetensors](https://github.com/foundation-model-stack/fastsafetensors/) for faster model loading. Use `--load-format fastsafetensors` to enable. Avoid it for models that consume >0.8 of available VRAM (without KV cache) as it may OOM.

## Mods and Patches

Mods are runtime patches applied when launching a container, useful for model-specific fixes without rebuilding.

### Available Mods

| Mod | Model | What it does |
| :--- | :--- | :--- |
| `fix-Salyut1-GLM-4.7-NVFP4` | Salyut1/GLM-4.7-NVFP4 | Patches GLM4 MoE parser for fused QKV quantization |
| `fix-glm-4.7-flash-AWQ` | cyankiwi/GLM-4.7-Flash-AWQ-4bit | Speed optimization patch + vLLM crash fix |
| `fix-qwen3-coder-next` | Qwen/Qwen3-Coder-Next-FP8 | Fixes startup crash, reverts perf regression, Triton allocator fix |
| `fix-qwen3-next-autoround` | Intel/Qwen3-Coder-Next-INT4-AutoRound | Reverts incompatible PR |
| `fix-qwen3.5-autoround` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | ROPE syntax fix + OpaqueBase crash workaround |
| `nemotron-nano` | nvidia/NVIDIA-Nemotron-3-Nano-* | Downloads reasoning parser plugin |

### Applying Mods

Mods are applied by running their `run.sh` inside the container before starting vllm:

```bash
docker run --privileged --gpus all -it --rm \
  --network host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/mods:/workspace/mods \
  vllm-node \
  bash -c "/workspace/mods/fix-qwen3.5-autoround/run.sh && vllm serve ..."
```

### Creating Mods

1. Create a directory in `mods/`
2. Add patch files (`.patch`) or other assets
3. Create a `run.sh` that applies the patches

## Recipes

Reference configurations in `recipes/*.yaml`. These were designed for the upstream `run-recipe.sh` launcher (removed in this fork), but remain useful as documentation of known-good vLLM arguments for each model.

| Recipe | Model |
| :--- | :--- |
| `glm-4.7-flash-awq` | cyankiwi/GLM-4.7-Flash-AWQ-4bit |
| `minimax-m2-awq` | QuantTrio/MiniMax-M2-AWQ |
| `minimax-m2.5-awq` | cyankiwi/MiniMax-M2.5-AWQ-4bit |
| `nemotron-3-nano-nvfp4` | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 |
| `openai-gpt-oss-120b` | openai/gpt-oss-120b |
| `qwen3-coder-next-fp8` | Qwen/Qwen3-Coder-Next-FP8 |
| `qwen3.5-122b-int4-autoround` | Intel/Qwen3.5-122B-A10B-int4-AutoRound |

## Build Cache Maintenance

Check build cache size and prune periodically:

```bash
docker system df
docker builder prune --filter until=72h
```

Don't prune after every build — the ccache and repo caches significantly speed up subsequent compilations.
