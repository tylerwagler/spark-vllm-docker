# syntax=docker/dockerfile:1.6

# Limit build parallelism to reduce OOM situations
ARG BUILD_JOBS=16
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.02-py3

# =========================================================
# STAGE 1: Base Image (Installs Dependencies)
# =========================================================
FROM ${BASE_IMAGE} AS base

# Build parallelism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

# Set non-interactive frontend to prevent apt prompts
ENV DEBIAN_FRONTEND=noninteractive

# Allow pip to install globally on Ubuntu 24.04 without a venv
ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Set the base directory environment variable
ENV VLLM_BASE_DIR=/workspace/vllm

# 1. Install Build Dependencies & Ccache
# Added ccache to enable incremental compilation caching
RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim ninja-build git \
    ccache \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn

# Install shared build/runtime dependencies
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
     uv pip install nvidia-nvshmem-cu13 "apache-tvm-ffi<0.2"

# Configure Ccache for CUDA/C++
ENV PATH=/usr/lib/ccache:$PATH
ENV CCACHE_DIR=/root/.ccache
# Limit ccache size to prevent unbounded growth (e.g. 50G)
ENV CCACHE_MAXSIZE=50G
# Enable compression to save space
ENV CCACHE_COMPRESS=1
# Tell CMake to use ccache for compilation
ENV CMAKE_CXX_COMPILER_LAUNCHER=ccache
ENV CMAKE_CUDA_COMPILER_LAUNCHER=ccache

# Setup Workspace
WORKDIR $VLLM_BASE_DIR

# 2. Set Environment Variables
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas

# =========================================================
# STAGE 2: vLLM Builder
# =========================================================
FROM base AS vllm-builder

ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
WORKDIR $VLLM_BASE_DIR

# Git reference (branch, tag, or SHA) to checkout
ARG VLLM_REF=main

# Smart Git Clone — shallow to minimize clone/fetch time
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "vllm" ]; then \
        echo "Cache miss: Cloning vLLM (shallow)..." && \
        git clone --depth 1 --recurse-submodules --shallow-submodules \
            https://github.com/vllm-project/vllm.git && \
        if [ "$VLLM_REF" != "main" ]; then \
            cd vllm && \
            git fetch --depth 1 origin ${VLLM_REF} && \
            git checkout FETCH_HEAD; \
        fi; \
    else \
        echo "Cache hit: Fetching vLLM updates (shallow)..." && \
        cd vllm && \
        git fetch --depth 1 origin ${VLLM_REF} && \
        git checkout FETCH_HEAD && \
        git submodule update --init --recursive && \
        git clean -fdx; \
    fi && \
    cp -a /repo-cache/vllm $VLLM_BASE_DIR/

WORKDIR $VLLM_BASE_DIR/vllm

ARG VLLM_PRS=""

RUN if [ -n "$VLLM_PRS" ]; then \
        echo "Applying PRs: $VLLM_PRS"; \
        for pr in $VLLM_PRS; do \
            echo "Fetching and applying PR #$pr..."; \
            curl -fL "https://github.com/vllm-project/vllm/pull/${pr}.diff" | git apply -v; \
        done; \
    fi

# Override CUTLASS with a newer version (vLLM pins v4.2.1, we want latest for SM121/GB10 fixes)
ARG CUTLASS_REF=v4.4.1
RUN --mount=type=cache,id=repo-cache,target=/repo-cache \
    cd /repo-cache && \
    if [ ! -d "cutlass" ]; then \
        echo "Cache miss: Cloning CUTLASS (shallow)..." && \
        git clone --depth 1 https://github.com/NVIDIA/cutlass.git && \
        cd cutlass && git checkout ${CUTLASS_REF}; \
    else \
        echo "Cache hit: Fetching CUTLASS updates (shallow)..." && \
        cd cutlass && \
        git fetch --depth 1 origin tag ${CUTLASS_REF} && \
        git checkout FETCH_HEAD && \
        git clean -fdx; \
    fi && \
    cp -a /repo-cache/cutlass /workspace/cutlass
ENV VLLM_CUTLASS_SRC_DIR=/workspace/cutlass

# Prepare build requirements
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    python3 use_existing_torch.py && \
    sed -i "/flashinfer/d" requirements/cuda.txt && \
    sed -i '/^triton\b/d' requirements/test.txt && \
    sed -i '/^fastsafetensors\b/d' requirements/test.txt && \
    uv pip install -r requirements/build.txt

# Final Compilation
RUN --mount=type=cache,id=ccache,target=/root/.ccache \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv build --no-build-isolation --wheel . --out-dir=/workspace/wheels -v && \
    echo "=== ccache stats ===" && ccache -s

# =========================================================
# STAGE 3: vLLM Wheel Export
# =========================================================
FROM scratch AS vllm-export
COPY --from=vllm-builder /workspace/wheels /

# =========================================================
# STAGE 4: Runner (Installs wheels from host ./wheels/)
# =========================================================
ARG BASE_IMAGE
FROM ${BASE_IMAGE} AS runner

# Transferring build settings from build image because of ptxas/jit compilation during vLLM startup
# Build parallelism
ARG BUILD_JOBS
ENV MAX_JOBS=${BUILD_JOBS}
ENV CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS}
ENV NINJAFLAGS="-j${BUILD_JOBS}"
ENV MAKEFLAGS="-j${BUILD_JOBS}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ARG VLLM_BASE_DIR=/workspace/vllm

# Set pip cache directory
ENV PIP_CACHE_DIR=/root/.cache/pip
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1
ENV UV_LINK_MODE=copy

# Install runtime dependencies
RUN apt update && \
    apt install -y --no-install-recommends \
    curl vim git \
    libxcb1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install uv && pip uninstall -y flash-attn # triton-kernels pytorch-triton

# Set final working directory
WORKDIR $VLLM_BASE_DIR

# Download Tiktoken files
RUN mkdir -p tiktoken_encodings && \
    wget -O tiktoken_encodings/o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken" && \
    wget -O tiktoken_encodings/cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

# Install wheels from host ./wheels/ (bind-mounted from build context — no layer bloat)
# Override vLLM's transformers<5 constraint to get transformers>=5
RUN --mount=type=bind,source=wheels,target=/workspace/wheels \
    --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    echo "transformers>=5.0.0" > /tmp/tf-override.txt && \
    uv pip install /workspace/wheels/*.whl --override /tmp/tf-override.txt

# Setup environment for runtime
ARG TORCH_CUDA_ARCH_LIST="12.1a"
ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV TIKTOKEN_ENCODINGS_BASE=$VLLM_BASE_DIR/tiktoken_encodings
ENV PATH=$VLLM_BASE_DIR:$PATH

# Final extra deps
RUN --mount=type=cache,id=uv-cache,target=/root/.cache/uv \
    uv pip install ray[default] fastsafetensors nvidia-nvshmem-cu13 \
        flashinfer-python flashinfer-cubin