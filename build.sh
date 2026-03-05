#!/bin/bash
set -e

# Start total time tracking
START_TIME=$(date +%s)

# Default values
IMAGE_TAG="vllm-node"
REBUILD_VLLM=false
VLLM_REF="main"
VLLM_REF_SET=false
VLLM_PRS=""
FULL_LOG=false
BUILD_JOBS=$(( $(nproc) - 4 ))
GPU_ARCH_LIST=""

CHECK_ONLY=false

VLLM_REPO="https://github.com/vllm-project/vllm.git"
CUTLASS_REPO="https://github.com/NVIDIA/cutlass.git"
CUTLASS_REF="v4.4.1"
NGC_IMAGE="nvcr.io/nvidia/pytorch"
NGC_TAG="26.02-py3"

# Resolve the HEAD SHA for a remote git ref (branch or tag)
# Returns empty string on failure (e.g. no network)
resolve_remote_sha() {
    local url="$1"
    local ref="$2"
    local sha
    sha=$(git ls-remote "$url" "refs/heads/$ref" 2>/dev/null | head -1 | cut -f1)
    if [ -z "$sha" ]; then
        sha=$(git ls-remote "$url" "refs/tags/$ref" 2>/dev/null | head -1 | cut -f1)
    fi
    echo "$sha"
}

# Check if this system requires building vLLM from source.
# Returns 0 (true) if source build is needed, 1 (false) if prebuilt wheels work.
requires_source_build() {
    # Custom PRs require source build
    [ -n "$VLLM_PRS" ] && return 0
    # Custom ref requires source build
    [ "$VLLM_REF_SET" = true ] && return 0
    # SM 12.x+ GPUs need unmerged upstream patches + custom CUTLASS
    for arch in $(echo "$GPU_ARCH_LIST" | tr ';' ' '); do
        local major="${arch%%.*}"
        major="${major%a}"
        [ "$major" -ge 12 ] && return 0
    done
    return 1
}

# Help function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -t, --tag <tag>               : Image tag (default: 'vllm-node')"
    echo "  --gpu-arch <arch>             : GPU architecture (default: auto-detect from nvidia-smi)"
    echo "  --rebuild-vllm                : Force rebuild of vLLM wheels (ignore cached wheels)"
    echo "  --vllm-ref <ref>              : vLLM commit SHA, branch or tag (default: 'main')"
    echo "  -j, --build-jobs <jobs>       : Number of concurrent build jobs (default: ${BUILD_JOBS})"
    echo "  --apply-vllm-pr <pr-num>      : Apply a specific PR patch to vLLM source. Can be specified multiple times."
    echo "  --check                       : Dry-run: report what's stale without building"
    echo "  --full-log                    : Enable full build logging (--progress=plain)"
    echo "  -h, --help                    : Show this help message"
    exit 1
}

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--tag) IMAGE_TAG="$2"; shift ;;
        --gpu-arch) GPU_ARCH_LIST="$2"; shift ;;
        --rebuild-vllm) REBUILD_VLLM=true ;;
        --vllm-ref) VLLM_REF="$2"; VLLM_REF_SET=true; shift ;;
        -j|--build-jobs) BUILD_JOBS="$2"; shift ;;
        --apply-vllm-pr)
            if [ -n "$2" ] && [[ "$2" != -* ]]; then
               if [ -n "$VLLM_PRS" ]; then
                   VLLM_PRS="$VLLM_PRS $2"
               else
                   VLLM_PRS="$2"
               fi
               shift
            else
               echo "Error: --apply-vllm-pr requires a PR number."
               exit 1
            fi
            ;;
        --check) CHECK_ONLY=true ;;
        --full-log) FULL_LOG=true ;;
        -h|--help) usage ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
    shift
done

# Auto-detect GPU architecture if not specified
if [ -z "$GPU_ARCH_LIST" ]; then
    if command -v nvidia-smi &>/dev/null; then
        # Query all GPUs, deduplicate, format as "major.minor" with Blackwell+ getting "a" suffix
        GPU_ARCH_LIST=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | while read -r cap; do
            major="${cap%%.*}"
            if [ "$major" -ge 12 ]; then
                echo "${cap}a"
            else
                echo "$cap"
            fi
        done | paste -sd ';')
        echo "Auto-detected GPU architecture: $GPU_ARCH_LIST"
    else
        echo "Error: No GPU detected and --gpu-arch not specified."
        exit 1
    fi
fi

# Ensure wheels directory exists
mkdir -p ./wheels

# ----------------------------------------------------------
# --check mode: report staleness and exit
# ----------------------------------------------------------
if [ "$CHECK_ONLY" = true ]; then
    echo ""
    echo "========================================="
    echo "         STALENESS CHECK"
    echo "========================================="
    STALE=0

    if requires_source_build; then
        echo "  Build mode: source (GPU arch $GPU_ARCH_LIST)"
        echo ""

        # vLLM — SHA-based staleness
        if compgen -G "./wheels/vllm*.whl" > /dev/null 2>&1; then
            VLLM_REMOTE_SHA=$(resolve_remote_sha "$VLLM_REPO" "$VLLM_REF")
            VLLM_LOCAL_SHA=""
            [ -f "./wheels/.vllm-sha" ] && VLLM_LOCAL_SHA=$(cat "./wheels/.vllm-sha")
            if [ -n "$VLLM_REMOTE_SHA" ] && [ "$VLLM_REMOTE_SHA" != "$VLLM_LOCAL_SHA" ]; then
                echo "  ✗ vLLM        — stale (local: ${VLLM_LOCAL_SHA:0:8}, remote: ${VLLM_REMOTE_SHA:0:8})"
                STALE=$((STALE + 1))
            else
                echo "  ✓ vLLM        — up to date (${VLLM_LOCAL_SHA:0:8})"
            fi
        else
            echo "  ✗ vLLM        — no wheels found"
            STALE=$((STALE + 1))
        fi

        # CUTLASS — only relevant for source builds
        CUTLASS_LATEST=$(git ls-remote --tags --sort=-v:refname "$CUTLASS_REPO" "refs/tags/v*" 2>/dev/null \
            | grep -v '{}' | head -1 | sed 's|.*/||')
        if [ -n "$CUTLASS_LATEST" ] && [ "$CUTLASS_LATEST" != "$CUTLASS_REF" ]; then
            echo "  ✗ CUTLASS     — pinned $CUTLASS_REF, latest $CUTLASS_LATEST"
            STALE=$((STALE + 1))
        else
            echo "  ✓ CUTLASS     — $CUTLASS_REF is latest"
        fi
    else
        echo "  Build mode: prebuilt (GPU arch $GPU_ARCH_LIST)"
        echo ""

        # vLLM — just check if wheel is present
        if compgen -G "./wheels/vllm*.whl" > /dev/null 2>&1; then
            VLLM_WHL=$(basename ./wheels/vllm*.whl | head -1)
            echo "  ✓ vLLM        — prebuilt wheel present ($VLLM_WHL)"
        else
            echo "  ✗ vLLM        — no prebuilt wheel (run build to download)"
            STALE=$((STALE + 1))
        fi
    fi

    # NGC base image (query Docker registry for latest tag)
    NGC_LATEST=""
    NGC_TOKEN=$(curl -s "https://nvcr.io/proxy_auth?scope=repository:nvidia/pytorch:pull" 2>/dev/null \
        | grep -oP '"token"\s*:\s*"\K[^"]+')
    if [ -n "$NGC_TOKEN" ]; then
        NGC_LATEST=$(curl -s -H "Authorization: Bearer $NGC_TOKEN" \
            "https://nvcr.io/v2/nvidia/pytorch/tags/list" 2>/dev/null \
            | python3 -c "
import sys, json
data = json.load(sys.stdin)
tags = [t for t in data.get('tags', []) if t.endswith('-py3') and not any(x in t for x in ['igpu', 'arm', 'qnx'])]
tags.sort()
print(tags[-1] if tags else '')
" 2>/dev/null)
    fi
    if [ -n "$NGC_LATEST" ] && [ "$NGC_LATEST" != "$NGC_TAG" ]; then
        echo "  ✗ NGC Image   — using $NGC_TAG, latest $NGC_LATEST"
        STALE=$((STALE + 1))
    elif [ -n "$NGC_LATEST" ]; then
        echo "  ✓ NGC Image   — $NGC_TAG is latest"
    else
        echo "  ? NGC Image   — could not check (registry unreachable)"
    fi

    echo "========================================="
    if [ "$STALE" -gt 0 ]; then
        echo "$STALE component(s) are stale."
        exit 1
    else
        echo "Everything is up to date."
        exit 0
    fi
fi

# Common build flags
COMMON_BUILD_FLAGS=()
if [ "$FULL_LOG" = true ]; then
    COMMON_BUILD_FLAGS+=("--progress=plain")
fi
COMMON_BUILD_FLAGS+=("--build-arg" "BUILD_JOBS=$BUILD_JOBS")
COMMON_BUILD_FLAGS+=("--build-arg" "TORCH_CUDA_ARCH_LIST=$GPU_ARCH_LIST")

# =====================================================
# Build image
# =====================================================
VLLM_BUILD_TIME=0
RUNNER_BUILD_TIME=0

# ----------------------------------------------------------
# Phase 1: vLLM wheels
# ----------------------------------------------------------
VLLM_WHEELS_EXIST=false
if compgen -G "./wheels/vllm*.whl" > /dev/null 2>&1; then
    VLLM_WHEELS_EXIST=true
fi

if requires_source_build; then
    # ---- Source build path (SM 12.x+, custom PRs/ref) ----
    echo "Source build required (GPU arch: $GPU_ARCH_LIST)"

    if [ "$VLLM_REF_SET" = true ] || [ -n "$VLLM_PRS" ]; then
        REBUILD_VLLM=true
    fi

    # Check if upstream has new commits
    if [ "$REBUILD_VLLM" = false ] && [ "$VLLM_WHEELS_EXIST" = true ]; then
        VLLM_REMOTE_SHA=$(resolve_remote_sha "$VLLM_REPO" "$VLLM_REF")
        VLLM_LOCAL_SHA=""
        [ -f "./wheels/.vllm-sha" ] && VLLM_LOCAL_SHA=$(cat "./wheels/.vllm-sha")
        if [ -n "$VLLM_REMOTE_SHA" ] && [ "$VLLM_REMOTE_SHA" != "$VLLM_LOCAL_SHA" ]; then
            echo "vLLM has upstream changes (${VLLM_REMOTE_SHA:0:8}) — rebuilding..."
            REBUILD_VLLM=true
        fi
    fi

    if [ "$REBUILD_VLLM" = true ] || [ "$VLLM_WHEELS_EXIST" = false ]; then
        if [ "$REBUILD_VLLM" = true ]; then
            if [ "$VLLM_REF_SET" = true ] && [ -n "$VLLM_PRS" ]; then
                echo "Rebuilding vLLM wheels (--vllm-ref and --apply-vllm-pr specified)..."
            elif [ "$VLLM_REF_SET" = true ]; then
                echo "Rebuilding vLLM wheels (--vllm-ref specified)..."
            elif [ -n "$VLLM_PRS" ]; then
                echo "Rebuilding vLLM wheels (--apply-vllm-pr specified)..."
            else
                echo "Rebuilding vLLM wheels (--rebuild-vllm specified)..."
            fi
        else
            echo "No vLLM wheels found in ./wheels/ — building..."
        fi

        # Back up existing vllm wheels; restore them if the build fails
        VLLM_BACKUP="./wheels/.backup-vllm"
        rm -rf "$VLLM_BACKUP" && mkdir -p "$VLLM_BACKUP"
        for f in ./wheels/vllm*.whl; do
            [ -f "$f" ] && mv "$f" "$VLLM_BACKUP/"
        done

        VLLM_CMD=("docker" "build"
            "--target" "vllm-export"
            "--output" "type=local,dest=./wheels"
            "--no-cache-filter" "vllm-builder"
            "${COMMON_BUILD_FLAGS[@]}"
            "--build-arg" "VLLM_REF=$VLLM_REF")

        if [ -n "$VLLM_PRS" ]; then
            echo "Applying vLLM PRs: $VLLM_PRS"
            VLLM_CMD+=("--build-arg" "VLLM_PRS=$VLLM_PRS")
        fi

        VLLM_CMD+=(".")

        echo "vLLM build command: ${VLLM_CMD[*]}"
        VLLM_START=$(date +%s)
        if "${VLLM_CMD[@]}"; then
            VLLM_END=$(date +%s)
            VLLM_BUILD_TIME=$((VLLM_END - VLLM_START))
            rm -rf "$VLLM_BACKUP"
            # Save the SHA we built from
            VLLM_REMOTE_SHA=${VLLM_REMOTE_SHA:-$(resolve_remote_sha "$VLLM_REPO" "$VLLM_REF")}
            [ -n "$VLLM_REMOTE_SHA" ] && echo "$VLLM_REMOTE_SHA" > ./wheels/.vllm-sha
        else
            echo "vLLM build failed — restoring previous wheels..."
            mv "$VLLM_BACKUP"/vllm*.whl ./wheels/ 2>/dev/null || true
            rm -rf "$VLLM_BACKUP"
            exit 1
        fi
    else
        echo "vLLM wheels are up to date."
    fi
else
    # ---- Prebuilt path (standard GPU archs, no custom patches) ----
    if [ "$REBUILD_VLLM" = true ] || [ "$VLLM_WHEELS_EXIST" = false ]; then
        echo "Downloading prebuilt vLLM wheel from PyPI..."
        rm -f ./wheels/vllm*.whl
        rm -f ./wheels/.vllm-sha
        VLLM_START=$(date +%s)
        pip download vllm --no-deps --only-binary :all: -d ./wheels/ \
            --python-version 3.12 \
            --platform "manylinux_2_35_$(uname -m)"
        VLLM_END=$(date +%s)
        VLLM_BUILD_TIME=$((VLLM_END - VLLM_START))
        echo "Downloaded: $(ls ./wheels/vllm*.whl | xargs -n1 basename)"
    else
        echo "vLLM prebuilt wheel is present."
    fi
fi

# ----------------------------------------------------------
# Phase 2: Runner image
# ----------------------------------------------------------
if ! compgen -G "./wheels/*.whl" > /dev/null 2>&1; then
    echo "Error: No wheel files found in ./wheels/ — cannot build runner image."
    exit 1
fi

RUNNER_CMD=("docker" "build"
    "-t" "$IMAGE_TAG"
    "${COMMON_BUILD_FLAGS[@]}"
    ".")

echo "Building runner image with command: ${RUNNER_CMD[*]}"
RUNNER_START=$(date +%s)
"${RUNNER_CMD[@]}"
RUNNER_END=$(date +%s)
RUNNER_BUILD_TIME=$((RUNNER_END - RUNNER_START))

# Calculate total time
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))

# Display timing statistics
echo ""
echo "========================================="
echo "         TIMING STATISTICS"
echo "========================================="
if [ "$VLLM_BUILD_TIME" -gt 0 ]; then
    echo "vLLM Build:       $(printf '%02d:%02d:%02d' $((VLLM_BUILD_TIME/3600)) $((VLLM_BUILD_TIME%3600/60)) $((VLLM_BUILD_TIME%60)))"
fi
if [ "$RUNNER_BUILD_TIME" -gt 0 ]; then
    echo "Runner Build:     $(printf '%02d:%02d:%02d' $((RUNNER_BUILD_TIME/3600)) $((RUNNER_BUILD_TIME%3600/60)) $((RUNNER_BUILD_TIME%60)))"
fi
echo "Total Time:       $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo "========================================="
echo "Done building $IMAGE_TAG."
