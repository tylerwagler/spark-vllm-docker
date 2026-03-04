#!/bin/bash
set -e

# Start total time tracking
START_TIME=$(date +%s)

# Default values
IMAGE_TAG="vllm-node"
REBUILD_FLASHINFER=false
REBUILD_VLLM=false
VLLM_REF="main"
VLLM_REF_SET=false
VLLM_PRS=""
FULL_LOG=false
BUILD_JOBS="16"
GPU_ARCH_LIST=""
WHEELS_REPO="eugr/spark-vllm-docker"
FLASHINFER_RELEASE_TAG="prebuilt-flashinfer-current"
# Space-separated list of GPU architectures for which prebuilt wheels are available
PREBUILT_WHEELS_SUPPORTED_ARCHS="12.1a"

# try_download_wheels TAG PREFIX
# Downloads wheels matching PREFIX*.whl from a GitHub release.
# Skips files that are already present and up to date (by remote updated_at vs local mtime).
# Returns 0 if all matching wheels are now available, 1 on any error.
try_download_wheels() {
    local TAG="$1"
    local PREFIX="$2"
    local WHEELS_DIR="./wheels"

    local arch
    for arch in $PREBUILT_WHEELS_SUPPORTED_ARCHS; do
        [ "$arch" = "$GPU_ARCH_LIST" ] && break
        arch=""
    done
    if [ -z "$arch" ]; then
        echo "GPU arch '$GPU_ARCH_LIST' not supported by prebuilt wheels (supported: $PREBUILT_WHEELS_SUPPORTED_ARCHS) — skipping download."
        return 1
    fi

    local RELEASE_JSON
    RELEASE_JSON=$(curl -sf --connect-timeout 10 \
        "https://api.github.com/repos/$WHEELS_REPO/releases/tags/$TAG") || {
        echo "Could not fetch release metadata for '$TAG' — skipping download."
        return 1
    }

    local DOWNLOAD_LIST
    DOWNLOAD_LIST=$(echo "$RELEASE_JSON" | python3 -c '
import json, sys, os
from datetime import datetime, timezone

wheels_dir, prefix = sys.argv[1], sys.argv[2]
data = json.load(sys.stdin)
assets = [a for a in data.get("assets", [])
          if a["name"].startswith(prefix) and a["name"].endswith(".whl")]

if not assets:
    print("No assets found matching prefix: " + prefix, file=sys.stderr)
    sys.exit(1)

for a in assets:
    local_path = os.path.join(wheels_dir, a["name"])
    remote_ts = datetime.strptime(a["updated_at"], "%Y-%m-%dT%H:%M:%SZ") \
                    .replace(tzinfo=timezone.utc).timestamp()
    if not os.path.exists(local_path) or remote_ts > os.path.getmtime(local_path):
        print(a["browser_download_url"] + " " + a["name"])
' "$WHEELS_DIR" "$PREFIX") || return 1

    if [ -z "$DOWNLOAD_LIST" ]; then
        echo "All $PREFIX wheels are up to date — skipping download."
        return 0
    fi

    # Back up existing wheels so we never leave a mix of old and new on failure
    local DL_BACKUP="$WHEELS_DIR/.backup-download-${PREFIX}"
    rm -rf "$DL_BACKUP" && mkdir -p "$DL_BACKUP"
    for f in "$WHEELS_DIR/${PREFIX}"*.whl; do
        [ -f "$f" ] && mv "$f" "$DL_BACKUP/"
    done

    local URL NAME TMP_WHL
    local DOWNLOADED=()
    while IFS=' ' read -r URL NAME; do
        echo "Downloading $NAME..."
        TMP_WHL=$(mktemp "$WHEELS_DIR/${NAME}.XXXXXX")
        if curl -L --progress-bar --connect-timeout 30 "$URL" -o "$TMP_WHL"; then
            mv "$TMP_WHL" "$WHEELS_DIR/$NAME"
            DOWNLOADED+=("$WHEELS_DIR/$NAME")
        else
            rm -f "$TMP_WHL"
            echo "Failed to download $NAME — removing other downloaded files."
            for f in "${DOWNLOADED[@]}"; do rm -f "$f"; done
            if compgen -G "$DL_BACKUP/${PREFIX}*.whl" > /dev/null 2>&1; then
                echo "Restoring previous $PREFIX wheels..."
                mv "$DL_BACKUP/${PREFIX}"*.whl "$WHEELS_DIR/"
            fi
            rm -rf "$DL_BACKUP"
            return 1
        fi
    done <<< "$DOWNLOAD_LIST"

    rm -rf "$DL_BACKUP"
    return 0
}

# Help function
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -t, --tag <tag>               : Image tag (default: 'vllm-node')"
    echo "  --gpu-arch <arch>             : GPU architecture (default: auto-detect from nvidia-smi)"
    echo "  --rebuild-flashinfer          : Force rebuild of FlashInfer wheels (ignore cached wheels)"
    echo "  --rebuild-vllm                : Force rebuild of vLLM wheels (ignore cached wheels)"
    echo "  --vllm-ref <ref>              : vLLM commit SHA, branch or tag (default: 'main')"
    echo "  -j, --build-jobs <jobs>       : Number of concurrent build jobs (default: ${BUILD_JOBS})"
    echo "  --apply-vllm-pr <pr-num>      : Apply a specific PR patch to vLLM source. Can be specified multiple times."
    echo "  --full-log                    : Enable full build logging (--progress=plain)"
    echo "  -h, --help                    : Show this help message"
    exit 1
}

# Argument parsing
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -t|--tag) IMAGE_TAG="$2"; shift ;;
        --gpu-arch) GPU_ARCH_LIST="$2"; shift ;;
        --rebuild-flashinfer) REBUILD_FLASHINFER=true ;;
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

# Common build flags
COMMON_BUILD_FLAGS=()
if [ "$FULL_LOG" = true ]; then
    COMMON_BUILD_FLAGS+=("--progress=plain")
fi
COMMON_BUILD_FLAGS+=("--build-arg" "BUILD_JOBS=$BUILD_JOBS")
COMMON_BUILD_FLAGS+=("--build-arg" "TORCH_CUDA_ARCH_LIST=$GPU_ARCH_LIST")
COMMON_BUILD_FLAGS+=("--build-arg" "FLASHINFER_CUDA_ARCH_LIST=$GPU_ARCH_LIST")

# =====================================================
# Build image
# =====================================================
FLASHINFER_BUILD_TIME=0
VLLM_BUILD_TIME=0
RUNNER_BUILD_TIME=0

# ----------------------------------------------------------
# Phase 1: FlashInfer wheels
# ----------------------------------------------------------
BUILD_FLASHINFER=false
if [ "$REBUILD_FLASHINFER" = true ]; then
    echo "Rebuilding FlashInfer wheels (--rebuild-flashinfer specified)..."
    BUILD_FLASHINFER=true
elif try_download_wheels "$FLASHINFER_RELEASE_TAG" "flashinfer"; then
    echo "FlashInfer wheels ready."
elif compgen -G "./wheels/flashinfer*.whl" > /dev/null 2>&1; then
    echo "Download failed — using existing local FlashInfer wheels."
else
    echo "No FlashInfer wheels available (download failed) — building..."
    BUILD_FLASHINFER=true
fi

if [ "$BUILD_FLASHINFER" = true ]; then
    # Back up existing flashinfer wheels; restore them if the build fails
    FI_BACKUP="./wheels/.backup-flashinfer"
    rm -rf "$FI_BACKUP" && mkdir -p "$FI_BACKUP"
    for f in ./wheels/flashinfer*.whl; do
        [ -f "$f" ] && mv "$f" "$FI_BACKUP/"
    done

    FI_CMD=("docker" "build"
        "--target" "flashinfer-export"
        "--output" "type=local,dest=./wheels"
        "${COMMON_BUILD_FLAGS[@]}")

    if [ "$REBUILD_FLASHINFER" = true ]; then
        FI_CMD+=("--build-arg" "CACHEBUST_FLASHINFER=$(date +%s)")
    fi

    FI_CMD+=(".")

    echo "FlashInfer build command: ${FI_CMD[*]}"
    FI_START=$(date +%s)
    if "${FI_CMD[@]}"; then
        FI_END=$(date +%s)
        FLASHINFER_BUILD_TIME=$((FI_END - FI_START))
        rm -rf "$FI_BACKUP"
    else
        echo "FlashInfer build failed — restoring previous wheels..."
        mv "$FI_BACKUP"/flashinfer*.whl ./wheels/ 2>/dev/null || true
        rm -rf "$FI_BACKUP"
        exit 1
    fi
fi

# ----------------------------------------------------------
# Phase 2: vLLM wheels
# ----------------------------------------------------------
VLLM_WHEELS_EXIST=false
if compgen -G "./wheels/vllm*.whl" > /dev/null 2>&1; then
    VLLM_WHEELS_EXIST=true
fi

if [ "$VLLM_REF_SET" = true ] || [ -n "$VLLM_PRS" ]; then
    REBUILD_VLLM=true
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
        "${COMMON_BUILD_FLAGS[@]}"
        "--build-arg" "VLLM_REF=$VLLM_REF")

    if [ "$REBUILD_VLLM" = true ]; then
        VLLM_CMD+=("--build-arg" "CACHEBUST_VLLM=$(date +%s)")
    fi

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
    else
        echo "vLLM build failed — restoring previous wheels..."
        mv "$VLLM_BACKUP"/vllm*.whl ./wheels/ 2>/dev/null || true
        rm -rf "$VLLM_BACKUP"
        exit 1
    fi
else
    echo "vLLM wheels already present in ./wheels/ — skipping build."
fi

# ----------------------------------------------------------
# Phase 3: Runner image
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
if [ "$FLASHINFER_BUILD_TIME" -gt 0 ]; then
    echo "FlashInfer Build: $(printf '%02d:%02d:%02d' $((FLASHINFER_BUILD_TIME/3600)) $((FLASHINFER_BUILD_TIME%3600/60)) $((FLASHINFER_BUILD_TIME%60)))"
fi
if [ "$VLLM_BUILD_TIME" -gt 0 ]; then
    echo "vLLM Build:       $(printf '%02d:%02d:%02d' $((VLLM_BUILD_TIME/3600)) $((VLLM_BUILD_TIME%3600/60)) $((VLLM_BUILD_TIME%60)))"
fi
if [ "$RUNNER_BUILD_TIME" -gt 0 ]; then
    echo "Runner Build:     $(printf '%02d:%02d:%02d' $((RUNNER_BUILD_TIME/3600)) $((RUNNER_BUILD_TIME%3600/60)) $((RUNNER_BUILD_TIME%60)))"
fi
echo "Total Time:       $(printf '%02d:%02d:%02d' $((TOTAL_TIME/3600)) $((TOTAL_TIME%3600/60)) $((TOTAL_TIME%60)))"
echo "========================================="
echo "Done building $IMAGE_TAG."
