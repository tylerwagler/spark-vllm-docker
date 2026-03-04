#!/bin/bash
# Check if locally cached HuggingFace models are up to date

UPDATE=false
if [ "$1" = "--update" ]; then
    UPDATE=true
fi

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  --update    Download outdated models after checking"
    echo "  -h, --help  Show this help message"
    exit 0
}

case "${1:-}" in
    --update) UPDATE=true ;;
    -h|--help) usage ;;
    "") ;;
    *) echo "Unknown option: $1"; usage ;;
esac

HUB_PATH="${HF_HOME:-$HOME/.cache/huggingface}/hub"

if [ ! -d "$HUB_PATH" ]; then
    echo "No HuggingFace cache found at $HUB_PATH"
    exit 1
fi

OUTDATED=0
CHECKED=0
OUTDATED_MODELS=()

for model_dir in "$HUB_PATH"/models--*; do
    [ -d "$model_dir" ] || continue

    # Parse model ID from directory name (models--org--name → org/name)
    dir_name=$(basename "$model_dir")
    model_id="${dir_name#models--}"
    model_id="${model_id/--//}"

    # Get local ref
    local_sha=""
    if [ -f "$model_dir/refs/main" ]; then
        local_sha=$(cat "$model_dir/refs/main")
    fi

    if [ -z "$local_sha" ]; then
        echo "  ? $model_id (no local ref)"
        continue
    fi

    # Get remote ref
    remote_sha=$(git ls-remote "https://huggingface.co/$model_id" refs/heads/main 2>/dev/null | cut -f1)

    if [ -z "$remote_sha" ]; then
        echo "  ? $model_id (could not reach remote)"
        continue
    fi

    CHECKED=$((CHECKED + 1))

    if [ "$local_sha" = "$remote_sha" ]; then
        echo "  ✓ $model_id"
    else
        echo "  ✗ $model_id (local: ${local_sha:0:8}, remote: ${remote_sha:0:8})"
        OUTDATED=$((OUTDATED + 1))
        OUTDATED_MODELS+=("$model_id")
    fi
done

echo ""
echo "$CHECKED models checked, $OUTDATED outdated."

if [ "$OUTDATED" -gt 0 ]; then
    if [ "$UPDATE" = true ]; then
        echo ""
        echo "Downloading ${#OUTDATED_MODELS[@]} outdated model(s)..."
        for model_id in "${OUTDATED_MODELS[@]}"; do
            echo ""
            echo "--- Updating $model_id ---"
            huggingface-cli download "$model_id" || echo "  Failed to update $model_id"
        done
    else
        echo "Run '$0 --update' to download outdated models."
    fi
fi
