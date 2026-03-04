#!/bin/bash
set -e
patch -p1 -N -d /usr/local/lib/python3.12/dist-packages < transformers.patch || true

# Fix vLLM OpaqueBase import crash on NVIDIA PyTorch builds
# NVIDIA's PyTorch 2.11.0a0 reports as >= 2.11.0.dev but lacks torch._opaque_base
TORCH_UTILS="/usr/local/lib/python3.12/dist-packages/vllm/utils/torch_utils.py"
if [ -f "$TORCH_UTILS" ] && python3 -c "from torch._opaque_base import OpaqueBase" 2>/dev/null; then
    echo "torch._opaque_base available, no patch needed"
else
    echo "Patching vLLM torch_utils.py: disabling OpaqueBase (not in this PyTorch build)"
    sed -i 's/^HAS_OPAQUE_TYPE = .*/HAS_OPAQUE_TYPE = False/' "$TORCH_UTILS"
fi