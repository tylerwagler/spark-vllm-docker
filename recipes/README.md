# Recipes

Reference configurations for running models with vLLM. Each YAML file documents the model, required mods, environment variables, and vLLM serve arguments.

## Available Recipes

| Recipe | Model | Notes |
| :--- | :--- | :--- |
| `glm-4.7-flash-awq` | cyankiwi/GLM-4.7-Flash-AWQ-4bit | Requires `fix-glm-4.7-flash-AWQ` mod |
| `minimax-m2-awq` | QuantTrio/MiniMax-M2-AWQ | Multi-GPU (TP=2) |
| `minimax-m2.5-awq` | cyankiwi/MiniMax-M2.5-AWQ-4bit | Multi-GPU (TP=2) |
| `nemotron-3-nano-nvfp4` | nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4 | Requires `nemotron-nano` mod |
| `openai-gpt-oss-120b` | openai/gpt-oss-120b | MXFP4 quantization |
| `qwen3-coder-next-fp8` | Qwen/Qwen3-Coder-Next-FP8 | Requires `fix-qwen3-coder-next` mod |
| `qwen3.5-122b-int4-autoround` | Intel/Qwen3.5-122B-A10B-int4-AutoRound | Requires `fix-qwen3.5-autoround` mod |

## Usage

These recipes are reference configs — read the YAML to extract the vLLM command and arguments for your own `docker run` or `docker-compose.yml`.

Example from `glm-4.7-flash-awq.yaml`:

```bash
docker run --privileged --gpus all -it --rm \
  --network host --ipc=host \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/mods:/workspace/mods \
  vllm-node bash -c "\
    /workspace/mods/fix-glm-4.7-flash-AWQ/run.sh && \
    vllm serve cyankiwi/GLM-4.7-Flash-AWQ-4bit \
      --port 8000 --host 0.0.0.0 \
      --tool-call-parser glm47 \
      --reasoning-parser glm45 \
      --enable-auto-tool-choice \
      --max-model-len 202752 \
      --gpu-memory-utilization 0.7 \
      --load-format fastsafetensors"
```
