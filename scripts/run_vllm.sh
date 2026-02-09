#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${OPENAI_MODEL_NAME:-Qwen2-7B-Instruct}"
PORT="${PORT:-8000}"

echo "Starting vLLM with model=${MODEL_NAME} on port=${PORT}"
python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --served-model-name "${MODEL_NAME}"


