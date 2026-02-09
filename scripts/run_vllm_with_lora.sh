#!/bin/bash

# LoRA 어댑터를 적용한 vLLM 서버 실행 스크립트

# 설정
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2-7B-Instruct}"
LORA_PATH="${LORA_PATH:-./models/qwen2-emotion-lora}"
GPU_ID="${GPU_ID:-3}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

echo "🚀 LoRA 적용 vLLM 서버 시작"
echo "  베이스 모델: $BASE_MODEL"
echo "  LoRA 어댑터: $LORA_PATH"
echo "  GPU: $GPU_ID"
echo "  포트: $PORT"
echo ""

# GPU 설정
export CUDA_VISIBLE_DEVICES=$GPU_ID

# vLLM 서버 실행 (LoRA 어댑터 활성화)
python -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL" \
    --enable-lora \
    --lora-modules emotion-lora="$LORA_PATH" \
    --max-lora-rank 16 \
    --port "$PORT" \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization 0.9 \
    --dtype bfloat16

# 사용 방법:
# 1. 기본 실행 (GPU 3번, 포트 8000)
#    bash scripts/run_vllm_with_lora.sh
#
# 2. 커스텀 설정
#    GPU_ID=2 PORT=8001 LORA_PATH=./models/my-lora bash scripts/run_vllm_with_lora.sh
#
# 3. LoRA 어댑터 사용 API 호출
#    curl -X POST http://localhost:8000/v1/chat/completions \
#      -H "Content-Type: application/json" \
#      -d '{
#        "model": "emotion-lora",  # LoRA 어댑터 이름
#        "messages": [
#          {"role": "system", "content": "당신은 공감적인 상담사입니다."},
#          {"role": "user", "content": "[감정: sad] 요즘 너무 힘들어요."}
#        ]
#      }'
