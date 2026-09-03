#!/usr/bin/env bash
set -Eeuo pipefail
set +x
umask 077

: "${IMAGE:?set IMAGE to the native image tag or digest}"
: "${MODEL_PATH:?set MODEL_PATH to the model directory}"
: "${API_KEY_FILE:?set API_KEY_FILE to a readable key file}"

CONTAINER_NAME=${CONTAINER_NAME:-deepseek-v4-vision-cmp170hx-pp4}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
HOST_PORT=${HOST_PORT:-9016}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-1048576}
PP_PARTITION=${PP_PARTITION:-11,11,12,9}

test -d "$MODEL_PATH"
test -r "$API_KEY_FILE"
test -s "$API_KEY_FILE"
docker image inspect "$IMAGE" >/dev/null
! docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1
if ss -ltnH "sport = :$HOST_PORT" | grep -q .; then
  echo "port $HOST_PORT is already in use" >&2
  exit 73
fi

api_key=$(tr -d '\r\n' <"$API_KEY_FILE")
test -n "$api_key"
trap 'unset api_key' EXIT

container_id=$(docker run -d \
  --name "$CONTAINER_NAME" \
  --restart=unless-stopped \
  --runtime=nvidia \
  --gpus "device=$GPU_DEVICES" \
  --ipc=host \
  --shm-size=32g \
  -p "$HOST_PORT:9016" \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e VLLM_MARLIN_FP8_DEQUANT_BF16=1 \
  -e "VLLM_PP_LAYER_PARTITION=$PP_PARTITION" \
  -e VLLM_DSPARK_FUSED_MARKOV=1 \
  -e VLLM_PREFILL_BLOCK_H=8 \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_SPARSE_DENSE_QUERY_BLOCK=4 \
  -e VLLM_PP_COMM_PRIME=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_CUMEM_ENABLE=1 \
  -e HF_HUB_OFFLINE=1 \
  -v "$MODEL_PATH:/models/dsv4-vision:ro" \
  "$IMAGE" \
  /opt/venv/bin/vllm serve /models/dsv4-vision \
    --served-model-name DeepSeek-V4-Flash-Vision-Exp \
    --host 0.0.0.0 \
    --port 9016 \
    --api-key "$api_key" \
    --trust-remote-code \
    --pipeline-parallel-size 4 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --block-size 256 \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    --tokenizer-mode deepseek_v4 \
    --enable-auto-tool-choice \
    --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --generation-config vllm \
    --override-generation-config '{"temperature":1.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0}' \
    --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
    --enable-prompt-tokens-details \
    --seed 1101 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic"}')

unset api_key
trap - EXIT
echo "container=$CONTAINER_NAME id=$container_id"
echo "waiting for http://127.0.0.1:$HOST_PORT/health"

deadline=$((SECONDS + 2700))
while (( SECONDS < deadline )); do
  state=$(docker inspect "$CONTAINER_NAME" --format '{{.State.Status}}')
  if [[ "$state" == exited || "$state" == dead ]]; then
    docker logs --tail 100 "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  if curl --fail --silent --max-time 5 \
      "http://127.0.0.1:$HOST_PORT/health" >/dev/null 2>&1; then
    echo "service healthy after ${SECONDS}s"
    exit 0
  fi
  sleep 10
done

echo "service did not become healthy before the deadline" >&2
exit 1
