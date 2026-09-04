#!/bin/bash
# DSV4-Vision PP8 test: 8-card pipeline parallel, faithful env set (nothing
# dropped -- that omission was the Xid31 root cause), overlay7 image
# (2 hardening patches + solution-B), bat2048, port 9016.
set -e
NAME=dsv4-vision-pp8-test
IMG=dsv4-a100:oldfast-vision-fix7-chatimg-overlay7

docker rm -f "$NAME" 2>/dev/null || true

BUSY=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, '$2>100' | wc -l)
if [ "$BUSY" -ne 0 ]; then echo "GATE_FAIL: GPUs not free"; exit 1; fi

docker run -d --name "$NAME" \
  --restart no --runtime=nvidia --ipc=host \
  -p 9016:9016 \
  -e NVIDIA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -e NCCL_IB_DISABLE=1 \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_DEBUG=INFO \
  -e NCCL_DEBUG_SUBSYS=INIT,GRAPH \
  -e VLLM_USE_BREAKABLE_CUDAGRAPH=1 \
  -e VLLM_DSPARK_FUSED_MARKOV=1 \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e NCCL_CUMEM_ENABLE=1 \
  -e VLLM_PP_COMM_PRIME=0 \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_SPARSE_DENSE_QUERY_BLOCK=4 \
  -e VLLM_PREFILL_BLOCK_H=8 \
  -e DSV4_LOGITS_ROW_CHUNK=64 \
  -e VLLM_MARLIN_FP8_DEQUANT_BF16=1 \
  -v /samsungssd/models/DeepSeek-V4-Flash-Vision-Exp:/models/dsv4-vision:ro \
  --entrypoint /opt/nvidia/nvidia_entrypoint.sh \
  "$IMG" \
  /opt/venv/bin/vllm serve /models/dsv4-vision \
    --served-model-name DeepSeek-V4-Flash-Vision-Exp \
    --host 0.0.0.0 --port 9016 --api-key sk_344303 \
    --trust-remote-code \
    --pipeline-parallel-size 8 --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 --block-size 256 \
    --max-model-len 1048576 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.85 \
    --enable-prefix-caching \
    --no-enable-flashinfer-autotune \
    --tokenizer-mode deepseek_v4 \
    --enable-auto-tool-choice --tool-call-parser deepseek_v4 \
    --reasoning-parser deepseek_v4 \
    --generation-config vllm \
    --override-generation-config '{"temperature":1.0,"top_p":1.0,"top_k":-1,"repetition_penalty":1.0}' \
    --default-chat-template-kwargs '{"thinking":true,"reasoning_effort":"high"}' \
    --enable-prompt-tokens-details \
    --seed 1101 \
    --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_tensor_parallel_size":1,"draft_sample_method":"probabilistic"}'

echo "PP8_LAUNCHED $(date +%T)"
