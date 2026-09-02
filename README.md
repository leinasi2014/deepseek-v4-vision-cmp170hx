# DeepSeek V4 Vision for CMP 170HX

[中文说明](README.zh-CN.md) · [Architecture](docs/cmp170hx/ARCHITECTURE.md) ·
[Deployment](deploy/README.md) · [Validation](docs/cmp170hx/VALIDATION.md)

An experimental vLLM fork for serving
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on NVIDIA CMP 170HX / SM80 GPUs.
It combines the fast SM80 DeepSeek-V4 path with the newer multimodal model,
Pipeline Parallel DSpark decoding, FP8 DeepSeek MLA KV cache, and a 1M context
configuration.

This repository is a hardware-specific research fork, not an official DeepSeek
or vLLM release.

## Validated profile

- Hardware: 4 × NVIDIA CMP 170HX (SM80), working CUDA peer access
- Parallelism: PP4 × TP1
- Model: `DeepSeek-V4-Flash-Vision-Exp`
- Context contract: `1,048,576` tokens
- KV cache: `fp8_ds_mla`
- Speculative decoding: DSpark, 5 draft tokens, probabilistic sampling
- Recommended SM80 prefill setting: `VLLM_PREFILL_BLOCK_H=8`
- CUDA graph mode: breakable CUDA graph

The latest validation source is published as commit
`f00f0eecc41146f79f0545c15612962b66b693c5`. The complete fixed-token test
method and results are documented in [VALIDATION.md](docs/cmp170hx/VALIDATION.md).

## What this fork adds

- DeepSeek-V4 Flash sparse MLA kernels and Ampere/SM80 routing.
- DeepSeek-V4 Flash Vision model, processor, vision encoder, and image-token
  routing.
- PP-safe DSpark proposal propagation and rank-consistent gating.
- 1M-context sparse-index and speculative decoding correctness fixes.
- Position-independent, pad-free multimodal cache blocks.
- Fail-closed image placeholder/embedding validation.
- OpenAI-compatible image-role validation before multimodal tracking.
- CMP170HX/A100 build tuning and a measured optimization record.

See [ARCHITECTURE.md](docs/cmp170hx/ARCHITECTURE.md) for the port layers and
design boundaries.

## Quick start

Prerequisites:

1. Linux x86_64, Docker, and NVIDIA Container Toolkit.
2. Four SM80 GPUs with verified CUDA peer correctness and NCCL P2P transport.
3. A native image built from this exact source tree.
4. Model weights stored outside this repository.

Launch the validated PP4 profile:

```bash
export IMAGE=dsv4-cmp170hx:local
export MODEL_PATH=/path/to/DeepSeek-V4-Flash-Vision-Exp
export API_KEY_FILE=/run/secrets/dsv4-api-key
bash deploy/serve-pp4.sh
```

Then use the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:9016/v1/models \
  -H "Authorization: Bearer $(< /run/secrets/dsv4-api-key)"
```

The deployment script never contains a built-in API key. Review
[deploy/README.md](deploy/README.md) before exposing the endpoint to a network.

## Build

The proven native build recipe is
[`docker/Dockerfile.cmp170hx-sm80`](docker/Dockerfile.cmp170hx-sm80). It builds
for `TORCH_CUDA_ARCH_LIST=8.0` and requires the pinned external CUTLASS/Triton
source bundle described in [docker/README.md](docker/README.md).

Do not combine Python files from this fork with native extensions from another
commit. The Python wrappers and compiled CUDA/libtorch ABI must be shipped as
one immutable image.

## Important boundaries

- No model weights are included. Follow the model publisher's license and
  access terms.
- No NVIDIA driver, firmware, or CMP 170HX unlock patch is included.
- `max_model_len=1048576` was verified at service startup. The published
  performance matrix intentionally used exact 10K prompts rather than a full
  1M request.
- The current PP4 single-stream profile does not produce equal utilization on
  all four pipeline stages. P2P can be healthy while PP utilization remains
  asymmetric.
- Upstream vLLM supports many platforms; this fork only claims the tested SM80
  DeepSeek-V4 profile above.

## Provenance

This project is derived from [vLLM](https://github.com/vllm-project/vllm) and
retains its Apache-2.0 license. The CMP170HX/DeepSeek work also builds on public
research from:

- [allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx)
- [tim-odonnell/cmp170hx-deepseek-v4-flash](https://github.com/tim-odonnell/cmp170hx-deepseek-v4-flash)
- [satspace-cpu/cmp170hx-linux-p2p](https://github.com/satspace-cpu/cmp170hx-linux-p2p)

See [NOTICE](NOTICE) and the Git history for attribution.

## License

Apache License 2.0. See [LICENSE](LICENSE).
