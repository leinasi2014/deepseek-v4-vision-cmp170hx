# DeepSeek V4 Vision for CMP 170HX

[中文说明](README.zh-CN.md) · [Architecture](docs/cmp170hx/ARCHITECTURE.md) ·
[Deployment](deploy/README.md) · [Validation](docs/cmp170hx/VALIDATION.md) ·
[Test data](benchmarks/results/cmp170hx-pp4-fix7-2026-09-02/README.md)

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
The sanitized per-request and per-GPU measurements are available in the
[machine-readable test dataset](benchmarks/results/cmp170hx-pp4-fix7-2026-09-02/README.md).

## 2026-09-03 optimization update

A factor sweep on a second 4-card group validated `NCCL_P2P_LEVEL=SYS` as the
headline win (+27% single-stream decode; SYS is the permissive P2P distance
ceiling per official NCCL semantics and engages P2P on NODE-level pairs the
default PIX threshold skips). **Production recommendation: `SYS` +
`--max-num-batched-tokens 4096`** (99.1 tok/s single-stream on the test group).
`SYS + bat2048` measured even better prefill/C16 but **triggered an Xid-31
crash on the drafter rank after 67 minutes of live traffic** — see the incident
note in
[OPTIMIZATION-2026-09-03.md](docs/cmp170hx/OPTIMIZATION-2026-09-03.md) before
considering it; it requires a long soak. Full method, 14-arm matrix, plan-2
engine comparison, and the Xid/memory/power safety ledger are in the same doc.

## Latest Fix7 benchmark

Measured on 2026-09-02 with the validated PP4 profile above. Every arm used
the same model, hardware, prompts, DSpark settings, and requested output
length. The ordinary decode result is the mean post-TTFT rate from 12 measured
requests per arm, each forced to exactly 400 completion tokens. Warm-up was
excluded.

| Arm | `BLOCK_H` | CUDA graph | Mean decode | Exact 10K prefill | Decode after 10K | DSpark acceptance |
|---|---:|---|---:|---:|---:|---:|
| B0 | 0 | breakable | 75.13 tok/s | 3834.54 tok/s | 82.57 tok/s | 27.43% |
| **B8** | **8** | **breakable** | **77.30 tok/s** | **4224.96 tok/s** | **92.54 tok/s** | **28.16%** |
| CG | 8 | explicit bounded | 73.52 tok/s | 2731.47 tok/s | 106.51 tok/s | 26.63% |
| CGR | 8 | explicit bounded + Ring/Simple | 72.96 tok/s | 2818.12 tok/s | 106.21 tok/s | 26.23% |

Recommended B8 details:

| Measurement | Result |
|---|---:|
| Fixed-token requests | 12 x 400 completion tokens |
| Mean E2E completion rate | 75.07 tok/s |
| Mean post-TTFT decode | 77.30 tok/s |
| Chinese post-TTFT mean | 78.19 tok/s |
| English post-TTFT mean | 76.41 tok/s |
| Exact 10K prefill | 4224.96 prompt tok/s |
| 192-token decode after exact 10K context | 92.54 tok/s |
| DSpark weighted acceptance | 28.16% (2802 / 9950) |
| Startup time | 444 s |

B8 mean GPU utilization across PP stages was 87.7%, 83.3%, 65.4%, and
47.8%; the corresponding peak temperatures were 75°C, 69°C, 65°C, and 62°C.
All four arms passed the 1M service-contract, exact-400-token, exact-10K,
vision, invalid-image HTTP 400, fatal-log, and P2P/CUMEM gates. A full 1M
request was not run. See the [complete 48-request CSV and GPU data](benchmarks/results/cmp170hx-pp4-fix7-2026-09-02/README.md)
for every sample and the metric definitions.

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
