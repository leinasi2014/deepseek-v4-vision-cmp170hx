# Validation record

## Scope

Date: 2026-09-02

Hardware: four NVIDIA CMP 170HX GPUs (SM80), PP4 × TP1.

Model: `DeepSeek-V4-Flash-Vision-Exp`.

Published source: `f00f0eecc41146f79f0545c15612962b66b693c5`. This is tree-equivalent to
the server-side shallow validation commit
`f09e0305445db93f628ce2ec71378a55370afa8e`; commit IDs changed when the
missing shallow parent boundary was converted into a publishable root history.

Every arm used the same model, PP partition, DSpark configuration, FP8 KV
cache, 1M service contract, prompts, and requested completion lengths.

## Correctness gates

- GPU unit gate passed.
- Service reported `max_model_len=1048576`.
- Twelve measured requests per arm returned exactly 400 completion tokens.
- Exact 10,000-token prefill and 10,000-token context decode completed.
- One-image smoke and visual brand check passed.
- One image passed at four different prompt-token alignment offsets.
- Structured tool images returned HTTP 400 rather than EngineCore 500.
- Startup/runtime fatal-log matches were empty.
- NCCL logs recorded `P2P/CUMEM` transport.

A full 1M request was intentionally not performed; 1M refers to the validated
service configuration, while the long-context performance sample used 10K.

## Machine-readable data

The publish-safe latest Fix7 dataset is in
[`benchmarks/results/cmp170hx-pp4-fix7-2026-09-02`](../../benchmarks/results/cmp170hx-pp4-fix7-2026-09-02/README.md).
It includes the four-arm summary, all 48 fixed 400-token requests, exact 10K
results, DSpark counters, API correctness gates, and per-GPU measurements.
Host addresses, credentials, local paths, prompts, and full model outputs are
excluded; per-request output hashes are retained for evidence matching.

## Four-arm matrix

| Arm | `BLOCK_H` | Graph | NCCL | Start | Mean decode | 10K prefill | 10K decode |
|---|---:|---|---|---:|---:|---:|---:|
| B0 | 0 | Breakable | default | 444 s | 75.13 tok/s | 3834.54 tok/s | 82.57 tok/s |
| **B8** | **8** | **Breakable** | **default** | **444 s** | **77.30 tok/s** | **4224.96 tok/s** | **92.54 tok/s** |
| CG | 8 | explicit bounded | default | 465 s | 73.52 tok/s | 2731.47 tok/s | 106.51 tok/s |
| CGR | 8 | explicit bounded | Ring/Simple | 464 s | 72.96 tok/s | 2818.12 tok/s | 106.21 tok/s |

Relative to B0, B8 improved mean post-TTFT decode by 2.89%, exact 10K prefill
by 10.18%, and post-TTFT decode after 10K context by 12.07%.

The explicit graph arms increased long-context decode but significantly hurt
prefill and ordinary mixed-prompt decode. They also logged that this model does
not support `torch.compile`, so they are not the default recommendation.

## Language and DSpark split

| Arm | Chinese post-TTFT | English post-TTFT | DSpark weighted acceptance |
|---|---:|---:|---:|
| B0 | 79.69 tok/s | 70.56 tok/s | 27.43% |
| B8 | 78.19 tok/s | 76.41 tok/s | 28.16% |
| CG | 76.99 tok/s | 70.05 tok/s | 26.63% |
| CGR | 76.88 tok/s | 69.04 tok/s | 26.23% |

B8's overall gain was workload-dependent: English cases improved while the
Chinese mean was about 1.9% below B0.

## PP utilization

B8 mean GPU utilization during the fixed workload was approximately:

| GPU stage | Mean utilization | Peak temperature |
|---|---:|---:|
| 0 | 87.7% | 75°C |
| 1 | 83.3% | 69°C |
| 2 | 65.4% | 65°C |
| 3 | 47.8% | 62°C |

This is a PP workload gradient, not proof of P2P failure. P2P/CUMEM transport
was present in every arm. The B8 setting improves throughput but does not make
single-stream stage utilization equal.

## Recommendation

```bash
VLLM_PREFILL_BLOCK_H=8
VLLM_USE_BREAKABLE_CUDAGRAPH=1
```

Keep NCCL and custom all-reduce selection at their defaults for this profile.
