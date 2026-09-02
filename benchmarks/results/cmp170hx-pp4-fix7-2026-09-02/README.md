# CMP 170HX PP4 Fix7 benchmark dataset

This directory contains the publish-safe, machine-readable data behind the
2026-09-02 validation of the latest Fix7 source. The source under test is
commit `f00f0eecc41146f79f0545c15612962b66b693c5`; later documentation-only
commits do not change the runtime tree.

## Test profile

- Model: `DeepSeek-V4-Flash-Vision-Exp`
- Hardware: 4 x NVIDIA CMP 170HX (SM80)
- Parallelism: PP4 x TP1
- KV cache: `fp8_ds_mla`
- Speculative decoding: DSpark, 5 draft tokens
- Service context contract: 1,048,576 tokens
- Exact long-context workload: 10,000 prompt tokens
- Fixed decode workload: 6 cases x 2 repetitions x 400 completion tokens
- Warm-up requests: one per arm, excluded from every summary

The four arms differ only in the prefill block, CUDA graph, and NCCL override:

| Arm | `VLLM_PREFILL_BLOCK_H` | Graph mode | NCCL override |
|---|---:|---|---|
| B0 | 0 | breakable | none |
| B8 | 8 | breakable | none |
| CG | 8 | explicit bounded | none |
| CGR | 8 | explicit bounded | Ring/Simple |

B8 is the recommended/latest production configuration.

## Files

- [`matrix-summary.csv`](matrix-summary.csv): arm-level throughput, DSpark,
  startup, and long-context results.
- [`decode-400-requests.csv`](decode-400-requests.csv): all 48 measured
  fixed-token requests, including TTFT, E2E/post-TTFT throughput, DSpark
  counters, and output hashes.
- [`gpu-summary.csv`](gpu-summary.csv): utilization, power, clocks,
  temperature, and memory by arm and PP stage.
- [`api-correctness.csv`](api-correctness.csv): correctness and service
  contract gates for every arm.
- [`SHA256SUMS`](SHA256SUMS): checksums for all published CSV files.

`post_ttft_tps` is `(completion_tokens - 1) / decode_window_seconds`.
`e2e_tps` is `completion_tokens / total_seconds`. DSpark acceptance is
`accepted_tokens / drafted_tokens`.

## Publication boundary

The original evidence includes service logs, metrics snapshots, exact prompts,
and full model outputs. Those artifacts are intentionally not published
because they contain host-local paths and model reasoning text. This dataset
removes addresses, API credentials, local paths, prompts, and output bodies.
Per-request output SHA-256 values are retained so an authorized holder of the
raw evidence can verify correspondence without disclosing the output.

No full 1M request was performed. The 1M figure is the validated service
contract; the longest measured prompt was exactly 10,000 tokens.
