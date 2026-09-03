# Optimization sweep — 2026-09-03 (GPU 4–7, PP4, 1M contract)

## Scope

A single-factor + combination sweep on 4× CMP 170HX (SM80) against the Fix7
production image (`dsv4-a100:oldfast-vision-fix7-f09e0305445d-sm80-overlay3`).
Every arm used the same model, PP partition baseline, DSpark n=5, breakable
CUDA graphs, prefix caching, fp8 KV cache, 1,048,576-token contract, fixed
prompt set, and a cold engine per arm (restart invalidates in-memory prefix
cache, so identical prompts stay cold across arms).

Metrics: 12×400 forced-completion decode (mean post-TTFT), exact 10K prefill,
192-token decode after a separate exact-10K prompt, 16-way concurrent chat
aggregate, and DSpark acceptance from engine counters
(`vllm:spec_decode_num_accepted_tokens_total` /
`vllm:spec_decode_num_draft_tokens_total`).

## Results

See [`matrix-summary.csv`](../../benchmarks/results/cmp170hx-pp4-opt-sweep-2026-09-03/matrix-summary.csv)
and per-arm `summary.json` / `requests.csv` in the same directory.

| Arm | Change vs baseline | Decode 12×400 | Decode 10K | Prefill 10K | C16 | Acceptance |
|---|---|---:|---:|---:|---:|---:|
| s0b | baseline (part 11,11,12,9 · bat4096 · util0.85 · NCCL default) | 77.72 | 150.8 | 3525 | 330.8 | 32.9% |
| s2b-part | partition 12,11,11,9 | 86.34 | 150.0 | 3984 | 289.0 | 37.8% |
| s4b-bat2048 | batched tokens 2048 | 87.19 | 147.0 | 3679 | 331.6 | 38.2% |
| s5b-util93 | gpu-memory-utilization 0.93 | 93.25 | 151.2 | 3527 | 284.0 | 39.9% |
| s6b | **NCCL_P2P_LEVEL=SYS** | **99.07** | 109.4 | 3930 | 284.6 | **45.9%** |
| s6bU93 | SYS + util0.93 | 83.52 | 111.6 | 3636 | 298.0 | 37.5% |
| s6bP12 | SYS + partition 12 | 80.27 | 151.1 | 3707 | 289.3 | 35.3% |
| **s6bB20** | **SYS + bat2048 (recommended)** | **96.70** | **147.2** | **4236** | **329.9** | 42.9% |
| s6bP12U93 | SYS + part12 + util0.93 | 92.69 | — | — | 297.8 | — |
| s3b-kaka | kaka86mm profile (part12+bat2048+util0.93+Ring+FULL_AND_PIECEWISE) | 82.64 | — | 3355 | 283.7 | 38.8% |

Additional observations:

- **NCCL_P2P_LEVEL=SYS is a distance *ceiling*, not a P2P disable** (official
  NCCL semantics: "maximum distance between GPUs where NCCL will use the P2P
  transport"). GPU 4–7 pairs are NODE-level topology, past the default PIX
  threshold, so the default never engages P2P; SYS re-enables it and is worth
  +27% single-stream decode on a same-NUMA 4-card PP group. Cross-NUMA
  (GPU 0–3 ↔ 4–7) pairs are SYS-level; do not use SYS for 8-card jobs.
- Factors do not stack on top of SYS: util0.93, partition 12,11,11,9, and the
  full four-factor combination all regress (80–93). Only bat2048 composes
  (s6bB20).
- DSpark: `num_speculative_tokens` must be ≥ `dspark_block_size` (5) on this
  fork; n=6 and n=7 hang during startup (engine-level, no Xid).
- A plan-2 engine build (wtdcode/vllm-backport `dsv4-vision-exp` + PR #54566
  first six commits + the four sm80 patches, image `dsv4-vision:sm80`) measured
  64.63 / 64.36 tok/s under the same harness (its DSpark requires n divisible
  by n_predict=3). It is retained as an upstream-tracking baseline, not for
  production.

## Recommended profile (= deploy/serve-pp4.sh defaults)

`NCCL_P2P_LEVEL=SYS` + `--max-num-batched-tokens 2048`, everything else as the
published Fix7 profile. Measured: 96.7 tok/s decode, 147.2 tok/s after 10K,
4236 tok/s prefill, 329.9 tok/s at C16, 42.9% DSpark acceptance
(baseline: 77.7 / 150.8 / 3525 / 330.8 / 32.9%).

## Safety ledger

- Xid errors: **0** across the entire campaign (dmesg scan, 20+ arms).
- Peak GPU memory: 62,404 MiB (60.9 GiB) on util0.93 arms; 58.8 GiB on util0.85
  arms — under the 62 GiB ceiling.
- Power: mean 57–62 W/card, P95 109–138 W; transient spikes up to ~293 W occur
  under any DVFS power limit (200/230/250 W) at ~0.5% of samples, ≤10 s each;
  no Xid was ever recorded with the stock 250 W limit.
- Temps 49–60 °C. The production container on GPU 0–3 was not modified during
  testing and stayed healthy.
