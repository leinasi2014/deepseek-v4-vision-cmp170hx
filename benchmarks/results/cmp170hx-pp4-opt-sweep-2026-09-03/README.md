# CMP 170HX PP4 optimization sweep — 2026-09-03

Single-factor + combination sweep on GPU 4–7 of the same 8-card host, Fix7
image, 1M contract, fixed prompt set, cold engine per arm. Round-1 arm names
without a `b` suffix (`s0-b8`, `s1-n3`, …) are **invalid** (env not exported →
all arms ran the same config; kept only for audit). Method and conclusions:
[`docs/cmp170hx/OPTIMIZATION-2026-09-03.md`](../../../docs/cmp170hx/OPTIMIZATION-2026-09-03.md).

- `matrix-summary.csv` — consolidated per-arm metrics (decode 12×400, decode
  after 10K, prefill 10K, C16 aggregate, DSpark acceptance).
- `<arm>/summary.json`, `<arm>/requests.csv` — per-arm raw bench output.
- `p2def/`, `p2ours/` — plan-2 engine comparison arms (n_predict=3 lineage).

Recommended profile from this sweep: `NCCL_P2P_LEVEL=SYS` +
`--max-num-batched-tokens 2048` (arm `s6bB20`).
