# Xid 31 Root Cause — RESOLVED (2026-09-03)

## Verdict

The production Xid 31 crashes (2026-09-03 04:33 and 05:48 UTC) were caused by a
**dropped environment variable during container recreation**, not by the
optimization, the hardware, NCCL, or the batch size.

`DSV4_LOGITS_ROW_CHUNK=64` was present in the original production container
(see the 03:25 backup JSON) but was **absent from the hand-written
`prod-relaunch.sh`** used to recreate the container for the 03:30 optimization
switch — and again for the 05:48 rollback. With the variable missing the engine
defaults to `ROW_CHUNK=0`, which routes the sparse-indexer prefill logits
through the **unchunked single-shot Triton path**. That path performs an
illegal GPU write once a single request's context reaches **~174k tokens**
(Xid 31, MMU REGION_VIOLATION VIRT_WRITE).

This also exonerates the 2026-09-02 factor sweep conclusions that were
suspected after the incident: `NCCL_P2P_LEVEL=SYS` and
`--max-num-batched-tokens 2048` are safe; the earlier incident note
("bat2048 Xid-31 in production; keep 4096") is superseded by this document.

## Evidence matrix (16+ controlled rounds, GPU0-3, 250k-token churn)

| DSV4_LOGITS_ROW_CHUNK | Outcome |
|---|---|
| 0 (default) | crashed at ~174k in **every** round (5+): graphs or eager, blocking on/off, KV_GROUP 1 or 8, bat 2048/4096, patched or unpatched image |
| 64 | **passed** in every round (8+): including the exact production image (overlay3), exact production config, graphs on, no launch blocking, no instrumentation, mixed decode+prefill load, image requests at 180k, agent-style growing prefixes over a cached 250k context, and 512-token DSpark decodes at 250k context |

Timeline corroboration: the original container (with the variable) ran 13h+
with zero Xid; crashes began exactly when the relaunch script replaced it.

## Resolution

- Production was rebuilt faithfully from the 03:25 backup JSON (all env vars,
  including `DSV4_LOGITS_ROW_CHUNK=64`) and then upgraded to
  `--max-num-batched-tokens 2048` + the `overlay5` image (two upstream
  hardening backports: draft-KV null-block writes `daf32add1`, SWA index width
  separation `02aaf87f3`).
- The hand-written `prod-relaunch.sh` is deprecated. All future changes must go
  through a complete-env script (this repo's `deploy/serve-pp4.sh` or a copy of
  the JSON-faithful restore script), changing one variable at a time.

## Operational lessons

1. Container recreation scripts must be generated from `docker inspect`
   (config JSON), never hand-transcribed — a single missing env var sat
   silently behind a multi-day crash hunt.
2. The unchunked indexer-logits path remains unsafe at very long context on
   SM80; `DSV4_LOGITS_ROW_CHUNK=64` is a **mandatory** setting for this model
   and is now documented as such in `deploy/serve-pp4.sh`.
3. Repeated Xid 31 events escalate to Xid 154 (driver demands OS reboot) and
   poison NCCL initialization on the affected GPU — after any Xid 31, expect a
   host reboot before multi-GPU jobs will start again.
