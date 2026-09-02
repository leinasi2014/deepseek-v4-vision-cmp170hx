# Changelog

## Unreleased

### Added

- DeepSeek-V4 Flash Vision support on the proven SM80 fast base.
- PP-safe DSpark speculative decoding and image-token routing.
- A 1M-context PP4 production profile for CMP 170HX.
- Reproducible deployment, architecture, and validation documentation.
- Publish-safe machine-readable Fix7 benchmark data with all 48 fixed-token
  requests, exact 10K results, DSpark counters, and per-GPU measurements.

### Fixed

- Sparse MLA and MTP row bounds for padded speculative tokens.
- PP draft propagation and rank-consistent DSpark gating.
- Vision prompt processing across the historical multimodal processor ABI.
- Position-independent pad-free image cache alignment.
- Image placeholder/embedding row mismatches now fail closed.
- Structured images outside user/developer messages are rejected before
  multimodal tracking, preventing an EngineCore HTTP 500.

### Validated

- Fixed-token PP4 matrix on four CMP 170HX GPUs.
- Recommended `VLLM_PREFILL_BLOCK_H=8` with breakable CUDA graphs.
- CUDA peer/NCCL transport observed as P2P/CUMEM in every test arm.
