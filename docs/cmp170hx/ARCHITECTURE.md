# Architecture

## Goal

Run `DeepSeek-V4-Flash-Vision-Exp` on SM80/CMP170HX while retaining the
measured performance properties of the earlier DeepSeek-V4 Flash SM80 fork.

The implementation is a full source fork because the model wrappers, sparse
attention kernels, speculative decoding path, and native CUDA operations must
share one revision and ABI.

## Port layers

### 1. SM80 execution base

The base provides:

- Ampere sparse MLA selection and Triton fallbacks;
- SM80 FP8 DeepSeek MLA cache handling;
- Marlin FP8/MXFP4 execution;
- pipeline communication priming;
- breakable CUDA graph integration;
- long-context sparse-index bounds.

### 2. Vision feature layer

The Vision port adds:

- `DeepseekV4ForConditionalGeneration` registration;
- vision preprocessing, encoder, aligner, and multimodal embeddings;
- image sentinel and image-token MoE routing;
- bidirectional in-image attention visibility;
- interleaved language/vision weight loading;
- OpenAI content-part and tokenizer integration.

### 3. PP + DSpark correctness layer

The PP path carries draft routing IDs and image-token metadata between stages.
DSpark proposal activation is rank-consistent, and padded MTP/indexer row
lengths are clamped before sparse access.

### 4. Multimodal cache and API boundary

Image cache blocks are position-independent and pad-free. Dynamic alignment
padding is added only when the cached block is spliced into the final prompt.
This avoids salting the cache by prompt position while preserving alignment.

Before multimodal items are tracked, structured images outside the
`user`/`developer` roles are rejected with a client error. Quoted text such as
`<image>` remains legal and does not become a false image target.

## Validated runtime

```text
OpenAI API
    ↓
DeepSeek-V4 tokenizer / multimodal processor
    ↓
Vision encoder + image-token routing
    ↓
PP0 → PP1 → PP2 → PP3
    ↘ DSpark draft proposal / rank-consistent gating
    ↘ sparse MLA / fp8_ds_mla KV cache
```

The validated PP partition is `11,11,12,9`. Unequal layer counts account for
first/last-stage modules and memory constraints; they do not guarantee equal
single-stream GPU utilization.

## Deliberate non-features

- No driver or P2P unlock code.
- No model weights.
- No claim that `torch.compile` is supported for this model on SM80.
- No claim that forced NCCL Ring/Simple improves the PP4 profile.
- No live bind-mount deployment of source files over an unrelated image.
