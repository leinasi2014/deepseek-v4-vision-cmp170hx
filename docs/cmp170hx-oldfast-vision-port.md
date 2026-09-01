# CMP170HX DSV4-F Vision port

## Decision

Use the old fast CMP170HX tree as the immutable base and port the official
DeepSeek-V4 Vision feature/correctness stack into it. Do not replace the newer
Vision runtime with the old eight bind-mounted Python files.

Base:

- repository: `/home/hyl/dsv4-build-tip`
- commit: `ac324f28226e9816588dba41208f7957b9fce2ff`
- branch: `cmp170hx-oldfast-vision-port`
- worktree: `/home/hyl/dsv4-build-oldfast-vision`

Vision sources:

- official feature/fix stack: `edafe3d..71165e0`
- validated CMP170HX Vision stack: `92067f8..e94690c`
- deployment-only profile: `5c28cb5`

The original `/home/hyl/dsv4-build-tip` worktree is intentionally untouched.
It contains uncommitted DSpark confidence work that is incompatible with the
historical old image API and must not be used as an implicit build input.

## Why this direction

The old base already owns the performance-sensitive implementation:

- PP8 communication priming and pipeline metadata handling;
- breakable CUDA graph adjustments;
- CMP170HX/SM80 sparse indexer and Ampere sparse MLA backend;
- old DSpark proposal/rejection hot paths;
- 1M-context sparse-index bounds.

The Vision stack adds semantics that cannot be recreated with launch flags:

- `DeepseekV4ForConditionalGeneration` registration and architecture routing;
- image preprocessing, placeholder/sentinel generation, vision encoder and
  aligner;
- image-token MoE routing through `bias_vl`;
- bidirectional in-image SWA visibility while preserving causal text windows;
- interleaved Vision/language checkpoint streaming and model-level weight
  finalization;
- trained `dspark_block_size` normalization and MTP image-token propagation.

Moving the old eight runtime files wholesale into the newer Vision tree is not
safe. In particular it can drop placeholder/block verification and residual
resampling fixes in rejection sampling, and it repeats the live-mount drift
that already made the historical old container fail with a new config field.

## Port content

The official Vision patch applied directly for most files. Ten files needed
manual three-way adaptation because the old base already had SM80 or DSpark
changes:

1. `vllm/config/speculative.py`: retain the old config flow and normalize the
   draft with the checkpoint's trained `dspark_block_size`.
2. `vllm/config/vllm.py`: include the Vision wrapper in the old breakable-CUDA-
   graph architecture gate.
3. `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`:
   preserve the old XPU fallback and add NVIDIA image-token routing arguments.
4. `vllm/model_executor/models/registry.py`: register the conditional Vision
   architecture in the old registry layout.
5. `vllm/models/deepseek_v4/common/ops/cache_utils.py`: combine SM80 paged/LUT
   handling with widened in-image SWA rows.
6. `vllm/models/deepseek_v4/nvidia/flashmla.py`: size workspace for
   `window + image + compressed top-k`.
7. `vllm/models/deepseek_v4/nvidia/model.py`: retain old model-level MegaMoE/MHC
   finalization and add image sentinel routing. The newer immediate finalization
   hunk was deliberately not copied.
8. `vllm/models/deepseek_v4/nvidia/mtp.py`: propagate image-token IDs into MTP.
9. `vllm/tokenizers/deepseek_v4_encoding.py`: flatten OpenAI content blocks to
   ordered text/image placeholders without removing the old reasoning-effort
   support.
10. `vllm/v1/attention/backends/mla/sparse_swa.py`: preserve old DSpark
    non-causal width and add per-token image visibility plus safe padding-row
    clearing.

New Vision modules are copied from the validated official stack:

- `vllm/models/deepseek_v4/common/mm_preprocess.py`
- `vllm/models/deepseek_v4/common/vision.py`
- `vllm/models/deepseek_v4/nvidia/vl_model.py`
- `vllm/models/deepseek_v4/vl_stub.py`

The patch also changes `csrc/libtorch_stable/moe`; therefore this branch
requires a full native image rebuild. A Python overlay or live bind mount is
not a valid deliverable because the Python wrapper and compiled op ABI must
match.

## Build and rollout plan

1. Build one immutable image from this exact branch and record the Git commit,
   image digest, CUDA/PyTorch/vLLM versions, and hashes of the five NVIDIA
   driver modules. Do not use source bind mounts.
2. Inside the built image run `import vllm`, import the Vision wrapper and
   preprocessing module, and run the ported CPU/unit tests before accessing a
   GPU.
3. Run the existing CUDA peer correctness, NCCL P2P/CUMEM and Xid gates. A
   topology table alone is not proof of correct P2P.
4. Start an experiment on a non-production port with DSpark disabled. Validate
   text-only and one-image requests, checkpoint streaming, PP stage ownership,
   and the configured `max_model_len=1048576` contract.
5. Enable PP8 DSpark only after the non-speculative path is correct. Validate
   sampled IDs, acceptance lengths and deterministic correctness before
   measuring speed.
6. Compare old text and Vision text on the same Vision checkpoint, prompt,
   output-token count and concurrency. Historical DSV4-FINAL numbers are useful
   context but are not an exact A/B because the weights differ.
7. Promote only if text regression is within the agreed budget, image requests
   pass, all eight ranks remain healthy and Xid stays empty. Rollback is the
   existing immutable production image, not a source-tree checkout.

## Current validation

- combined patch dry-run: 22 tracked files applied plus 4 new Vision modules;
- manual conflict resolution: 10 files / 23 rejected hunks resolved;
- `git diff --check`: pass;
- Python `compileall`: pass;
- Ruff undefined-name/import checks (`F` family): pass;
- host import audit: PyTorch and base `vllm` import successfully when the
  installed `_C_stable_libtorch` is exposed to the source tree; the Vision
  import then reaches the next absent source-tree artifact, the compiled
  FlashAttention extension. The complete native import gate therefore belongs
  inside the immutable rebuilt image.

No model container, benchmark, or production endpoint change is part of this
source-port commit.
