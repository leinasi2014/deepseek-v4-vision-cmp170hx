# Vision port notes

## Direction

The newer DeepSeek-V4 Vision semantics were ported into the proven SM80 fast
base. Replacing the newer runtime with a handful of historical Python files was
rejected because it would mix incompatible multimodal and native-extension
ABIs.

## Main adaptation points

1. `vllm/config/speculative.py`: normalize DSpark using the checkpoint's
   trained block size.
2. `vllm/config/vllm.py`: include the Vision wrapper in the breakable CUDA
   graph architecture gate.
3. Fused MoE routing: retain the SM80 fallback while adding image-token bias.
4. Model registry: expose the conditional Vision architecture.
5. Sparse cache utilities: combine SM80 paged/LUT handling with image-window
   visibility.
6. `flashmla.py`: size workspaces for image and compressed sparse tokens.
7. `nvidia/model.py`: retain model-level SM80 finalization and add image
   sentinel routing.
8. `nvidia/mtp.py`: propagate image-token IDs into MTP.
9. DeepSeek-V4 tokenizer: preserve reasoning-effort support while mapping
   ordered OpenAI text/image blocks.
10. Sparse SWA: retain the PP/DSpark width rules and add per-token image
    visibility with safe padding-row clearing.

The historical multimodal processor predates a newer private planning API.
The compatibility planner therefore stays local to DeepSeek-V4 Vision and
reuses the existing processor's own conflict-resolution helpers.

## fix7 correctness hardening

Commit `d584b9350d46a808c5883cb59e991561a8d18790` adds image normalization,
pad-free cache regression tests, and multimodal embedding row checks.

Commit `f00f0eecc41146f79f0545c15612962b66b693c5` rejects unsupported image
roles before multimodal tracking. This converts a reproducible EngineCore 500
into an OpenAI-compatible client error.

## Native build requirement

The branch changes both Python and native code. Build and deploy one immutable
image from a single commit. A Python overlay was used only to validate the
final two pure-Python hardening commits against the already matching native
fix6 image; a clean public release image should be rebuilt from the final tree.
