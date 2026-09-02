# Agent instructions

These rules apply to AI-assisted work in the DeepSeek V4 Vision for CMP 170HX
fork. A human contributor must review, understand, and defend every change.

## Project scope

- Keep changes focused on the DeepSeek-V4 Vision SM80/CMP170HX path.
- General vLLM fixes should normally be proposed upstream and linked here.
- Preserve upstream copyright, license, and commit attribution.
- Do not add model weights, API keys, private addresses, driver modules,
  firmware, or unlock binaries.

## Runtime invariants

- Python wrappers and native CUDA/libtorch extensions must come from one source
  revision and one immutable image.
- Treat P2P topology as a hint, not a correctness test. Hardware-affecting
  changes require CUDA peer, NCCL transport, and Xid/fatal-log gates.
- Preserve the 1M configuration contract unless a change explicitly and
  visibly changes it.
- Fail closed on multimodal placeholder, embedding, role, and cache alignment
  mismatches.

## Development

- Use `uv`; never use system `python3` or bare `pip`.
- Match the inherited vLLM style and keep Python lines within 88 characters.
- Prefer the nearest existing test file and the cheapest test level that
  catches the behavior.
- Run model evaluations for changes that can affect output or accuracy.
- Read any nested `AGENTS.md` before changing files in that subtree.
- Read the agent-instruction editing guide before changing this file.

Typical setup:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements/lint.txt
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto
```

## Performance claims

Every performance change must include:

- exact baseline and candidate commits;
- identical model, hardware, parallelism, prompts, and requested output length;
- fixed-token results rather than early-stop timing;
- correctness and fatal-log gates;
- raw or reproducible evidence;
- an explicit rollback condition.

Do not infer a win from configuration, topology, profiler occupancy, or one
favorable request.

## Pull requests

- Search this repository and upstream vLLM for duplicate issues/PRs.
- Explain why the change belongs in this hardware-specific fork.
- State all test commands and results.
- Disclose AI assistance and add the repository's normal commit trailers.
- Never publish secrets or private deployment evidence in a public issue.
