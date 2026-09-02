# Contributing

Contributions should stay focused on the DeepSeek-V4 Vision SM80/CMP170HX
profile. General vLLM changes should normally be proposed upstream first.

## Before opening a pull request

1. Explain the hardware, driver, CUDA, model, parallelism, and context profile.
2. State whether P2P correctness was tested rather than inferred from topology.
3. Add the smallest relevant unit test.
4. For performance changes, provide a fixed-token A/B result and retain the
   raw JSON or command line needed to reproduce it.
5. For model-affecting changes, include a text and visual correctness check.
6. Confirm no model weights, API keys, private addresses, or driver binaries
   are included.

The inherited vLLM style and development rules in `AGENTS.md` continue to
apply. Use `uv` and the repository's existing lint/test configuration.

## Pull request scope

A useful pull request should identify:

- the baseline commit;
- the exact change and expected mechanism;
- correctness gates;
- performance gates;
- rollback behavior.

Do not combine Python wrappers from one revision with native extensions built
from another revision.

## Upstream attribution

When a change is suitable for general vLLM, link the corresponding upstream
issue or pull request and keep upstream attribution intact.
