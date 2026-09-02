# Security Policy

## Reporting a vulnerability

Use this repository's GitHub **Security → Report a vulnerability** form. Do
not publish API keys, host addresses, model access tokens, crash dumps, or
driver binaries in a public issue.

For vulnerabilities in unmodified vLLM code, also follow the
[upstream vLLM security policy](https://github.com/vllm-project/vllm/security/policy).

## Deployment assumptions

- Treat model files and `--trust-remote-code` inputs as trusted code.
- Keep the OpenAI-compatible endpoint behind authentication and a trusted
  reverse proxy.
- Store API keys outside the repository and restrict their file permissions.
- Do not run a container with mixed Python/native-extension revisions.
- Driver or P2P unlock procedures are outside this repository's trust boundary.
- Validate CUDA peer correctness, NCCL transport, and Xid logs after driver or
  hardware changes.

No model weights, API keys, driver modules, or firmware should be committed to
this repository.
