# Native SM80 image

`Dockerfile.cmp170hx-sm80` is the full native build recipe used by this fork.
It targets CUDA architecture 8.0 and keeps dependency layers ahead of the
source-tree copy so rebuilds can reuse Docker cache.

## External source bundle

The build context must contain:

```text
dsv4-srcs/
├── cutlass/
└── triton/
    └── python/triton_kernels/
```

These are source dependencies, not generated binaries. Pin their exact Git
revisions in the release/build record used for an image. Do not silently reuse
an unversioned directory from another source tree.

The validated 2026-09-02 image reported PyTorch `2.13.0+cu130`, CUDA `13.0`,
and the source-tree placeholder version `0.0.0`. Its copied CUTLASS/Triton
directories did not retain `.git` metadata, so this repository does not invent
unverifiable revision IDs for that historical image. New builds must record
them explicitly.

Example context preparation:

```bash
mkdir -p build-context/dsv4-srcs
cp -a /path/to/pinned/cutlass build-context/dsv4-srcs/
cp -a /path/to/pinned/triton build-context/dsv4-srcs/
rsync -a --exclude .git ./ build-context/
docker build \
  -f build-context/docker/Dockerfile.cmp170hx-sm80 \
  -t dsv4-cmp170hx:local \
  build-context
```

## Required release evidence

Record at minimum:

- repository commit;
- final image ID/digest;
- CUTLASS and Triton revisions;
- CUDA, PyTorch, and vLLM versions;
- `import vllm` and DeepSeek-V4 Vision import gates;
- GPU unit tests;
- CUDA peer correctness and NCCL P2P evidence;
- Xid/fatal log scan.

The source tree includes both Python wrappers and native CUDA changes. Do not
deploy it by bind-mounting Python files over an image built from another commit.
