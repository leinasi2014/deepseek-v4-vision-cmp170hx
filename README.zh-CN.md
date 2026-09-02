# 面向 CMP 170HX 的 DeepSeek V4 Vision

[English](README.md) · [架构说明](docs/cmp170hx/ARCHITECTURE.md) ·
[部署说明](deploy/README.md) · [实测结果](docs/cmp170hx/VALIDATION.md)

这是一个面向 NVIDIA CMP 170HX / SM80 的实验性 vLLM 分支，用于运行
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`。项目把已验证的 SM80
DeepSeek‑V4 快速路径与视觉模型、PP 下的 DSpark 投机解码、FP8 DeepSeek
MLA KV Cache 和 1M 上下文配置合并到同一源码树中。

本项目不是 DeepSeek 或 vLLM 官方发布版本。

## 已验证配置

- 4 × NVIDIA CMP 170HX（SM80），CUDA Peer Access 正确
- PP4 × TP1
- `DeepSeek-V4-Flash-Vision-Exp`
- 最大上下文配置：`1,048,576`
- `fp8_ds_mla` KV Cache
- DSpark：5 个 draft token，probabilistic sampling
- 推荐：`VLLM_PREFILL_BLOCK_H=8`
- Breakable CUDA Graph

当前验证提交为
`f09e0305445db93f628ce2ec71378a55370afa8e`。完整固定-token 测试方法与
结果见 [VALIDATION.md](docs/cmp170hx/VALIDATION.md)。

## 主要改造

- DeepSeek-V4 Flash 稀疏 MLA 与 Ampere/SM80 路由。
- DeepSeek-V4 Flash Vision 模型、处理器、视觉编码器和图像 token 路由。
- PP 安全的 DSpark draft 传递与各 rank 一致的 gating。
- 1M 上下文下稀疏索引和投机解码正确性修复。
- 位置无关、pad-free 的多模态缓存块。
- 图像 placeholder 与 embedding 行数的 fail-closed 校验。
- 在多模态跟踪前完成 OpenAI API 图片角色校验，避免 EngineCore 500。

## 快速启动

前提条件：Linux x86_64、Docker、NVIDIA Container Toolkit、4 张具备正确
CUDA P2P/NCCL P2P 的 SM80 显卡，以及从本提交完整构建的原生镜像。

```bash
export IMAGE=dsv4-cmp170hx:local
export MODEL_PATH=/path/to/DeepSeek-V4-Flash-Vision-Exp
export API_KEY_FILE=/run/secrets/dsv4-api-key
bash deploy/serve-pp4.sh
```

模型权重、API Key、显卡驱动和 CMP170HX 解锁补丁都不进入仓库。
部署到网络前请完整阅读 [deploy/README.md](deploy/README.md)。

## 必须知道的边界

- 1M 是已验证的服务配置；公开跑分按要求使用精确 10K prompt，没有发送
  完整 1M 请求。
- PP4 单请求仍存在 stage 利用率梯度；P2P 正常不代表四个 PP stage
  利用率完全相等。
- Python 源码与 CUDA/libtorch 原生扩展必须来自同一个提交和镜像，不能
  用 bind mount 混装。
- 这里只声明经过实测的 CMP170HX/SM80 + DSV4 Vision 配置，不替代上游
  vLLM 对其他模型和硬件的支持承诺。

## 来源与许可证

项目基于 [vLLM](https://github.com/vllm-project/vllm)，保留 Apache‑2.0
许可证，并吸收了 `allover326/deepseek-v4-cmp170hx`、
`tim-odonnell/cmp170hx-deepseek-v4-flash` 和
`satspace-cpu/cmp170hx-linux-p2p` 的公开研究。详见 [NOTICE](NOTICE)。
