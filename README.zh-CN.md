# 面向 CMP 170HX 的 DeepSeek V4 Vision

[English](README.md) · [架构说明](docs/cmp170hx/ARCHITECTURE.md) ·
[部署说明](deploy/README.md) · [实测结果](docs/cmp170hx/VALIDATION.md) ·
[测试数据](benchmarks/results/cmp170hx-pp4-fix7-2026-09-02/README.md)

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

当前验证源码发布为提交
`f00f0eecc41146f79f0545c15612962b66b693c5`。完整固定-token 测试方法与
结果见 [VALIDATION.md](docs/cmp170hx/VALIDATION.md)。
最新 Fix7 的四臂汇总、48 次固定 400-token 请求、10K 长上下文、DSpark
接受率和逐卡统计见[可复算测试数据](benchmarks/results/cmp170hx-pp4-fix7-2026-09-02/README.md)。

## 最新 Fix7 实测数据

测试日期为 2026-09-02，使用上面的 PP4 配置。四组测试使用完全相同的
模型、硬件、提示词、DSpark 设置和输出长度。普通解码速度取每组 12 次
实测请求的 post-TTFT 平均值，每次都强制生成完整 400 token；预热请求
不计入统计。

| 组别 | `BLOCK_H` | CUDA Graph | 平均解码 | 精确 10K Prefill | 10K 后解码 | DSpark 接受率 |
|---|---:|---|---:|---:|---:|---:|
| B0 | 0 | Breakable | 75.13 tok/s | 3834.54 tok/s | 82.57 tok/s | 27.43% |
| **B8** | **8** | **Breakable** | **77.30 tok/s** | **4224.96 tok/s** | **92.54 tok/s** | **28.16%** |
| CG | 8 | 显式有界 Graph | 73.52 tok/s | 2731.47 tok/s | 106.51 tok/s | 26.63% |
| CGR | 8 | 显式有界 Graph + Ring/Simple | 72.96 tok/s | 2818.12 tok/s | 106.21 tok/s | 26.23% |

推荐配置 B8 的详细结果：

| 测试项 | 结果 |
|---|---:|
| 固定-token 请求 | 12 × 400 completion token |
| 平均端到端生成速度 | 75.07 tok/s |
| 平均纯解码速度（post-TTFT） | 77.30 tok/s |
| 中文纯解码平均值 | 78.19 tok/s |
| 英文纯解码平均值 | 76.41 tok/s |
| 精确 10K Prefill | 4224.96 prompt tok/s |
| 精确 10K 上下文后的 192-token 解码 | 92.54 tok/s |
| DSpark 加权接受率 | 28.16%（2802 / 9950） |
| 启动耗时 | 444 秒 |

B8 四个 PP stage 的平均 GPU 利用率分别为 87.7%、83.3%、65.4% 和
47.8%，对应峰值温度为 75°C、69°C、65°C 和 62°C。四组均通过 1M
服务配置、固定 400-token、精确 10K、视觉、非法图片 HTTP 400、无致命
日志以及 P2P/CUMEM 门禁。没有执行完整 1M 请求。全部 48 条逐请求结果、
逐卡采样统计和指标定义见[完整 CSV 数据](benchmarks/results/cmp170hx-pp4-fix7-2026-09-02/README.md)。

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
