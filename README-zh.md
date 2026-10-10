# 24GB 显卡跑 Qwen3.8-27B（llama.cpp）

在**单张 24GB NVIDIA 显卡**上用 [llama.cpp](https://github.com/ggml-org/llama.cpp) 本地部署 **Qwen3.8-27B** 的开箱即用方案。提供两套配方——**IQ3_S-mtp**（含视觉）与 **UD-Q4_K_XL**——共用同一引擎（b11491, CUDA 13.4）与 OpenAI 兼容 API。每套配方都配有安装、启动与压测脚本。两套配方不能同时运行：它们占用同一端口（8080）与同一块 GPU。

> **English: [README.md](README.md)**

---

## 特性

- **两套配方**——按显存预算、视觉需求与量化质量选择。
- **原生 256K 上下文**——无需 rope 缩放，q4_0 KV 缓存。
- **MTP 投机解码**——使用模型自带的多 token 预测头加速生成。
- **可选 GPU 视觉**——配方 1 将 `mmproj` 编码器卸载到 GPU（图片走标准 OpenAI `image_url` 格式）。
- **OpenAI 兼容 API**——可用于 opencode 及任意 OpenAI 兼容客户端。
- **一条命令安装**——支持断点续传与字节校验。

## 选哪套配方？

| | 配方 1 — IQ3_S | 配方 2 — UD-Q4_K_XL |
|---|---|---|
| 模型 | ISTA-DASLab `IQ3_S-mtp`（12.1 GB） | unsloth `UD-Q4_K_XL`（17.5 GB） |
| 上下文 / KV | 262144 @ q4_0（或 262144 @ q8_0） | 262144 @ q4_0（或 131072 @ q8_0） |
| 视觉 | ✅ mmproj，GPU 卸载（q8_0 KV 变体：纯文本） | ❌ |
| 显存占用 | ~23.0 GiB | ~23.2–24.0 GiB |
| 短 ctx 解码 | ~57–58 t/s | ~38–40 t/s |
| 量化质量 | IQ3_S（GPQA 89.4 / LCB 85.7） | Q4_K_XL（更高） |
| 适用 | 显存紧张 / 要视觉 / 满 256K | 无视觉需求 / 质量优先 |

两者都是 `qwen35` **混合架构**（65 层，`full_attention_interval=4` → 只有 16 层真注意力；4 kv 头 × 256 维），所以 256K 上下文显存开销很低。配方 1 的 IQ3_S 比 Q4_K_XL 省 ~3.2 GB，正好够把视觉编码器留在 GPU 上。

## 环境要求

| 项目 | 要求 |
|---|---|
| GPU | 24GB NVIDIA（实测 RTX A5000，驱动 597.16；3090/4090 兼容） |
| 引擎构建 | 必须 **CUDA 13.x** Windows zip。CUDA 12.4 构建缺 `cudart64_12.dll`/`cublas64_12.dll`，会静默回退 CPU——不要使用 |
| 内存 | 32GB 及以上 |
| 磁盘 | 引擎 ~0.5GB + 模型 12.1–17.5GB（配方 1 视觉再加 0.9GB mmproj） |
| 系统 | Windows 10/11（PowerShell 5.1+）；其他平台请用对应 llama.cpp 构建 |

## 快速开始

### 1. 安装

```powershell
# 配方 1 — IQ3_S（含视觉）
.\install-iq3s.ps1                 # 引擎 + IQ3_S + mmproj
.\install-iq3s.ps1 -TextOnly       # 跳过 mmproj（无视觉，省 ~0.9GB）

# 配方 2 — UD-Q4_K_XL
.\install-udq4.ps1                 # 引擎 + UD-Q4_K_XL
```

两个安装脚本都会下载 llama.cpp **b11491**（CUDA 13.4），并使用断点续传 + 字节校验的下载循环，中断后重跑即可。需要代理时加 `-Proxy <url>`；只装引擎用 `-SkipModel`。

### 2. 启动服务

```powershell
# 配方 1 — IQ3_S
.\start-server-iq3s-q4kv-text-image-256k.ps1   # q4_0 KV + 视觉，256K，ub2048
.\start-server-iq3s-q8kv-text-only-256k.ps1    # q8_0 KV，纯文本（无视觉），256K，ub512

# 配方 2 — UD-Q4_K_XL
.\start-server-udq4-q4kv-text-only-256k.ps1    # q4_0 KV，256K，ub512
.\start-server-udq4-q8kv-text-only-128k.ps1    # q8_0 KV，128K，ub2048
```

API：`http://127.0.0.1:8080/v1`，模型 id 固定为 **`qwen3.8-27b`**（`--alias`，与文件路径无关），无鉴权。

### 3. 验证

```powershell
.\verify-server.ps1                # 健康检查 + 对话（思考模型 max_tokens 要 >= 1024）
```

### 后台运行（tmux）

要让服务在 SSH 会话结束后继续运行，用 tmux 启动：

```powershell
tmux new-session -d -s llm "pwsh -File <repo>\start-server-iq3s-q4kv-text-image-256k.ps1 > <repo>\server.log 2>&1"
```

把 `<repo>` 替换为仓库路径。

## 配置

### 通用设置

| 项目 | 值 |
|---|---|
| 引擎 | llama.cpp b11491（CUDA 13.4） |
| API | `http://0.0.0.0:8080`，别名 `qwen3.8-27b` |
| 并行槽位 | 1（一个 256K 槽位；请求串行处理） |
| 思考深度 | 服务端默认 `--reasoning-effort medium`；支持请求级 `reasoning_effort`（`low`/`medium`/`high`/`xhigh`/`none`）与 `reasoning_budget_tokens` |

### 配方 1 — IQ3_S

```
-m models\Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp.gguf
--mmproj models\mmproj-Qwen3.8-27B-BF16.gguf
-c 262144 --flash-attn on
--cache-type-k q4_0 --cache-type-v q4_0
--alias qwen3.8-27b --parallel 1
-ngl all -t 6 -ub 2048
--spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-poll 1
--no-reasoning-preserve --reasoning-effort medium
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--host 0.0.0.0 --port 8080
```

| 参数 | 说明 |
|---|---|
| `-c 262144` | 架构原生上下文，无需 rope 缩放 |
| `q4_0/q4_0 KV` | 对称 KV 量化以保持 CUDA FA 快速路径；非对称混搭会回退 CPU |
| `-ub 2048` | 提升 prefill 吞吐；需要 >1.5GB 显存余量。q8_0 KV 请用 `-ub 512` |
| `--mmproj` | 将视觉编码器卸载到 GPU；纯文本时去掉该参数或用 `--no-mmproj` |
| `--spec-type draft-mtp` | 启用模型自带 MTP 头做投机解码 |
| `--parallel 1` | 一个 256K 槽位；请求串行 |
| `-t 6` | 非卸载部分的 CPU 线程数 |
| 采样 | Qwen3.8-27B 思考模式预设 |

### 配方 2 — UD-Q4_K_XL

```
-m models\Qwen3.8-27B-UD-Q4_K_XL.gguf
-c 262144 --flash-attn on
--cache-type-k q4_0 --cache-type-v q4_0
--alias qwen3.8-27b --no-mmproj --parallel 1
-ngl all -t 6 -ub 512
--spec-type draft-mtp --spec-draft-poll 1
--no-reasoning-preserve --reasoning-effort medium
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 --presence-penalty 0.0 --repeat-penalty 1.0
--host 0.0.0.0 --port 8080
```

| 参数 | 说明 |
|---|---|
| `--spec-type draft-mtp` | 启用模型自带 MTP 头；`draft-n-max 3` 为默认最优值 |
| `-ub 512`（q4_0）/ `-ub 2048`（q8_0） | q4_0 @256K 需要较小 ubatch 才放得下；128K q8_0 变体有余量用 `-ub 2048` |
| `-c 262144` / `-c 131072` | 256K 配 q4_0，或 128K 配 q8_0（KV 精度更稳） |
| `--no-mmproj` | 无视觉 |

### 采样预设（Qwen3.8-27B）

| 参数 | 思考模式 | 非思考 |
|---|---|---|
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | 1.5 |
| repetition_penalty | 1.0 | 1.0 |

### 显存预算

256K 上下文下，q4_0 KV 约需 4.8GB 显存（q8_0 约需 8.6GB，在 24GB 卡上无法开到 256K）。模型权重、`mmproj` 与 KV 合计约 23GB，在 24GB 卡上仍有余量。换成更小的量化（如 IQ2_S）可再省约 3GB。

## 基准测试

RTX A5000 参考实测（b11491，LAN 客户端）。配方 1 使用 q4_0 KV、256K、含视觉。

**Prefill / 冷启动 TTFT**（`cache_prompt=false`，temp 0）：

| 输入 | 速度 | TTFT |
|---|---|---|
| 2K | 897 t/s | 2.3 s |
| 8K | 972 t/s | 8.9 s |
| 32K | 897 t/s | 31.6 s |
| 128K | 645 t/s | 169.6 s |
| 250K | 478 t/s | 526 s（8.8 分钟） |

32K 以内 prefill 大致恒定在 ~900 t/s，之后是正常的 O(n²) 注意力衰减。空闲短请求 TTFT：0.24–0.31s。视觉：小图（约 30 token）TTFT ~1.9s；完整截图（约 565 token）约 6.6s 完成，控制字符串可精确识别。

**Decode（暖缓存，MTP 开）：** 短上下文 57–58 t/s → 32K 49.8 → 128K 33.2 → 185K 31.6 → 250K 27.5 t/s。

配方 2（UD-Q4_K_XL，`benchmark.ps1`）：

| 场景 | 结果 |
|---|---|
| Prompt eval（2K / 4K，未缓存） | 950 / 973 tok/s |
| 生成（短上下文，temp 1.0） | ~38–40 tok/s |
| 生成 @64K / @128K 上下文 | ~35 / ~32 tok/s |
| TTFT（短 prompt） | ≈0.3 s |
| MTP 接受率 | temp 0: 55–75%；temp 1.0: 48–62% |

## 故障排查

- **Prefill 崩到 ~34 t/s**——由非对称 KV 量化引起（`K ≠ V` 混搭，或 q4_1/q5_x）。请使用对称组合（`q4_0/q4_0`、`q8_0/q8_0` 或 `f16`）；只有对称组合有 CUDA FA 内核。
- **CUDA graph 捕获静默失败（~7 倍减速）**——显存余量不足。`-ub 2048` 需要 >1.5GB 空闲；q8_0 KV + `-ub 2048` 可能被 MTP draft 缓冲 OOM，故 q8_0 配 `-ub 512`。
- **MTP 无效果**——未设置 `--spec-type draft-mtp` 时，MTP 头会作为未使用的张量加载。`draft-n-max` 大于 3 反而更慢。
- **长上下文 prefill 开销大**——单次 250K prompt 需约 9 分钟 prefill；建议分批或先检索。
- **同一时间只能处理一个请求**——`--parallel 1` 会串行化请求。要服务并发用户需大幅降低 `-c`。
- **q8_0 KV 超过 128K 不稳定**——本架构上如此；配方 2 的 q8_0 变体封顶 128K。
- **引擎构建**——只用 CUDA 13.x 的 Windows zip。

## 脚本

| 脚本 | 配方 | 用途 |
|---|---|---|
| `install-iq3s.ps1` | 1 | 引擎 b11491 + IQ3_S + mmproj（`-TextOnly` 跳过视觉） |
| `install-udq4.ps1` | 2 | 引擎 b11491 + UD-Q4_K_XL |
| `start-server-iq3s-q4kv-text-image-256k.ps1` | 1 | q4_0 KV + 视觉，256K，ub2048 |
| `start-server-iq3s-q8kv-text-only-256k.ps1` | 1 | q8_0 KV，无视觉，256K，ub512 |
| `start-server-udq4-q4kv-text-only-256k.ps1` | 2 | q4_0 KV，256K，ub512 |
| `start-server-udq4-q8kv-text-only-128k.ps1` | 2 | q8_0 KV，128K，ub2048 |
| `verify-server.ps1` | 两配方 | 健康检查 + 对话 |
| `benchmark.ps1` | 两配方 | 压测工具（≤128K） |
| `benchmark-full.ps1` | 两配方 | 长上下文压测（196K–262K） |

## 编辑器接入（opencode）

```jsonc
// ~/.config/opencode/opencode.jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "llama.cpp": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "llama-server (local)",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen3.8-27b": {
          "name": "Qwen3.8-27B (Local)",
          "tool_call": true,
          "reasoning": true,
          "attachment": true,            // 仅配方 1：--mmproj 视觉
          "limit": { "context": 262144, "input": 262144, "output": 65536 }
        }
      }
    }
  },
  // 在上下文硬上限前压掉 ~16K token
  "compaction": { "reserved": 16000 },
  // 可选：把压缩交给云端模型，避免占用本地 GPU
  "agent": { "compaction": { "model": "deepseek/deepseek-v4-flash" } }
}
```

配方 2 的 128K 变体把 `limit.context` / `limit.input` 改成 `131072`。图片走标准 OpenAI `image_url` content-part 格式（仅配方 1）。

## 参考

- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- 配方 1 模型 + mmproj: <https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF>
- 配方 2 模型: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>（UD-Q4_K_XL）
- KV 量化: llama.cpp issues #27109（FA 内核回退）、#21385（混合架构上的 q4_0 KV）
- opencode: <https://opencode.ai>

## License

[MIT](LICENSE) © 2026 Mike Lee
