# Qwen3.8-27B on a 24 GB GPU with llama.cpp

A ready-to-run setup for serving **Qwen3.8-27B** locally on a single **24 GB NVIDIA GPU** with [llama.cpp](https://github.com/ggml-org/llama.cpp). Two recipes are provided — **IQ3_S-mtp** (with vision) and **UD-Q4_K_XL** — sharing the same engine (b11491, CUDA 13.4) and an OpenAI-compatible API. Each recipe ships install, start, and benchmark scripts. Only one recipe runs at a time: they share port 8080 and the GPU.

> **中文版：[README-zh.md](README-zh.md)**

---

## Features

- **Two recipes** — pick by VRAM budget, vision support, and quant quality.
- **Full 256K native context** — no rope scaling, q4_0 KV cache.
- **MTP speculative decoding** — the model's built-in multi-token-prediction head for faster generation.
- **Optional GPU vision** — Recipe 1 offloads the `mmproj` encoder (images via the OpenAI `image_url` format).
- **OpenAI-compatible API** — works with opencode and any OpenAI-compatible client.
- **One-command installs** — resumable, size-verified downloads.

## Choosing a recipe

| | Recipe 1 — IQ3_S | Recipe 2 — UD-Q4_K_XL |
|---|---|---|
| Model | ISTA-DASLab `IQ3_S-mtp` (12.1 GB) | unsloth `UD-Q4_K_XL` (17.5 GB) |
| Context / KV | 262144 @ q4_0 (or 262144 @ q8_0) | 262144 @ q4_0 (or 131072 @ q8_0) |
| Vision | ✅ mmproj, GPU-offloaded (q8_0 KV variant: text-only) | ❌ |
| VRAM used | ~23.0 GiB | ~23.2–24.0 GiB |
| Decode (short ctx) | ~57–58 t/s | ~38–40 t/s |
| Quant quality | IQ3_S (GPQA 89.4 / LCB 85.7) | Q4_K_XL (higher) |
| Use when | tight VRAM / want vision / full 256K | no vision needed / max quality |

Both are `qwen35` **hybrid** models (65 blocks, only 16 attention layers via `full_attention_interval=4`, 4 kv heads × 256 dim), so 256K context is cheap in VRAM. Recipe 1's IQ3_S quant saves ~3.2 GB vs Q4_K_XL, which is what allows the vision encoder to stay on the GPU.

## Requirements

| Item | Requirement |
|---|---|
| GPU | 24 GB NVIDIA (tested: RTX A5000, driver 597.16; 3090/4090 compatible) |
| Engine build | **CUDA 13.x** Windows zip. The CUDA 12.4 build ships without `cudart64_12.dll`/`cublas64_12.dll` and silently falls back to CPU — do not use it. |
| RAM | 32 GB or more |
| Disk | engine ~0.5 GB + model 12.1–17.5 GB (+ 0.9 GB mmproj for Recipe 1 vision) |
| OS | Windows 10/11 (PowerShell 5.1+); other platforms: use the corresponding llama.cpp build |

## Quick start

### 1. Install

```powershell
# Recipe 1 — IQ3_S (+ vision)
.\install-iq3s.ps1                 # engine + IQ3_S + mmproj
.\install-iq3s.ps1 -TextOnly       # skip mmproj (no vision, saves ~0.9 GB)

# Recipe 2 — UD-Q4_K_XL
.\install-udq4.ps1                 # engine + UD-Q4_K_XL
```

Both installers download llama.cpp **b11491** (CUDA 13.4) and use a resumable, size-verified download loop, so an interrupted download can simply be re-run. Add `-Proxy <url>` if a proxy is required, or `-SkipModel` to install the engine only.

### 2. Start the server

```powershell
# Recipe 1 — IQ3_S
.\start-server-iq3s-q4kv-text-image-256k.ps1   # q4_0 KV + vision, 256K, ub2048
.\start-server-iq3s-q8kv-text-only-256k.ps1    # q8_0 KV, text-only (no vision), 256K, ub512

# Recipe 2 — UD-Q4_K_XL
.\start-server-udq4-q4kv-text-only-256k.ps1    # q4_0 KV, 256K, ub512
.\start-server-udq4-q8kv-text-only-128k.ps1    # q8_0 KV, 128K, ub2048
```

API: `http://127.0.0.1:8080/v1`, model id **`qwen3.8-27b`** (fixed via `--alias`, independent of file path), no API key.

### 3. Verify

```powershell
.\verify-server.ps1                # health check + chat (use max_tokens >= 1024 for the thinking model)
```

### Run in the background (tmux)

To keep the server alive across SSH sessions, run it under tmux:

```powershell
tmux new-session -d -s llm "pwsh -File <repo>\start-server-iq3s-q4kv-text-image-256k.ps1 > <repo>\server.log 2>&1"
```

Replace `<repo>` with the repository path.

## Configuration

### Shared settings

| Setting | Value |
|---|---|
| Engine | llama.cpp b11491 (CUDA 13.4) |
| API | `http://0.0.0.0:8080`, alias `qwen3.8-27b` |
| Parallel slots | 1 (a single 256K slot; requests are serialized) |
| Reasoning | server default `--reasoning-effort medium`; per-request `reasoning_effort` (`low`/`medium`/`high`/`xhigh`/`none`) and `reasoning_budget_tokens` are supported |

### Recipe 1 — IQ3_S

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

| Parameter | Notes |
|---|---|
| `-c 262144` | Native context length of the architecture; no rope scaling. |
| `q4_0/q4_0 KV` | Symmetric KV quantization keeps the fast CUDA FA path; asymmetric mixes fall back to CPU. |
| `-ub 2048` | Higher prefill throughput; requires >1.5 GB VRAM headroom. Use `-ub 512` with q8_0 KV. |
| `--mmproj` | Offloads the vision encoder to the GPU. Omit the flag (or use `--no-mmproj`) for text-only. |
| `--spec-type draft-mtp` | Enables the model's MTP head for speculative decoding. |
| `--parallel 1` | One 256K slot; requests are serialized. |
| `-t 6` | CPU threads for non-offloaded work. |
| Sampling | Qwen3.8-27B thinking-mode preset. |

### Recipe 2 — UD-Q4_K_XL

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

| Parameter | Notes |
|---|---|
| `--spec-type draft-mtp` | Enables the model's MTP head; `draft-n-max 3` is the default optimum. |
| `-ub 512` (q4_0) / `-ub 2048` (q8_0) | q4_0 at 256K needs the smaller ubatch to fit; the 128K q8_0 variant has room for `-ub 2048`. |
| `-c 262144` / `-c 131072` | 256K with q4_0, or 128K with q8_0 (safer KV precision). |
| `--no-mmproj` | No vision. |

### Sampling presets (Qwen3.8-27B)

| Parameter | Thinking mode | Non-thinking |
|---|---|---|
| temperature | 1.0 | 0.7 |
| top_p | 0.95 | 0.80 |
| top_k | 20 | 20 |
| min_p | 0.0 | 0.0 |
| presence_penalty | 0.0 | 1.5 |
| repetition_penalty | 1.0 | 1.0 |

### Memory budget

At 256K context, q4_0 KV needs ~4.8 GB of VRAM (q8_0 would need ~8.6 GB, which does not fit a 24 GB card at 256K). Model weights, `mmproj`, and KV together use ~23 GB, leaving headroom on a 24 GB card. A smaller quant such as IQ2_S can reduce the footprint by ~3 GB.

## Benchmarks

Reference measurements on an RTX A5000 (b11491, LAN client). Recipe 1 with q4_0 KV at 256K plus vision.

**Prefill / cold TTFT** (`cache_prompt=false`, temp 0):

| Input | Speed | TTFT |
|---|---|---|
| 2K | 897 t/s | 2.3 s |
| 8K | 972 t/s | 8.9 s |
| 32K | 897 t/s | 31.6 s |
| 128K | 645 t/s | 169.6 s |
| 250K | 478 t/s | 526 s (8.8 min) |

Prefill is roughly flat at ~900 t/s up to 32K, then follows the usual O(n²) attention falloff. Idle short-request TTFT: 0.24–0.31 s. Vision: TTFT ~1.9 s for a small image (~30 tok); a full screenshot (~565 tok) answers in ~6.6 s with exact OCR of a control string.

**Decode (warm, MTP on):** short ctx 57–58 t/s → 32K 49.8 → 128K 33.2 → 185K 31.6 → 250K 27.5 t/s.

Recipe 2 (UD-Q4_K_XL, `benchmark.ps1`):

| Scenario | Result |
|---|---|
| Prompt eval (2K / 4K, uncached) | 950 / 973 tok/s |
| Generation (short context, temp 1.0) | ~38–40 tok/s |
| Generation @64K / @128K context | ~35 / ~32 tok/s |
| TTFT (short prompt) | ≈0.3 s |
| MTP acceptance | temp 0: 55–75%; temp 1.0: 48–62% |

## Troubleshooting

- **Prefill collapses to ~34 t/s** — caused by asymmetric KV quantization (a `K ≠ V` mix, or q4_1/q5_x). Use a symmetric combination (`q4_0/q4_0`, `q8_0/q8_0`, or `f16`); only symmetric pairs have CUDA FA kernels.
- **CUDA graph capture silently fails (~7× slower)** — insufficient VRAM headroom. `-ub 2048` needs >1.5 GB free; q8_0 KV + `-ub 2048` can OOM on the MTP draft buffer, so use `-ub 512` with q8_0.
- **MTP has no effect** — the MTP head loads as an unused tensor unless `--spec-type draft-mtp` is set. `draft-n-max` values above 3 are slower.
- **Long-context prefill is expensive** — a single 250K prompt takes ~9 minutes to prefill; prefer batching or retrieval.
- **Only one request at a time** — `--parallel 1` serializes requests. To serve concurrent users, reduce `-c` substantially.
- **q8_0 KV above 128K is unstable** on this architecture; Recipe 2 caps the q8_0 variant at 128K.
- **Engine build** — use the CUDA 13.x Windows zip only.

## Scripts

| Script | Recipe | Purpose |
|---|---|---|
| `install-iq3s.ps1` | 1 | engine b11491 + IQ3_S + mmproj (`-TextOnly` skips vision) |
| `install-udq4.ps1` | 2 | engine b11491 + UD-Q4_K_XL |
| `start-server-iq3s-q4kv-text-image-256k.ps1` | 1 | q4_0 KV + vision, 256K, ub2048 |
| `start-server-iq3s-q8kv-text-only-256k.ps1` | 1 | q8_0 KV, no vision, 256K, ub512 |
| `start-server-udq4-q4kv-text-only-256k.ps1` | 2 | q4_0 KV, 256K, ub512 |
| `start-server-udq4-q8kv-text-only-128k.ps1` | 2 | q8_0 KV, 128K, ub2048 |
| `verify-server.ps1` | both | health check + chat |
| `benchmark.ps1` | both | benchmark harness (≤128K) |
| `benchmark-full.ps1` | both | long-context benchmark (196K–262K) |

## Editor integration (opencode)

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
          "attachment": true,            // Recipe 1 only: vision via --mmproj
          "limit": { "context": 262144, "input": 262144, "output": 65536 }
        }
      }
    }
  },
  // compact ~16K tokens before the context hard limit
  "compaction": { "reserved": 16000 },
  // optional: delegate compaction to a cloud model instead of the local GPU
  "agent": { "compaction": { "model": "deepseek/deepseek-v4-flash" } }
}
```

For Recipe 2's 128K variant, set `limit.context` / `limit.input` to `131072`. Images use the standard OpenAI `image_url` content-part format (Recipe 1 only).

## References

- llama.cpp: <https://github.com/ggml-org/llama.cpp>
- Recipe 1 model + mmproj: <https://huggingface.co/ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF>
- Recipe 2 model: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF> (UD-Q4_K_XL)
- KV quantization: llama.cpp issues #27109 (FA kernel fallback) and #21385 (q4_0 KV on hybrid models)
- opencode: <https://opencode.ai>

## License

[MIT](LICENSE) © 2026 Mike Lee
