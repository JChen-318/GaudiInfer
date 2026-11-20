<div align="center">

# GaudiInfer

**为 Intel® Gaudi®2 打造的高性能、高吞吐大模型推理引擎**

原生 C++ · 直接调用 SynapseAI · DSpine 投机解码 · 连续批处理 · 自研 TPC 内核

[![Hardware](https://img.shields.io/badge/hardware-Intel%20Gaudi2-0071C5)](#环境要求)
[![Language](https://img.shields.io/badge/C%2B%2B-20-00599C?logo=cplusplus&logoColor=white)](#编译)
[![SDK](https://img.shields.io/badge/SynapseAI-1.24-6E40C9)](#环境要求)
[![API](https://img.shields.io/badge/API-OpenAI%20compatible-10A37F)](#api)
[![Model](https://img.shields.io/badge/model-Qwen3.8--27B%20FP8-FF6A00)](#模型准备)

[English](README.md) · **简体中文**

</div>

---

GaudiInfer（代码里叫 `g2e`）在单张 Gaudi2 上运行 **Qwen3.8-27B（FP8）**。它直接调用 SynapseAI 的图接口，服务路径上不用 PyTorch，也没有 Python。生成时用 **DSpine** 投机草稿模型加速，最多 8 个请求同时运行，各请求的验证和起草都合批执行。

## 亮点

| | |
|---|---|
| ⚡ **单路解码快** | DSpine 每轮起草 16 个 token，目标模型一次前向全部验证。贪心和采样请求都走投机解码；采样请求用拒绝采样，输出分布不变。 |
| 🚀 **连续批处理** | 动态槽位共享一份 KV 预算。调度器每一步先跑解码轮，再按到达顺序做 prefill。最多 8 个请求的验证和起草各合成一次前向。 |
| 📚 **长上下文** | 最长 **262,144 token**。prefill 用 FP8 SDPA，验证用自研的 masked softmax 内核。可选 FP8 KV 缓存（`G2E_KV_FP8=1`），容量翻倍。 |
| 🧠 **前缀缓存** | 在主机内存里按 LRU 缓存 KV 块和 GDN、草稿模型的状态快照，配锁页内存池和后台锁页。多轮 agent 对话几乎不用重新 prefill。 |
| 🔧 **自研 TPC 内核** | 为这个模型专门写的 TPC-C 内核：WY 形式的 GDN（门控 delta-net）递推和单 token 解码、融合 RMSNorm + FP8 量化、门控归一化 + 量化、SiLU 乘法 + 量化、masked softmax、因果卷积。 |
| 🌐 **兼容 OpenAI 的服务** | `/v1/chat/completions` 和 `/v1/completions`，支持流式输出。C++ 实现的分词器和 Jinja 对话模板（与 HF 结果一致）、Qwen3 工具调用解析、可选的推理内容解析、带 `cached_tokens` 的用量统计，以及图片输入。 |

## 性能

用 **[llm_speedtest](https://github.com/gengchaogit/llm_speedtest)** 的全场景顺序测试，在一张 Gaudi2 上实测：Qwen3.8-27B FP8 加 DSpine-b16 草稿模型，通过 OpenAI 兼容 API 一次跑一个请求，6 类任务 × 9 种 prompt 长度（512 到 131,072 token）。**54 组全部成功。**

| 单路解码 | 出字间隔（P50） | prefill 峰值 | 128K prompt 首 token |
|:---:|:---:|:---:|:---:|
| **205 tok/s**<br><sub>6 类任务平均，189–219</sub> | **24.7 ms**<br><sub>各任务 24.5–24.9</sub> | **6,161 tok/s**<br><sub>8K prompt</sub> | **36.4 s**<br><sub>通用混合任务</sub> |

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/prefill-dark.svg">
  <img alt="prefill 吞吐随 prompt 长度的变化" src="docs/images/prefill-light.svg" width="760">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/decode-workload-dark.svg">
  <img alt="各任务的解码吞吐" src="docs/images/decode-workload-light.svg" width="760">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/decode-length-dark.svg">
  <img alt="解码吞吐随 prompt 长度的变化" src="docs/images/decode-length-light.svg" width="760">
</picture>

- **Prefill**：2K 到 32K 的 prompt 约 5.5K–5.8K tok/s，128K 时注意力占大头，仍有约 3.5K tok/s。
- **解码**：上下文到 128K 时中位数仍高于 160 tok/s，出字间隔只从 23 ms 增加到 30 ms。
- **任务之间**相差 1.16 倍：投机解码在数理逻辑上收益最大，在自由行文的散文上最小。

**按任务**

| 任务 | 解码（tok/s） | 相对通用混合 | 首 token P50 | 出字间隔 P50 |
|---|---:|---:|---:|---:|
| 通用混合 | 203.96 | 基准 | 6,822 ms | 24.9 ms |
| 代码解码 | 200.32 | -1.8% | 6,896 ms | 24.6 ms |
| 结构化 JSON | 207.69 | +1.8% | 6,897 ms | 24.7 ms |
| 散文问答 | 188.82 | -7.4% | 8,118 ms | 24.8 ms |
| 文档知识库 | 211.00 | +3.5% | 7,731 ms | 24.7 ms |
| 数理逻辑 | 218.95 | +7.3% | 6,911 ms | 24.5 ms |

**按 prompt 长度**（6 类任务的平均值；解码取中位数）

| prompt（token） | 首 token | 出字间隔 | prefill（tok/s） | 解码（tok/s） |
|---:|---:|---:|---:|---:|
| 512 | 156 ms | 22.9 ms | 3,704 | 225 |
| 1,024 | 235 ms | 23.2 ms | 4,731 | 227 |
| 2,048 | 390 ms | 23.0 ms | 5,552 | 229 |
| 4,096 | 741 ms | 23.5 ms | 5,647 | 239 |
| 8,192 | 1,457 ms | 23.5 ms | 5,847 | 210 |
| 16,384 | 2,892 ms | 23.9 ms | 5,711 | 195 |
| 32,768 | 6,201 ms | 25.0 ms | 5,480 | 190 |
| 65,536 | 14.6 s | 26.6 ms | 4,655 | 175 |
| 131,072 | 38.6 s | 30.4 ms | 3,524 | 162 |

<details>
<summary><b>全部 54 组结果</b></summary>


**通用混合**

| prompt | 首 token（ms） | 出字间隔（ms） | prefill（tok/s） | 解码（tok/s） |
|---:|---:|---:|---:|---:|
| 512 | 101.59 | 23.10 | 5,167.04 | 251.76 |
| 1,024 | 185.17 | 23.15 | 5,601.95 | 307.09 |
| 2,048 | 335.01 | 23.47 | 6,155.62 | 272.52 |
| 4,096 | 668.10 | 24.69 | 6,152.20 | 272.77 |
| 8,192 | 1,334.70 | 23.54 | 6,160.71 | 247.00 |
| 16,384 | 2,737.77 | 23.76 | 5,969.91 | 97.90 |
| 32,768 | 5,873.38 | 25.12 | 5,581.41 | 169.82 |
| 65,536 | 13,763.80 | 27.09 | 4,755.54 | 143.09 |
| 131,072 | 36,400.16 | 30.50 | 3,601.10 | 73.63 |

**代码解码**

| prompt | 首 token（ms） | 出字间隔（ms） | prefill（tok/s） | 解码（tok/s） |
|---:|---:|---:|---:|---:|
| 512 | 150.84 | 23.34 | 3,522.43 | 220.42 |
| 1,024 | 239.42 | 23.70 | 4,485.58 | 224.97 |
| 2,048 | 393.45 | 23.55 | 5,336.92 | 208.90 |
| 4,096 | 730.03 | 23.57 | 5,682.59 | 208.78 |
| 8,192 | 1,403.01 | 22.93 | 5,873.46 | 169.09 |
| 16,384 | 2,797.60 | 23.44 | 5,875.69 | 194.96 |
| 32,768 | 5,959.40 | 24.66 | 5,507.37 | 225.63 |
| 65,536 | 13,917.90 | 26.69 | 4,712.42 | 169.95 |
| 131,072 | 36,880.34 | 29.74 | 3,594.26 | 172.09 |

**结构化 JSON**

| prompt | 首 token（ms） | 出字间隔（ms） | prefill（tok/s） | 解码（tok/s） |
|---:|---:|---:|---:|---:|
| 512 | 162.95 | 23.14 | 3,430.92 | 230.39 |
| 1,024 | 241.00 | 22.63 | 4,483.09 | 226.19 |
| 2,048 | 395.29 | 22.55 | 5,313.79 | 210.07 |
| 4,096 | 730.23 | 23.49 | 5,682.30 | 209.45 |
| 8,192 | 1,407.15 | 25.07 | 5,860.05 | 260.60 |
| 16,384 | 2,805.34 | 24.20 | 5,859.58 | 195.86 |
| 32,768 | 5,953.50 | 24.20 | 5,512.97 | 195.80 |
| 65,536 | 13,892.25 | 26.32 | 4,721.18 | 180.00 |
| 131,072 | 36,502.01 | 31.03 | 3,592.17 | 152.73 |

**散文问答**

| prompt | 首 token（ms） | 出字间隔（ms） | prefill（tok/s） | 解码（tok/s） |
|---:|---:|---:|---:|---:|
| 512 | 181.78 | 22.19 | 3,320.22 | 192.33 |
| 1,024 | 233.68 | 22.85 | 5,040.11 | 180.62 |
| 2,048 | 404.62 | 23.08 | 5,740.24 | 226.05 |
| 4,096 | 817.83 | 23.82 | 5,672.01 | 260.52 |
| 8,192 | 1,636.13 | 23.34 | 5,633.61 | 210.06 |
| 16,384 | 3,153.77 | 23.57 | 5,040.91 | 226.22 |
| 32,768 | 6,900.97 | 25.49 | 5,361.60 | 161.93 |
| 65,536 | 16,299.45 | 27.13 | 4,533.99 | 117.94 |
| 131,072 | 43,849.19 | 30.93 | 3,394.15 | 114.92 |

**文档知识库**

| prompt | 首 token（ms） | 出字间隔（ms） | prefill（tok/s） | 解码（tok/s） |
|---:|---:|---:|---:|---:|
| 512 | 165.42 | 23.42 | 3,623.69 | 246.17 |
| 1,024 | 286.86 | 23.46 | 3,997.82 | 227.28 |
| 2,048 | 431.47 | 23.07 | 5,227.45 | 231.07 |
| 4,096 | 787.53 | 22.93 | 5,673.05 | 232.46 |
| 8,192 | 1,533.54 | 22.71 | 5,769.71 | 152.26 |
| 16,384 | 3,009.24 | 24.42 | 5,751.92 | 249.52 |
| 32,768 | 6,547.09 | 24.70 | 5,422.93 | 184.43 |
| 65,536 | 15,491.90 | 26.22 | 4,503.54 | 195.20 |
| 131,072 | 41,260.62 | 30.80 | 3,451.12 | 180.62 |

**数理逻辑**

| prompt | 首 token（ms） | 出字间隔（ms） | prefill（tok/s） | 解码（tok/s） |
|---:|---:|---:|---:|---:|
| 512 | 176.31 | 22.45 | 3,159.70 | 211.07 |
| 1,024 | 224.84 | 23.63 | 4,776.96 | 270.72 |
| 2,048 | 379.25 | 22.49 | 5,536.05 | 247.37 |
| 4,096 | 712.69 | 22.71 | 5,022.32 | 244.95 |
| 8,192 | 1,424.67 | 23.49 | 5,786.17 | 209.45 |
| 16,384 | 2,850.73 | 23.95 | 5,765.30 | 172.33 |
| 32,768 | 5,974.09 | 26.03 | 5,492.70 | 245.72 |
| 65,536 | 13,945.72 | 25.95 | 4,702.09 | 197.26 |
| 131,072 | 36,531.26 | 29.62 | 3,509.22 | 171.65 |

</details>

### 与 vLLM-gaudi 对比

对比对象是我们之前在同一张卡、同一个模型上调优过的 vLLM-gaudi 部署：

| | vLLM-gaudi | **GaudiInfer** |
|---|---:|---:|
| prefill，8K / 32K / 64K prompt | 1.46 / 6.49 / 14.89 s | **1.34 / 5.78 / 13.65 s** |
| 单路解码，对话 prompt | 约 180 tok/s | **约 357 tok/s**（贪心） |
| 105K 上下文每轮验证 | 37.5 ms | **29.5 ms** |

**并发服务**：8 路各带 11K 到 18K 上下文、各生成 2048 token（TokenSpeed 并发测试，默认采样），端到端约 **650 tok/s**，8 路同时解码时约 **800 tok/s**。**Agent 回合**：90K 上下文、新增 300 token 时，借助前缀缓存，首 token 约 **285 ms**。

> tok/s 取决于草稿 token 被接受的数量，会随内容变化；每轮耗时是更稳定的比较口径。

## 架构

```mermaid
flowchart LR
    C([OpenAI 客户端]) -->|HTTP / SSE| S[g2e_server<br/>对话模板 · 分词器<br/>工具调用和推理内容解析]
    S --> Q[调度器<br/>解码优先，prefill 按到达顺序<br/>动态 KV 槽位]
    Q --> G[Generator<br/>投机解码轮<br/>拒绝采样]
    G --> D[DSpine 草稿模型<br/>5 层 BF16 环形缓存<br/>批量起草]
    G --> T[Qwen3.8-27B 目标模型<br/>FP8 权重 · GDN 与注意力混合<br/>批量验证]
    G <--> P[(前缀缓存<br/>主机内存 LRU · 锁页池)]
    D --> R
    T --> R[Synapse recipe<br/>编译后的图，缓存在磁盘]
    R --> K[自研 TPC 内核<br/>+ SynapseAI 内核]
    K --> HW[[Gaudi2：MME · TPC · HBM]]
```

每个请求每轮做一次投机解码，多个请求合批执行：

1. **起草**：DSpine 根据目标模型捕获的隐藏特征，提出 16 个候选 token。
2. **验证**：目标模型一次前向给 16 个位置打分，注意力和 GDN 状态按槽位分开。
3. **接收**：贪心请求直接和目标模型的选择比较，采样请求用拒绝采样。接收的前缀提交进 KV 缓存和 GDN 状态，其余回滚。

Synapse 以上的部分全是 C++。每种形状（上下文分档、批大小）的图只编译一次并缓存到磁盘，之后启动约 30 秒。

## 环境要求

- Intel **Gaudi2** 卡和 Habana 驱动。
- **SynapseAI 1.24**（头文件、`libSynapse`、`libtpc_kernels`）和 TPC 编译器 `tpc-clang`。
- CMake ≥ 3.24、支持 C++20 的编译器，推荐用 Ninja。
- Python 3 和 NumPy，只有转换权重时需要。
- 显存：96 GiB HBM 减去权重、草稿模型和工作区之后，剩下的都作为 KV 缓存（自动预算约 51 GiB，约 81.9 万 token）。

## 编译

```bash
git clone https://github.com/JChen-318/GaudiInfer.git
cd GaudiInfer
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

会编译引擎库、自研 TPC 内核库（`tpc/flashinfer_gaudi`）和服务器 `build/g2e_server`。SDK 不在 `/usr` 下时，设置 `SYNAPSEAI_ROOT`。

## 模型准备

先把模型转换成 `.g2w` 权重包，只需要做一次：

```bash
# 目标模型：Qwen3.8-27B FP8（128x128 块缩放转成按行缩放，合并投影矩阵）
python3 tools/convert_qwen35.py  /models/qwen3.8-27b-fp8  packs/qwen3.8-27b.g2w

# 投机草稿模型：DSpine-b16（BF16）
python3 tools/convert_dspine.py  /models/qwen3.8-27b-dspine-b16  packs/dspine-b16.g2w

# 可选：视觉编码器，用于图片输入
python3 tools/convert_vision.py  /models/qwen3.8-27b-fp8  packs/vision.g2w
```

把权重包放在内存盘上（比如 `/dev/shm`），服务器启动可以缩短到约 30 秒。

## 运行

```bash
sudo ./build/g2e_server \
    --model-dir    /models/qwen3.8-27b-fp8 \
    --target-pack  packs/qwen3.8-27b.g2w \
    --draft-pack   packs/dspine-b16.g2w \
    --vision-pack  packs/vision.g2w \
    --served-model-name qwen3.8-27b \
    --port 8000 --max-model-len 262144 --slots 8 \
    --reasoning-parser qwen3
```

设置 `G2E_RECIPE_CACHE=/path/to/dir` 可以让编译好的图在重启后继续使用；第一次启动要编译，约 10 分钟。`tools/server_ctl.sh start|stop|restart` 把上面的命令包装成后台服务。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--max-model-len` | 32768 | prompt 加生成的最大长度（最大 262144） |
| `--slots` | 8 | 同时驻留的请求数 |
| `--kv-cache-gib` | 0（自动） | 所有槽位共享的 KV 预算 |
| `--draft-pack` | 无 | 开启 DSpine 投机解码 |
| `--vision-pack` | 无 | 开启图片输入（`data:` URL） |
| `--reasoning-parser` | 无 | `qwen3`：把 `<think>` 内容拆到 `reasoning_content` |
| `--api-key` | 无 | 要求请求带 `Authorization: Bearer <key>` |

## API

```bash
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen3.8-27b",
  "messages": [{"role": "user", "content": "用两句话解释投机解码。"}],
  "stream": true
}'
```

| 接口 | |
|---|---|
| `POST /v1/chat/completions` | 对话，流式或非流式；工具调用；图片；`stream_options.continuous_usage_stats` |
| `POST /v1/completions` | 普通补全 |
| `GET /v1/models`、`/health`、`/version` | 模型列表和存活检查 |
| `GET /stats` | 槽位、KV 用量、前缀缓存命中、吞吐 |

请求和响应格式与 vLLM 的 OpenAI 兼容服务一致，现有客户端不用改。

## 环境变量开关

| 变量 | 作用 |
|---|---|
| `G2E_RECIPE_CACHE=DIR` | 编译好的 Synapse 图的磁盘缓存目录 |
| `G2E_KV_FP8=1` | FP8（E4M3）KV 缓存：KV 和前缀缓存容量翻倍，速度不变 |
| `G2E_NO_BATCH=1` | 各请求逐个跑，不合批 |
| `G2E_BATCH_PROF=1`、`G2E_VERIFY_PROF=1` | 批量轮各部分的耗时 |
| `G2E_PREFILL_LOG=1` | 每次 prefill 前向和状态快照的耗时 |

## 目录结构

```
apps/                 g2e_server：HTTP 服务、调度器、OpenAI 请求和响应处理
include/g2e/, src/    引擎：运行时、图构建、算子、Qwen3.8 目标模型、GDN、DSpine 草稿模型、
                      Generator、前缀缓存、分词器、视觉编码器、Qwen3 解析器
tpc/flashinfer_gaudi/ 自研 TPC-C 内核及其 Synapse 接入代码
tools/                权重转换（生成 .g2w 权重包）和服务启动脚本
third_party/          cpp-httplib、nlohmann/json、stb_image、llama.cpp 的 unicode 和 jinja（随仓库附带）
```

## 进度与计划

GaudiInfer 仍在积极开发中，目前面向单卡、单个模型。

- [x] FP8 目标模型、DSpine 投机解码（贪心和采样）、262K 上下文
- [x] 连续批处理，批量验证和批量起草
- [x] 前缀缓存、图片输入、OpenAI 兼容 API
- [x] 可选的 FP8 KV 缓存
- [ ] 树形投机（每轮多条草稿分支）
- [ ] 融合的长上下文注意力（分数和概率留在 SRAM）
- [ ] 8 路合计解码 ≥ 1000 tok/s

## 致谢

- [cpp-httplib](https://github.com/yhirose/cpp-httplib)、[nlohmann/json](https://github.com/nlohmann/json)、[stb_image](https://github.com/nothings/stb)，以及 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的 unicode 和 Jinja 代码（放在 `third_party/`，附各自的许可证）。
- Qwen 团队的 Qwen3.8 模型，以及 DSpine 的作者提供的投机草稿模型。
- Intel Habana 的 Gaudi2 SynapseAI SDK 和 TPC 工具链。
