# Inkling-2-Mini on GaudiInfer

[thinkingmachines/Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) (`model_type: inkling2_loop`)
runs on one Gaudi2 card through the Direct engine (with Synapse graphs as reference and fallback). `g2e_server` detects
the pack and serves it with the OpenAI-compatible API, the Inkling message parser (reasoning, tool calls), and image
input.

## The model in one paragraph

24 stored layers and 48 layer applications: layers 0-5 run once, layers 6-17 run three times (the loop), then layers
18-23 run once. Layers 5, 11, 17, and 23 attend to the whole context, and the others use a 1024-token sliding window.
Each application keeps its own KV cache and convolution states. Every block of 6 layers mixes the earlier block outputs
through Block Attention Residuals, a softmax over the states. In passes 2 and 3 the q/k/v/MLP inputs also pass through
message gates fed by the previous passes' outputs. Attention has no RoPE; it uses a learned relative position bias
(zero beyond 1024 for full layers), a per-head sink, a sigmoid output gate, a log length scale on the full layers'
queries, and four short convolutions per application. The MoE has 128 experts, top-8 sigmoid routing with a bias, and a
softcapped SwiGLU. Expert weights are NVFP4 in layers 0-19 and MXFP8 in layers 20-23. The vision tower is a 23-layer
ViT with a 2x2 merge and a 2-layer projector.

## Convert

```bash
# text model: native NVFP4 (layers 0-19) and MXFP8 (layers 20-23) experts, BF16 rest (17.65 GiB, pack v6)
python3 tools/convert_inkling2.py /models/Inkling-2-Mini packs/inkling2-mini.g2w --workers 6

# optional: verify every expert value in the pack against the checkpoint
python3 tools/verify_inkling2_pack.py /models/Inkling-2-Mini packs/inkling2-mini.g2w --workers 6

# vision tower + projector (BF16, 0.59 GiB)
python3 tools/convert_inkling2_vision.py /models/Inkling-2-Mini packs/inkling2-mini-vision.g2w

# DSpark drafter for speculative decoding (mtp.safetensors, BF16 + LM-head folds, 0.69 GiB)
python3 tools/convert_dspark.py /models/Inkling-2-Mini packs/inkling2-mini-dspark.g2w
```

The converters and pack verifier need only numpy; converting the text model takes about half a minute. Older pack
formats are refused at load and must be reconverted.

### Weights and numerics

The checkpoint stores expert weights as NVFP4 in layers 0-19 (4-bit E2M1 codes, one E4M3 block scale per 16 values,
and a per-expert outer factor) and MXFP8 in layers 20-23 (E4M3 codes and one power-of-two block scale per 32 values).
Pack v6 stores those formats natively. `tools/verify_inkling2_pack.py` decodes the pack independently and compares all
24,461,180,928 expert values against the checkpoint with 0 differences. Attention, norms, convolutions, embeddings,
and the LM head keep the checkpoint's BF16 values.

* Decode, batched decode, and verification on the Direct engine read the stored codes in place: TPC GEMV kernels
  accumulate bare codes per block in F32, multiply once per block by the block scale, and apply the per-expert factor
  at the end. No weight value is rounded anywhere.
* Prefill on the Direct engine and the Synapse graphs read an exact BF16 copy of the expert weights produced on the
  device by `ink_nvf4_deq_bf16_gaudi2` and `ink_mxfp8_deq_bf16_gaudi2` (about 1.6-1.8 ms per layer), through BF16
  GEMMs with BF16 activations and F32 output.

What still differs from the Hugging Face implementation: on the Direct engine the tanh of the message gates and of the
MoE activation are rational (Pade) approximations; accumulation order differs; and the Hugging Face loader itself
multiplies code x scale x factor in float32 and casts to the model dtype, so its bf16 run rounds the expert weights
whereas its float32 run does not.

## Run

Inkling uses its own SCAL config, `tools/scal-g2e-ink.json`: two plan lanes need 256 sync objects and 1424 monitors of
the compute stream, and the prefill plans use all 48 MiB of SRAM (`tpc_barrier_in_sram: false`). The stock
`tools/scal-g2e-direct.json` of the other models is not changed. Both SCAL variables must be set before the device
opens.

```bash
export SCAL_CFG_FILE_PATH=$PWD/tools/scal-g2e-ink.json SCAL_CONFIG_FILE_PATH=$PWD/tools/scal-g2e-ink.json
export G2E_RECIPE_CACHE=/path/to/recipes-inkling            # compiled Synapse graphs, kept between starts
export G2E_INK_DIRECT=15 G2E_INK_DIRECT_SRAM=1 G2E_INK_DPREFILL=15
./build/g2e_server \
    --model-dir    /models/Inkling-2-Mini \
    --target-pack  packs/inkling2-mini.g2w \
    --vision-pack  packs/inkling2-mini-vision.g2w \
    --draft-pack   packs/inkling2-mini-dspark.g2w \
    --served-model-name inkling-2-mini \
    --port 8001 --max-model-len 131072 --slots 8
```

The `G2E_INK_*` switches configure the Direct engine (see `docs/inkling2-direct-plan.md`). The server prints which
engine it runs (`[g2e] inkling2 engine: ...`):

| Variable | Production value | Meaning |
|---|---|---|
| `G2E_INK_DIRECT` | `15` | Decode, batched decode, and verification on the Direct engine. Bit mask of layer-application kinds: 1 sliding, 2 block boundary, 4 with messages (passes 2 and 3), 8 full layers. Batched decode and speculation need all four. |
| `G2E_INK_DIRECT_SRAM` | `1` | Keeps decode activations and scratch in SRAM. |
| `G2E_INK_DPREFILL` | `15` | Prefill on the Direct engine (same bit mask as `G2E_INK_DIRECT`). |
| `G2E_INK_BF16_CACHE` | `auto` | Which layers keep their expanded BF16 expert copy in HBM (`auto`, `all`, `loop`, `none`; see below). |

### The BF16 weight cache and memory

Prefill needs BF16 expert weights (1.9 GiB per layer). `G2E_INK_BF16_CACHE` selects how many layers stay resident in
HBM on the Direct engine: `all` caches all 24 layers; `loop` caches the 12 looped layers 6-17 (each used by three
applications); `none` keeps only one shared staging buffer; and `auto` (default) sizes the cache after all packs are
loaded. Uncached layers are expanded into the staging buffer before each chunk pair that needs them.

In `auto` mode, `g2e_server` reserves the KV need (`--kv-cache-gib` if set, otherwise
`min(slots * max-model-len, 1,048,576)` tokens), the drafter's KV reserve, and a 4 GiB margin, then caches as many
layers as fit (looped layers first). `/stats` reports the active cache state under `bf16_cache` and the KV budget
under `slots`.

| Start | Cached layers | KV budget |
|---|---|---|
| default (`--slots 8`, `--max-model-len` 131072 to 1048576) | 15 of 24 + staging | 33.1 GiB (1.06 M tokens) |
| `--kv-cache-gib 20` | 21 of 24 + staging | 20 GiB (0.64 M tokens) |
| `G2E_INK_BF16_CACHE=all` | all 24 | 17.9 GiB (0.57 M tokens) |
| `G2E_INK_BF16_CACHE=none` | staging only | 61.6 GiB (1.97 M tokens) |
| Synapse graphs only | all 24 (expanded at load) | 23.8 GiB (0.76 M tokens) |

8K prefill latency by cache state (engine alone): 878 ms with all 24 layers cached, 959 ms with the 12 looped layers
cached, and 1.19 s with `none`. The default server start (15 layers) is 4 to 9 % slower in prefill than all layers
cached (table under Performance).

### Fallback, templates, and Docker

Without `G2E_INK_DIRECT` and `G2E_INK_DPREFILL` the model runs on the Synapse graphs only: all 24 layers are expanded
to BF16 at load, speculation is disabled (`--draft-pack` is ignored), decode is not batched across slots, and prefill
is slower (see Performance).

The Direct engine loads MME descriptor templates from `tools/mme_descs_ink` (`G2E_INK_MME_DIR` overrides) and TPC
kernel objects from the build directory (`G2E_TPC_OBJ_DIR`). The templates were captured under SynapseAI 1.24.1 with
`ink_direct_tool capture` and carry no version stamp; for another SynapseAI version, capture them again and check them
with `ink_direct_tool gemmtest` before use. While a request runs, the engine thread polls the device's completion
counter (one busy host core; `G2E_DIRECT_SPIN=0` blocks in the driver instead, at higher per-token latency).

`G2E_RECIPE_CACHE` keeps compiled Synapse graphs on disk. The first request at a new shape (chunk size, far-context
bucket, image size) compiles its recipes once, and later starts reuse them. To fill the cache ahead of time without
reference data: `inkling2_test --pack packs/inkling2-mini.g2w --max-ctx 131072 --precompile 1 --precompile 1024`.

Docker support in `docker/entrypoint.sh` and `docker-compose.yml` accepts `G2E_MODE=inkling`, `G2E_INK_BF16_CACHE`,
and `G2E_KV_CACHE_GIB` (setting `tools/scal-g2e-ink.json` and the Direct switches automatically). Docker support for
Inkling mode is untested.

### Server behaviour

* Up to 8 requests run at once (`--slots`), each in its own slot sharing one dynamic KV budget (`--kv-cache-gib`,
  default: remaining device memory after packs, BF16 cache, drafter KV reserve, and margin). With the Direct engine,
  two or more decoding slots are batched into one forward of 2, 4, or 8 rows; prefill runs one slot at a time between
  decode steps.
* With `--draft-pack`, a request decoding alone (up to `G2E_INK_DSPARK_MAX_CTX`, default 32768) uses speculative
  decoding: each round runs either one plain decode step or verifies the DSpark drafter's first 1-3 tokens in one
  Direct forward of 2-4 rows, chosen from the running acceptance estimate. Greedy requests preserve the greedy output
  (up to near-ties); sampled requests use rejection sampling, so the output distribution is unchanged. `G2E_INK_SPEC=0`
  disables speculation at runtime, and `G2E_INK_NO_SPEC=1` skips loading the drafter.
* `reasoning_effort` is passed to the chat template (`"off"` maps to `"none"`). Reasoning is returned in
  `message.reasoning` (streaming: `delta.reasoning`), separate from `content`.
* Tool calls (`invoke_tool_json`) are returned as OpenAI `tool_calls`.
* Images use `image_url` parts with `data:` URLs (base64). The processor does not resize: one image token per 40x40
  pixel cell, padded. Images over 4096 tokens are shrunk to fit (`--max-image-pixels` sets the limit, at 1600 pixels
  per token).
* Prefix reuse across requests uses state snapshots in pinned host memory (KV, convolution states, drafter state),
  managed LRU within `G2E_INK_SNAPSHOT_GB` (default 32; `0` disables snapshots). They are allocated on demand and
  require an adequate memlock limit (`docker-compose.yml` sets memlock to unlimited); when an allocation fails, the
  snapshot is skipped. Images enter the snapshot key.

## Validate

Reference dumps are produced on the CPU with `tools/inkling2_ref/dump_golden.py` using a Hugging Face `transformers`
build with `Inkling2LoopForConditionalGeneration` (5.24.0 here). Rendering the checkpoint's chat template requires
`jinja2 >= 3.1.6`. `INKLING2_MODEL_DIR` names the checkpoint directory (default `/models/Inkling-2-Mini`). Use
`--dtype float32` for an unrounded reference or `--dtype bfloat16` (default) for the bf16 reference.

```bash
python3 tools/verify_inkling2_pack.py /models/Inkling-2-Mini packs/inkling2-mini.g2w --workers 6

python3 tools/inkling2_ref/dump_golden.py golden --long 1600 --dtype bfloat16 && \
    python3 tools/inkling2_ref/export_golden.py golden
python3 tools/inkling2_ref/dump_golden.py golden-f32 --long 1600 --dtype float32 && \
    python3 tools/inkling2_ref/export_golden.py golden-f32
python3 tools/inkling2_ref/dump_vision_golden.py /models/Inkling-2-Mini golden/vision

./build/inkling2_test --pack packs/inkling2-mini.g2w --bf16-cache-check
./build/inkling2_test --pack packs/inkling2-mini.g2w --golden golden-f32/short --decode 16 --bench 32
./build/inkling2_test --pack packs/inkling2-mini.g2w --golden golden-f32/long --max-ctx 65536 --decode 16
./build/inkling2_test --pack packs/inkling2-mini.g2w --golden golden/long --max-ctx 65536 --decode 0 \
    --prefill-bench 36000 --prefill-decode 16
./build/inkling_vision_test --pack packs/inkling2-mini-vision.g2w --golden golden/vision
```

Validation results (`inkling2_test`, Direct engine, pack v6; 27-token short prompt and 1625-token long prompt, 16
greedy decode tokens):

| Reference | Short prompt | Long prompt |
|---|---|---|
| Hugging Face float32 | top-1 16/16, mean KL 0.00006 | top-1 16/16, mean KL 0.00104 |
| Hugging Face bf16 | top-1 16/16, mean KL 0.00006 | top-1 16/16, mean KL 0.00255 |

The two Hugging Face references differ from each other by mean KL 0.00013 on the short prompt and 0.00132 on the long
prompt; on the long prompt one decode step sits at a near-tie where the bf16 and float32 runs disagree by KL 0.017.

Other checks that pass:

* `--direct-check` (Direct against Synapse graphs): top-1 same, KL 0.0005.
* `--long-check 9000` (32 decode steps at 9K context, Direct vs Synapse): 32/32 top-1, max KL 0.0001.
* `--verify-check` (4-row verification, Direct vs Synapse): rows 0-2 top-1 same; row 3 sits at a near-tie, KL 0.002.
* `--batch-check` (B = 2..8 against single-row decode over 16 steps per slot): argmax equal, per-slot median KL at
  most 0.0004.
* `--spec-test 64` (DSpark greedy speculation): identical to plain greedy for the first 26 tokens, then diverges at a
  near-tie.
* `--ctx-1m-test` (135K and 1M position prefill and decode): finite outputs and consistent tokens.
* `--bf16-cache-check`: verifies cached and staged on-device BF16 weights against a host decode.
* Synapse graphs only (`golden/short`, 16 greedy tokens): 16/16 top-1, mean KL 0.00013.
* `inkling_vision_test`: checks image preprocessing and vision tower features against the Hugging Face reference.

Near-ties: the router has experts whose score sits near the top-8 cutoff while carrying a large route weight; a
rounding-level difference flips the choice and shifts that row's logits significantly. On a 1500-token test prompt,
the Hugging Face bf16 and float32 runs themselves disagree by KL 0.02-0.06 on four consecutive steps (with three
different top-1 tokens on one of them). The multi-step checks therefore judge medians over steps rather than maxima.
Additional test flags are listed in `docs/inkling2-direct-plan.md`.

## Performance (one Gaudi2)

Measured 2026-11-25 with SynapseAI 1.24.1 on one HL-225 attached through an M.2 OCuLink port (PCIe Gen4 x4). Timings
are taken on the second run after a build (the first run compiles Synapse recipes).

| Workload | Measured |
|---|---|
| Decode, one request (short context) | 8.0 ms/token (125 tok/s) |
| Decode, batched, 2 / 4 / 8 requests | 11.2 / 15.3 / 23.3 ms per step (8 requests: 343 tok/s total) |
| Greedy with the DSpark drafter | 6.3-6.9 ms/token on the test text (about 2.9 tokens accepted per round) |
| Prefill, 8K / 36K prompt, all layers cached | 878 ms (9.33K tok/s) / 4.76 s (7.56K tok/s) |
| 2K-token prefill at a 135K / 1M-token position | 543 ms / 2.34 s |
| Decode at a 135K / 1M-token position | 11.0 / 25.7 ms/token |
| Synapse graphs only: decode / 4K prefill | 18 ms/token / about 4.0K tok/s |

Server (`g2e_server --max-model-len 262144 --slots 8`, all three packs), one request at a time, 128 generated tokens,
sampled (`temperature` 0.7), median of six workload types per prompt length. Prefill is prompt tokens over the time
to the first token:

| Prompt tokens | Default cache (15 of 24 layers) | | All layers cached (`G2E_INK_BF16_CACHE=all`) | |
|---|---|---|---|---|
| | first token | prefill | first token | prefill |
| 1,024 | 206 ms | 4,971 tok/s | 191 ms | 5,351 tok/s |
| 2,048 | 343 ms | 5,978 tok/s | 332 ms | 6,163 tok/s |
| 4,096 | 510 ms | 8,032 tok/s | 465 ms | 8,804 tok/s |
| 8,192 | 974 ms | 8,414 tok/s | 914 ms | 8,962 tok/s |
| 16,384 | 1.98 s | 8,271 tok/s | 1.89 s | 8,682 tok/s |
| 32,768 | 4.16 s | 7,877 tok/s | 3.93 s | 8,328 tok/s |
| 65,536 | 9.34 s | 7,019 tok/s | 8.87 s | 7,389 tok/s |
| 131,072 | 22.7 s | 5,774 tok/s | 21.7 s | 6,029 tok/s |

Decode in these runs (with the drafter, acceptance varies with the text) is 91 to 132 tok/s, lowest at the 131K
context. A 512-token prompt takes about 150 ms to the first token.

Other server measurements (default start): a 6360-token prompt takes 1.26 s to the first token; repeating the same
prompt takes 0.36 s via the host-RAM snapshot; 8 concurrent requests of 21-3627 prompt tokens complete in 5.5 s.

## How the engine works

Direct decode, batched decode, and verification (`src/inkling2_direct.cpp`): a single-row decode step
(`InkDirect::run_decode_full`) chains token embedding (`ink_dembed_gaudi2`), all 48 layer applications grouped into
multi-application spans (`G2E_INK_DSPAN`), final depth attention and norm (`ink_dgram_gaudi2`, `ink_dnorm_gaudi2`),
LM head (`d_lmhead`), and fused Pade softcap plus argmax (`ink_dsoftcap_argmax_gaudi2`) with no Synapse calls. Expert
GEMVs read native NVFP4 or MXFP8 weights directly from HBM using block accumulation in F32
(`ink_dmoe_up9_nvf4_gaudi2`, `ink_dmoe_dn9_nvf4_gaudi2`, `ink_dmoe_up9_mxf8_gaudi2`, `ink_dmoe_dn9_mxf8_gaudi2`).
Batched decode (`B` padded to 2, 4, or 8 via `run_decode_batch`) and speculative verification (`T = 2..4` via
`run_verify_full`) deduplicate routed experts across rows (`ink_vmoe_plan_gaudi2` feeding
`ink_vmoe_{up,dn}_{nvf4,mxf8}_gaudi2`) and defer KV/convolution commits on verification until `accept(n)`. Full-layer
far attention switches to reusable patched tiled FlashDecode plans up to 1M tokens.

Direct prefill (`src/inkling2_dprefill.cpp`): prompts run in 1024-token chunks (`T = 1024`; tails of 5-1023 tokens pad
to 1024 with a valid-row mask). Pairs of chunks (`run_pair`) interleave Plan B (MoE and post-MLP convolution) of one
chunk with Plan A (attention, convolutions, message gates, router, and routing histogram/table) of the next across two
Direct plan lanes (`DirectRuntime::run_waves`). Plan A places near-attention buffers in a 48 MiB SRAM overlay and,
past 32K far keys, runs pooled in-place SRAM FlashAttention (`ink_pfar_head_io_gaudi2`, `ink_attn_far_step_gaudi2`).
For each application's Plan B, the host reads the 128-entry expert count table via PDMA, builds 4-wave dynamic BF16
grouped-GEMM descriptors (`DirectDynMmeSet`, templates `p_bgt*` and `p_bdt*`, F32 output) over at most 12,288 flat
rows (zero overflow), and stages uncached layers into the shared BF16 buffer with `ink_nvf4_deq_bf16_gaudi2` or
`ink_mxfp8_deq_bf16_gaudi2` before the chunk pair uses them. TPC stages `ink_rplan_dyn_slots_bf16_gaudi2`,
`ink_sgather_bf16_gaudi2`, `ink_act_bf16_gaudi2`, and `ink_moe_combine_bf16_gaudi2` surround the MME waves.

Synapse graphs (`src/inkling2.cpp`): the reference and fallback path when the Direct switches are unset. At load,
`prepare_synapse_bf16` expands all 24 layers into BF16 using `ink_nvf4_deq_bf16_gaudi2` and
`ink_mxfp8_deq_bf16_gaudi2`. Each layer kind compiles one Synapse recipe
(`ink_{F|S}_n<block>[b][z][m]_bf16_T<T>[_f<far>][_qb<Tq>]`) using BF16 batched expert GEMMs (`moe_bf16_small` for
`T <= 4`, zero-overflow capacity tiers `moe_bf16_cap` for larger `T`). A forward runs 48 layer launches writing new
K/V rows to scratch, followed by one `commit_recipe` launch (`ink_kv_commit_bf16_gaudi2`) that writes them into the
persistent rings and linear caches.

## Layout of the implementation

* `include/g2e/inkling2.hpp`, `src/inkling2.cpp`: model state, multi-slot dynamic KV management, pack v6 validation,
  Synapse graph builders, and load-time Synapse BF16 expansion.
* `include/g2e/inkling2_direct.hpp`, `src/inkling2_direct.cpp`, `src/inkling2_dprefill.cpp`: `InkDirect` decode,
  batched decode, verification, and prefill plans, plus the BF16 weight cache and staging logic.
* `include/g2e/direct_mme.hpp`, `src/direct_mme.cpp`: shared Direct runtime (`DirectRuntime`, `DirectPlan`,
  `DirectDynMmeSet`), plan lanes, wave execution, diff-encoded MME descriptors, scheduler PDMA transfers, and polled
  fence waits (`DirectRuntime::Options`, off unless enabled by a model's engine).
* `src/inkling2_dspark.cpp`: `InkDSpark` device drafter. `tools/dspark_ref.py`: its CPU reference.
* `src/inkling_vision.cpp`: vision tower (`InklingVision`). `src/inkling_parser.cpp`: streaming message parser.
* `tpc/flashinfer_gaudi/kernels/ink_*_gaudi2.c`: TPC kernels for Direct and Synapse paths.
* `tools/mme_descs_ink/`: captured MME descriptor templates (`apps/ink_direct_tool.cpp`).
* `tools/convert_inkling2.py`, `tools/verify_inkling2_pack.py`, `tools/convert_inkling2_vision.py`,
  `tools/convert_dspark.py`: converters and pack verifier.
* `apps/inkling2_test.cpp`, `apps/ink_direct_tool.cpp`, `apps/inkling_vision_test.cpp`: validation and kernel tools.
* Synapse-path debug switches: `G2E_INK_PROFILE=1` (per-recipe device time), `G2E_INK_REF=1` (graph-op reference
  path), `G2E_INK_KMASK` (fused kernels bit mask), `G2E_INK_CHUNK`, `G2E_INK_SCORE_ROWS`, `G2E_INK_QBLOCKS`,
  `G2E_INK_MG_NOLUT=1`, `G2E_INK_PAD_TAIL=0`.
  Direct-engine switches are listed in `docs/inkling2-direct-plan.md`.

## Not done yet

* Batched decode pays for weight unpacking in kernels that serve several rows per expert (8 requests: 23.3 ms per
  step).
* The MXFP8 decode kernels spend about 1.6x the time of a plain FP8 GEMV on their 4 applications.
* Prefill MME work is dominated by the per-expert BF16 weight read. One MoE step per two chunks (four chunks in
  flight) was measured at about 10 % less time for an 8K prompt in the engine alone, but its plans do not fit the
  plan pool (about 125 MiB usable) next to the decode, verification, and two-chunk prefill plans the server keeps, so
  it is not included; it needs smaller plans first (for example plans shared across layer applications).
* A short request that directly follows a very long one starts late: a 512-token prompt right after a 131K-token one
  took 0.7 to 2.1 s to the first token instead of 0.15 s. The cause has not been investigated.
* Single-request decode remains limited in part by TPC and MME stage boundary latency across the 48 applications.
* MME descriptor templates carry no SynapseAI version stamp; they were captured under 1.24.1.
* Speculation runs for a request that decodes alone (within `G2E_INK_DSPARK_MAX_CTX`, default 32768); batched decode
  steps are not speculative.
