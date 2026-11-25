# Inkling-2-Mini on the Direct engine

`InkDirect` (`include/g2e/inkling2_direct.hpp`, `src/inkling2_direct.cpp`, `src/inkling2_dprefill.cpp`) runs
Inkling-2-Mini decode, batched decode, speculative verification, and prefill on `DirectRuntime`
(`include/g2e/direct_mme.hpp`, `src/direct_mme.cpp`). Prepared plans patch captured MME descriptors with buffer
addresses, launch TPC kernels directly through hardware descriptors, synchronize stages with sync objects (SOBs) and
monitors, and chain plans on the scheduler ARC. The Synapse graph path (`src/inkling2.cpp`) remains as reference and
fallback when `G2E_INK_DIRECT` and `G2E_INK_DPREFILL` are unset.

## Runtime, plans, lanes, and limits

`DirectRuntime` borrows the active Synapse device's SCAL instance. Inkling requires `SCAL_CFG_FILE_PATH` and
`SCAL_CONFIG_FILE_PATH` pointing to `tools/scal-g2e-ink.json` before `Device` construction:

* `mme_attention_sos`: 256 sync objects; `mme_attention_monitors`: 1424 monitors (`InkDirect` checks for at least 129
  SOBs and 1396 monitors).
* `tpc_barrier_in_sram: false`, so `DirectRuntime::sram_usable()` exposes all 48 MiB of SRAM (prefill falls back to a
  reduced layout if usable SRAM is smaller).

`InkDirect` enables three opt-in `DirectRuntime::Options` (off by default for other models, each overridable with `0`
or `1`): `spin_wait` (`G2E_DIRECT_SPIN`), `mme_diff` (`G2E_DIRECT_MME_DIFF`), and `pdma_patches`
(`G2E_DIRECT_PDMA_PATCH`).

Per-plan resource limits (`src/direct_mme.cpp`):

* Lane 0 (single plan or `DirectRuntime::run_chain`): up to 255 private SOBs (1 reserved for chaining), 48 waiting MME
  stages (monitors `[0, 96)`), and up to 55 waiting TPC stages (monitors `[96, 1420)`).
* Lanes 1 and 2 (`DirectPlan::set_lane`, used concurrently in `DirectRuntime::run_waves`): disjoint halves. Lane 1
  uses SOBs `[0, 64)`, 24 waiting MME stages (`[0, 48)`), 27 waiting TPC stages (`[96, 744)`), and QMAN fence 3. Lane 2
  uses SOBs `[64, 128)`, 24 waiting MME stages (`[48, 96)`), 27 waiting TPC stages (`[744, 1392)`), and QMAN fence 2.
* Per TPC stage: at most 16 tensors, 32 SRFs, and 24 TPC cores.

Prepared ECB code and metadata live in SCAL `hbm_shared` and `host_shared`. `DirectPlan` supports in-place address and
SRF patching (`patch_addr`, `patch_srf`, uploaded via `flush_patches` / `flush_patches_and_upload`), optional-stage
bit masks (`set_opt_id`, `set_opt_mask`), and dynamic wait targets (`set_next_dyn_wait`, `set_dyn_wait`). When plan
pool allocation fails on a new far-context bucket, `build_evicting` drops cached plans of inactive buckets and
retries.

## Decode, batched decode, and verification

Single-row decode (`T = 1`, `InkDirect::run_decode_full`) runs the entire token in one hardware-chained `run_chain`
with no Synapse calls: `ink_dembed_gaudi2`, 48 layer applications grouped into multi-application span plans
(`G2E_INK_DSPAN`, default 8), final depth attention and norm (`ink_dgram_gaudi2`, `ink_dnorm_gaudi2`), `d_lmhead`, and
fused Pade softcap plus argmax (`ink_dsoftcap_argmax_gaudi2`), using SCAL PDMA scatter transfers for token input and
argmax/logits output. With `G2E_INK_DIRECT_SRAM=1`, decode scratch and (with `G2E_INK_SRAM_CONST=1`) small layer
constants and interleaved depth-attention queries reside in SRAM; `invalidate_sram()` marks the SRAM constant cache
dirty after any external Synapse recipe such as vision or DSpark.

All decode-class MoE paths read native pack v6 weights directly from HBM and accumulate bare codes per block in F32
before multiplying once per block by the block scale (`bs`) and applying the per-expert outer factor (`g`):

* Single-row decode (`T = 1`): `ink_dmoe_up9_nvf4_gaudi2` / `ink_dmoe_dn9_nvf4_gaudi2` for NVFP4 layers (0-19,
  16-element blocks) and `ink_dmoe_up9_mxf8_gaudi2` / `ink_dmoe_dn9_mxf8_gaudi2` for MXFP8 layers (20-23, 32-element
  blocks), followed by `ink_dtail_rw_gaudi2` (or fused `ink_dtail_rw_dgram_gaudi2` / `ink_dtail_rw_dgram_msg_gaudi2`).
* Batched decode (`run_decode_batch`, padded to `Bp` in `{2, 4, 8}`, slot-independent patched spans across mixed
  contexts and mixed greedy/sampled rows) and speculative verification (`build_v` / `run_verify_full`, `T = 2..4`):
  `ink_vmoe_plan_gaudi2` deduplicates routed experts across rows (splitting experts with more than 4 picks) into groups
  served by `ink_vmoe_up_nvf4_gaudi2` / `ink_vmoe_dn_nvf4_gaudi2` and `ink_vmoe_up_mxf8_gaudi2` /
  `ink_vmoe_dn_mxf8_gaudi2` (`G2E_INK_BMOE_DEDUP`, `G2E_INK_VMOE_DEDUP`). Verification defers K/V and convolution
  commits until `accept(n)` runs `commit_and_rollback` (`ink_rollback_gaudi2` and K/V commit plans).

Full-layer far attention uses non-tiled far buckets up to `G2E_INK_DFAR_TILE` (default 131072, in SRAM when
`G2E_INK_DFAR_SRAM=1` and it fits) and reusable patched tiled FlashDecode plans (`dfar_pool_` and `dfar_pipe_pool_`,
widths `{1024, 4096, 32768, 65536, 131072}`) up to 1M tokens.

## Prefill, waves, and BF16 staging

Direct prefill (`src/inkling2_dprefill.cpp`) processes 1024-token chunks (`T = 1024`). With `G2E_INK_DPREFILL_TAIL=1`
(default), short prompts or tails of 2-4 tokens run single-token Direct decode and lengths 5-1023 pad to 1024 with a
`valid_T` mask so no Synapse layer recipes are compiled.

Each application is split into two plans:

* Plan A (`build_pa`, plus `build_pa2` after tiled far attention): depth attention, message gates (passes 2-3),
  attention projections, near/far attention, output gate and projection, post-attention convolution, MLP depth
  attention/message gate, router (`p_router`, `ink_route_gaudi2`), and routing plan histogram/offsets
  (`ink_rplan_hist_gaudi2`, `ink_rplan_dyn_tab_bf16_gaudi2`).
* Plan B (`build_pb`, shared across far buckets in `plans_pb_`): dynamic BF16 MoE (`ink_rplan_dyn_slots_bf16_gaudi2`,
  `ink_sgather_bf16_gaudi2`, 4 pipelined waves of dynamic BF16 MME GEMMs `p_bdyngu` -> `ink_act_bf16_gaudi2` ->
  `p_bdyndn`, and `ink_moe_combine_bf16_gaudi2`), post-MLP convolution (`ink_conv_resm_bf16_gaudi2`), and optional
  message-source or DSpark tap copy (`ink_pcopy_gaudi2`).

Two-chunk wave execution (`run_pair`, `G2E_INK_DPREFILL_PAIR=1`): two chunk slots hold disjoint HBM scratch
(`pscratch_[2]`) and dynamic MME sets (`pdyn_[2]`). Between steps, `patch_b` downloads the 128-entry expert count
table via PDMA, computes flat-row offsets matching `ink_rplan_dyn_tab_bf16_gaudi2` (at most 12,288 flat rows for
8,192 routed slots, zero overflow), packs BF16 tile jobs (`p_bgt{16,32,64,96,128,192,256}`,
`p_bdt{16,32,64,96,128,192,256}`, F32 output) across 4 waves of 32 experts into `DirectDynMmeSet`, and uploads the
diff-encoded QMAN chunks in one PDMA. `DirectRuntime::run_waves` then runs Plan B of one chunk on one lane concurrently
with Plan A of the other chunk on the second lane using a topology-keyed merge schedule (`merge_order`, with optional
online selection via `G2E_INK_WAVE_AUTOTUNE=1`).

BF16 weight cache and staging (`init_bf16_cache`, `ensure_bf16_staged`, `G2E_INK_BF16_CACHE=auto|all|loop|none`):
`ink_nvf4_deq_bf16_gaudi2` and `ink_mxfp8_deq_bf16_gaudi2` expand native pack weights to exact BF16 (1.9 GiB and
about 1.6-1.8 ms per layer). Cached layers stay resident in HBM (looped layers 6-17 prioritized in `auto` and `loop`
modes). Uncached layers are expanded into a single 1-layer staging buffer by a reusable patched lane-0 plan
(`pbf16_deq_plan_`): unmasked (`use_mask=0`) once before `(B0[i] || A1[i])` in `run_pair` so `B1[i]` reuses the same
staged layer without re-expanding, and masked (`use_mask=1` against `cnt`) in single-chunk `queue_p` and
`G2E_INK_WAVE_SEQ=1` so short tails only expand active experts.

Prefill SRAM overlay and far attention (`G2E_INK_DPREFILL_SRAM=1`, `G2E_INK_DPREFILL_SRAM_EXTRA=1`,
`G2E_INK_FAR_FLASH=auto|1|0`): because only one Plan A runs at a time, Plan A overlays double-buffered near scores
and probabilities (`sr`, `sw`, `pw`, zeroed by `ink_zero_gaudi2`), `gated`, `a1`, and MLP message-gate buffers across
48 MiB of SRAM. For full layers with far keys (`fr > 0`), `fr <= 1024` runs inline in SRAM, `1024 < fr <= 32768` uses
capped HBM `pfar_`, and larger contexts (or `G2E_INK_FAR_FLASH=1`) run pooled in-place 48 MiB SRAM FlashAttention
(`build_pfar_blk`, `ink_pfar_head_io_gaudi2`, `ink_attn_far_step_gaudi2`, tile width `G2E_INK_PFAR_TW` default 3072,
block sizes `{1, 2, 8, 24}`, dual MME fences 2 and 3) chained via `run_chain` up to 1M tokens.

## MME descriptor templates

Templates live in `tools/mme_descs_ink/<name>/mme_debug` (`G2E_INK_MME_DIR` overrides). They are captured from
one-node Synapse graphs using `ink_direct_tool capture` (`ENABLE_EXPERIMENTAL_FLAGS=true PRINT_MME_DESCRIPTORS=1`).
`DirectPlan` patches only the A/B/C base addresses and signaling SOB; dimensions and strides are fixed per template:

* `d_*`: single-row decode projections, attention, router, message gates, LM head, and far-attention buckets.
* `v{2,3,4}_*`: speculative verification for `T = 2..4`.
* `b{2,4,8}_*`: batched decode projections (`b8_*`) and LM heads (`b{2,4,8}_lmhead`).
* `p_*`: prefill projections (512-row tiles), near/far attention (`p_fqk_*_m512`, `p_fpv_*_m512`), router, message
  gates, and dynamic BF16 MoE tiles (`p_bgt{16..256}`, `p_bdt{16..256}`).

Batched graphs with `B > 1` can capture only part of the work; use M-stacked or per-batch templates instead. Templates
carry no SynapseAI version stamp (captured under 1.24.1); after recapturing, verify every template bit-identically
against Synapse with `ink_direct_tool gemmtest <dir> <name>` or `gemmtest_prefix`.

## Rules for Direct-engine TPC kernels

* Table-free instructions only: Direct never programs the hardware special-function lookup tables. Vector float
  division compiles to `lookup_*` and scalar float division to `CALC_FP_SPECIAL`; either faults the device. Audit
  every Direct kernel object with `tools/check_direct_kernels.sh <objdir> <kernel>...`.
* Start-up anchor: every TPC prologue invalidates the table cache, and Direct dispatches touch static engine state
  from the last Synapse recipe. A process must run a Synapse TPC launch before any Direct TPC launch, and a Synapse
  recipe containing an MME node (`InkDirect::anchor()`) before any Direct MME launch.
* Compiler behaviour: inspect `tpc-llvm-objdump -d --mcpu=gaudi2 <obj>` for unexpected spills and unrolling.
  `tpc-clang` miscompiles runtime-indexed or copied register arrays and does not reliably unroll larger loops.
  `v_bf16_ld_g` broadcasts one BF16 from global memory without scalar-load stalls.
* Table-free math: Direct kernels use Pade rational approximations for `tanh` (message gates and MoE SwiGLU) and
  bit-trick/polynomial `exp` with Newton reciprocals for `sigmoid` and `softmax`.

## Tools and debug switches

Validation and benchmark binaries:

* `apps/inkling2_test.cpp`: `--pack`, `--golden`, `--max-ctx`, `--decode`, `--bench`, `--prefill-bench`,
  `--prefill-decode`, `--tbench`, `--direct-check`, `--verify-check`, `--long-check`, `--batch-check`, `--spec-test`,
  `--spec-sampled-test`, `--dspark-pack`, `--dspark-dump`, `--dspark-test`, `--dspark-step`, `--ctx-1m-test`,
  `--bf16-cache-check`, `--decode-prefill`, `--precompile`, `--precompile-vh`, `--direct-prof`, `--split`, `--dump`.
* `apps/ink_direct_tool.cpp`: `capture`, `gemmtest`, `gemmtest_prefix`, `deqtest`, `mxdeqtest`, `bf16pbtest`,
  `nvf4decodetest`, `mxf8decodetest`, `lanetest`, `tilebench`, `tpctest`, `pdmatest`, `hostrt`, `luttest`,
  `attnbench`, `mixbench`, `dnormbench`.
* `tools/verify_inkling2_pack.py` (pack vs checkpoint check) and `tools/check_direct_kernels.sh` (table-free audit).

Direct-engine, Synapse-path, and Inkling server environment switches (every `G2E_INK_*` / `G2E_DIRECT_*` switch read by
Inkling sources and `src/direct_mme.cpp`):

| Variable | Default | Selects |
|---|---|---|
| `G2E_INK_DIRECT` | `0` | Decode, batched decode, and verification Direct-engine bit mask (`1` sliding, `2` boundary, `4` messages, `8` full; `15` for all). |
| `G2E_INK_DIRECT_SRAM` | `0` | `1` places decode, batched-decode, and verification activation scratch in SRAM. |
| `G2E_INK_DPREFILL` | `0` | Prefill Direct-engine bit mask (`1` sliding, `2` boundary, `4` messages, `8` full; `15` for all). |
| `G2E_INK_BF16_CACHE` | `auto` | Static HBM BF16 expert cache mode (`auto`, `all`, `loop`, `none`). |
| `G2E_INK_MME_DIR` | `tools/mme_descs_ink` | Directory containing captured MME descriptor templates. |
| `G2E_TPC_OBJ_DIR` | build dir | Directory containing compiled `.o` TPC kernel objects. |
| `G2E_DIRECT_SPIN` | `1` (Inkling) | `1` polls the device completion counter; `0` blocks in the driver. |
| `G2E_DIRECT_MME_DIFF` | `1` (Inkling) | `1` diff-encodes MME descriptor QMAN streams in prepared plans; `0` emits full descriptors. |
| `G2E_DIRECT_PDMA_PATCH` | `1` (Inkling) | `1` uploads dynamic plan patches over scheduler PDMA; `0` uses host-mapped writes. |
| `G2E_DIRECT_NO_CHAIN` | unset | When set, executes `run_chain` plans one at a time instead of hardware-chaining them. |
| `G2E_DIRECT_POOL` | unset | When set, logs per-plan ECB/code sizes and cumulative SCAL pool usage. |
| `G2E_DIRECT_DYN_VALIDATE` | unset | When set, verifies diff-encoded dynamic MME packets against full packets. |
| `G2E_DIRECT_VERIFY_ECB` | unset | When set, reads back uploaded plan ECBs from HBM to verify host-to-device transfer. |
| `G2E_INK_DIRECT_FULL` | `1` | `1` chains whole-token B=1 decode (`embed` + 48 apps + `head`) without Synapse; `0` per-app decode only. |
| `G2E_INK_DSPAN` | `8` | Max B=1 decode applications fused into one multi-application span plan (`<= 1` disables spanning). |
| `G2E_INK_SRAM_CONST` | `1` | `1` caches small decode layer constants and interleaved depth-attention queries in SRAM. |
| `G2E_INK_SRAM_ACT` | `1` | `1` keeps B=1 decode `blocks`/`h`/`prev` activations in SRAM during full-token decode. |
| `G2E_INK_DFAR_SRAM` | `1` | `1` places B=1 decode far-attention score/probability buffers in SRAM when they fit. |
| `G2E_INK_DFAR_TILE` | `131072` | Far-row threshold above which B=1 decode switches from single-bucket far attention to tiled FlashDecode. |
| `G2E_INK_DFAR_PIPE` | `1` | `1` pipelines multi-tile decode FlashDecode (`build_dfar_pipe`); `0` runs tiles sequentially. |
| `G2E_INK_DFAR_TPC_PV` | `1` | `1` uses TPC `ink_dfar_pv` for decode far `P*V`; `0` uses MME `d_fpv_*` templates. |
| `G2E_INK_DMG_FUSE` | `1` | `1` fuses B=1/verify message-gate up-projection into `ink_dmg_fgy`; `0` uses MME `*_mgf*` + `ink_dmgy`. |
| `G2E_INK_DROUTE1` | `1` | `1` uses `ink_droute1` for B=1/verify top-8 routing; `0` falls back to `ink_route`. |
| `G2E_INK_FUSE_TAIL_GRAM` | `1` | `1` fuses each B=1 MoE tail with the next application's depth-attention Gram (`ink_dtail_rw_dgram*`). |
| `G2E_INK_BMOE_DEDUP` | `1` | `1` deduplicates routed experts across rows in batched decode (`ink_vmoe_plan` + `ink_vmoe_*`); `0` runs per-row `dmoe_*9`. |
| `G2E_INK_VMOE_DEDUP` | `1` | `1` deduplicates routed experts across rows in verification (`ink_vmoe_plan` + `ink_vmoe_*`); `0` runs per-row `dmoe_*9`. |
| `G2E_INK_B_DMG_FUSE` | `1` | `1` fuses batched-decode message-gate up-projection into `ink_dmg_fgy`; `0` uses MME `*_mgf*` + `ink_dmgy`. |
| `G2E_INK_B_SPAN` | `1` | `1` packs batched decode into multi-application span plans; `0` emits standalone plans per stage. |
| `G2E_INK_B_OPT_MME` | `1` | `1` masks padded-row near-attention MME stages in batched decode via optional-stage bit masks. |
| `G2E_INK_B_KV4` | `1` | `1` commits batched full-layer KV rows with `ink_kv_commit_b4`; `0` uses per-row `ink_kv_commit_d1`. |
| `G2E_INK_DIRECT_VFULL` | `1` | `1` runs whole-block verification (`embed` + 48 apps + `head`) on Direct; `0` uses Synapse embed/head. |
| `G2E_INK_DIRECT_VCOMMIT` | `1` | `1` runs verification `accept(n)` KV commit and conv rollback on Direct; `0` uses Synapse `commit_recipe`. |
| `G2E_INK_DPREFILL_SRAM` | `1` | `1` enables Plan-A 48 MiB SRAM overlay for near-attention scores/probabilities and message-gate scratch. |
| `G2E_INK_DPREFILL_SRAM_EXTRA` | `1` | `1` additionally aliases `gated`, `a1` (and `q` when `pw` is in HBM) into Plan-A SRAM. |
| `G2E_INK_FAR_FLASH` | `auto` | Prefill far-attention policy: `auto` (inline <=1K, HBM <=32K, SRAM Flash >32K), `1` (pure SRAM Flash), `0` (legacy HBM). |
| `G2E_INK_PFAR_TW` | `3072` | Overrides SRAM FlashAttention tile width (`1024`, `2048`, or `3072`). |
| `G2E_INK_DPREFILL_PAIR` | `1` | `1` interleaves two 1024-token prefill chunks in `run_pair`; `0` runs chunks singly. |
| `G2E_INK_DPREFILL_TAIL` | `1` | `1` handles 2-4 token tails with B=1 Direct decode and 5-1023 tails with padded `T=1024` Direct prefill. |
| `G2E_INK_WAVE_AUTOTUNE` | `0` | `1` enables online selection among `merge_order_candidates` schedules in `run_pair`. |
| `G2E_INK_PADE` | bit `2` on | Bit `2` (`& 2`) selects Pade `tanh` (`ink_mg_out_pade`) vs `ink_tanh` (`ink_mg_out`) in Direct prefill message gates. |
| `G2E_INK_NO_WIN` | unset | When set, disables windowed softmax (`ink_attn_sm16w`) in Direct prefill and uses `ink_attn_sm16b`. |
| `G2E_INK_DP_WAITS` | unset | When set, adds explicit `s_proj` waits to `prprep` and `kvconv` stages in `build_pa` (fence-ordering debug). |
| `G2E_INK_DPREFILL_APP` | unset | Restricts Direct prefill to a single application index (`<app>`) to bisect Direct vs Synapse. |
| `G2E_INK_DPREFILL_LIST` | unset | When set, prints every application's kind (`F`/`S`, `b`, `m`) during prefill initialization. |
| `G2E_INK_DIRECT_CHECK` | unset | Sets call index `<N>` at which single-app Direct decode compares intermediate scratch against Synapse. |
| `G2E_INK_DIRECT_PROF` | unset | When set, disables decode chaining and prints per-application mean Direct decode latency on exit. |
| `G2E_INK_DIRECT_NOCHAIN` | unset | When set, runs decode and verification plans individually instead of chaining them. |
| `G2E_INK_DIRECT_TIME` | unset | When set, logs chain/pair/single-chunk Direct execution times. |
| `G2E_INK_DPREFILL_PROF` | unset | Sets `<app>` to profile Plan A/B stage-by-stage cumulative times on the second prefill chunk. |
| `G2E_INK_DPREFILL_PROF_ALONE` | unset | Under `G2E_INK_DPREFILL_PROF`, times every Plan A/B stage call in isolation. |
| `G2E_INK_PAIR_PROF` | unset | When set, logs `run_pair` / `run_forward_pair` device, patch, dequant, and far-attention breakdown. |
| `G2E_INK_WAVE_PROF` | unset | Sets half-step index `<i>` to time `B0`, `A1`, and `(B0 \|\| A1)` after `run_pair` (leaves model state invalid; benchmark only). |
| `G2E_INK_WAVE_TIMELINE` | unset | Under `G2E_INK_WAVE_PROF`, times progressive stage cuts of the merged wave dispatch order. |
| `G2E_INK_WAVE_DEBUG` | `0` | `1` logs `merge_order` simulated makespan; `>= 2` dumps the full simulated wave timeline. |
| `G2E_INK_WAVE_SEQ` | unset | When set, runs the two chunks of `run_pair` sequentially instead of overlapping Plan B and Plan A. |
| `G2E_INK_WAIT_CHECK` | unset | When set, prints verbose stage-pair details when `check_waits` finds unordered fence waits. |
| `G2E_INK_PLAN_DUMP` | unset | Sets `<app>` to dump Plan A and Plan B stages, SOB waits, and signals on plan build. |
| `G2E_INK_HASH` | unset | `1` hashes final chunk state, blocks, and rings; `2` hashes every application output and runs `verify_rplan`. |
| `G2E_INK_HASH_APP` | unset | Under `G2E_INK_HASH=2` at `pos=0`, hashes every intermediate scratch tensor of `<app>`. |
| `G2E_INK_HASH_FIELDS` | unset | Under `G2E_INK_HASH=2` at `pos=0`, hashes every `PScratch` region of slot 0 per application. |
| `G2E_INK_HASH_KVS` | unset | Under `G2E_INK_HASH=2` at `pos=0`, hashes K/V convolution outputs and verifies `vcat` new-V copy. |
| `G2E_INK_LOADS` | unset | Under `G2E_INK_HASH=2` (`verify_rplan`), prints per-application routed expert load histograms. |
| `G2E_INK_DPCHECK` | unset | Sets `<app>` to run both Synapse and Direct prefill on the same inputs and report output diffs. |
| `G2E_INK_DBGOUT` | unset | When set, Synapse recipes emit intermediate debug tensors (`dbg_*`) for comparison under `G2E_INK_DPCHECK`. |
| `G2E_INK_DPDUMP` | unset | Directory path to dump raw Direct and Synapse intermediate tensors under `G2E_INK_DPCHECK` + `G2E_INK_DBGOUT`. |
| `G2E_INK_PROFILE` | unset | When set, records and prints per-recipe Synapse device times. |
| `G2E_INK_REF` | unset | When set, disables all fused Synapse TPC kernels and runs pure graph ops (reference path). |
| `G2E_INK_KMASK` | `255` | Bit mask of fused Synapse TPC kernels (`1` qprep, `2` kv conv, `4` attn/mlp conv, `8`/`16` sliding/full kv conv, `32` depth/attn, `64` router, `128` message gates). |
| `G2E_INK_CHUNK` | `1024` | Maximum token chunk size for `Inkling2::forward`. |
| `G2E_INK_SCORE_ROWS` | `8388608` | Synapse full-layer `Tq * (Sf + W + T)` score-tensor budget for query blocking. |
| `G2E_INK_QBLOCKS` | `16` | Maximum Synapse query blocks per full-layer prefill chunk. |
| `G2E_INK_OLD_FAR` | unset | When set, disables `ink_attn_sm16f` on Synapse full layers and uses query-blocked F32 far attention. |
| `G2E_INK_DBGPOST` | unset | When set, runs Synapse K/V `fused_conv` in mode 0 and computes K-clip / V-RMS as graph ops. |
| `G2E_INK_MG_NOLUT` | unset | When set, selects LUT-free `ink_mg_out` instead of `ink_mg_out_lut` in Synapse message gates. |
| `G2E_INK_PAD_TAIL` | `1` | `1` pads non-power-of-two Synapse prefill tails in verify mode; `0` splits them into power-of-two pieces. |
| `G2E_INK_DMA_COMMIT` | unset | When set, Synapse KV commit uses per-head `stream_.copy` DMAs instead of `commit_recipe`. |
| `G2E_INK_RESET_ZERO` | unset | When set, `reset()` re-zeros KV rings and linear caches on every call instead of once at startup. |
| `G2E_INK_DSPARK_MAX_CTX` | `32768` | Maximum context length for DSpark speculative decoding. |
| `G2E_INK_SPEC` | `1` | `0` disables speculative decoding in `g2e_server` at runtime even when `--draft-pack` is loaded. |
| `G2E_INK_NO_SPEC` | unset | `1` skips loading the DSpark drafter in `g2e_server`. |
| `G2E_INK_SNAPSHOT_GB` | `32` | Pinned host-RAM budget (GiB) for Inkling prefix-cache state snapshots in `g2e_server` (`0` disables). |

`G2E_INK_WAVE_PROF` reruns half-steps for measurement and leaves model state invalid afterwards; use it only with
`inkling2_test --prefill-bench`, never in `g2e_server`.
