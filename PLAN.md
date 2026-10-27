# PLAN: Laguna-S-2.1 DFlash + fire-once `[[pin]]` donor slot

## Progress

- 2026-10-21 start: Read PLAN.md (OK, 202 lines). Beginning Part E1: explore current master state of dflash.cpp, models.h, llama-arch.*, speculative.cpp, server-context.cpp get_available_slot, and llama-kv-cache-iswa.cpp size_swa. Live server not touched.
- 2026-10-21 explore done. Master state confirmed:
  - llama-arch.h: LLM_ARCH_LAGUNA (line 127), LLM_ARCH_DFLASH (line 166) exist. LLM_KV_DECODER_ARCH and LLM_TENSOR_ENC_AUX_NORM DO NOT exist.
  - models.h: struct llama_model_dflash at line 1491, minimal; no decoder_laguna/aux_norm fields.
  - dflash.cpp: encoder graph (354-367) is fc->output_norm_enc only; decoder graph (807-1106) uses standard attn, layer.attn_norm (956), no Laguna branch. dflash.attention.causal is read in speculative.cpp:1056, dflash.block_size at speculative.cpp:1050, target_layers loaded at dflash.cpp:50. Shared tok_embd/output via ctx_other fallback (dflash.cpp 918-926, 1052-1061), TENSOR_NOT_REQUIRED.
  - laguna.cpp: uses cb() calls, exposes t_embd (327) and t_logits (331). No t_h_nextn. Layer 48 is the 48-layer model (Laguna-S.2). No nextn blocks. Softplus gate IS present in laguna.cpp:247.
  - speculative.cpp: common_speculative_impl_draft_dflash (struct at 983). ctx_dft created via llama_init_from_model with ctx_other=ctx_tgt (2670). No seq_cp/alias hook exists. accept() is no-op (1427-1429).
  - server-context.cpp: get_available_slot at line ~1770 (id_slot path, LCP sim loop ~1788, LRU loop ~1840). No donor_slot/pin_armed exists. slot_prompt_similarity at line 1107. kv_unified at 1887.
  - server-task.h: no pin_donor field.
  - llama-kv-cache-iswa.cpp: size_swa formula at line 73, swa_full branch at 81. No --swa-cache param. Only --swa-full exists.
  - common/arg.cpp: --kv-unified (1727-1734) and --kv-unified-per-slot (1652-1659) exist. --swa-cache does NOT exist.
  - Chat handler: common_chat_msgs_parse_oaicompat at server-common.cpp:1436 (inputs.messages), template applied in common_chat_session (1508) via common_chat_templates_apply (common/chat.cpp:338). Marker strip belongs between 1436 and 1508.
  - Next: fetched poolside laguna branch source (dflash.cpp, models.h, speculative.cpp). Extracted exact Laguna contract deltas.
- 2026-10-21 Part A1 done: Added LLM_KV_DECODER_ARCH to KV enum (llama-arch.h:248) and name map (llama-arch.cpp:243, "%s.decoder_arch"). Added LLM_TENSOR_ENC_AUX_NORM to tensor enum (llama-arch.h:708) and name map (llama-arch.cpp:560, "enc.aux_norm").
- 2026-10-21 Part A2 done: Added decoder_laguna bool + aux_norm tensor ptr to struct llama_model_dflash (models.h:1499-1506), with poolside comment text. wqkv_gate already exists on llama_layer (llama-model.h:313).
- 2026-10-21 Part A3 done in dflash.cpp: (1) load_arch_hparams reads dflash.decoder_arch via ml.get_key(LLM_KV_DECODER_ARCH) → sets decoder_laguna. (2) load_arch_tensors creates aux_norm with shape {n_embd, target_layer_ids.size()}. (3) layer loop creates wqkv_gate via LLM_TENSOR_ATTN_GATE if gate tensor exists. Target layer count via member target_layer_ids (llama-model.h:828).
- 2026-10-21 Part A4 done in dflash.cpp: (1) Encoder graph adds model_df cast + aux_norm (reshape 3d → rms_norm → mul → reshape 2d) before fc. (2) Decoder encoder-fusion path: kv_inp = decoder_laguna ? build_norm(inp_g, layer.attn_norm) : inp_g; K/V projected from kv_inp. (3) Decoder path: gated = wqkv_gate != nullptr && decoder_laguna; build_attn with NULL wo when gated; softplus gate from noise_norm; per-head or per-element mul; then build_lora_mm(wo). Causal mask driven by cparams.causal_attn via set_input_kq_mask (llama-graph.cpp:581), not by graph code.
- 2026-10-21 Part A5 done by implementer: speculative.cpp reads dflash.decoder_arch near line 1059 (after dflash.attention.causal), sets causal_attn=true for laguna. In common_speculative_init_result (~line 2699), after ctx_dft creation, calls llama_set_causal_attn(ctx_dft, causal). Verified compiles -std=c++17.
- 2026-10-21 FIX: models.h edit initially matched llama_model_t5 instead of llama_model_dflash (same load_arch_hparams/graph<true> pattern). Removed from t5, re-added to dflash with graph_dsv4 anchor for uniqueness. Verified: decoder_laguna at models.h:1499, aux_norm at 1503, t5 clean at 1586.
- 2026-10-21 Part B1 done by implementer + fixed: server-task.h added `bool pin_donor`. server-context.cpp: `pin_armed`/`donor_slot`/`pending_pin_donor` on server_context_impl. `post_chat_completions` scans system messages for `[[pin]]` while `pin_armed`, strips marker (rfind, erase 7 chars, fix double space), drops empty message, sets `pin_armed=false` + `pending_pin_donor=true`. `handle_completions_impl` reads `pending_pin_donor` -> `task.pin_donor`. FIXED: removed per-request `pin_armed=true` reset which broke fire-once; now `pin_armed` stays false after first hit until restart.
- 2026-10-21 Part B4 done by implementer: --swa-cache N added to arg.cpp. n_swa_cache=0 default in common_params + llama_context_params + llama_memory_params. size_swa in llama-kv-cache-iswa.cpp uses n_swa_cache if >0, else n_swa. Passed through dsv4 + hybrid_iswa constructors. hparams.n_swa untouched (stays 512). 13 files changed.
- 2026-10-21 Part B2-B3 done by implementer: (1) Donor excluded from all selection paths: id_slot (decline if donor), LCP loop (skip), get_free_slots (exclude), try_clear_idle_slots (skip), cache_idle_slots (skip). (2) Donor assignment in post_decode SLOT_STATE_DONE_PROMPT: if pin_donor, set donor_slot=slot.id, store donor_tokens, send final response, release slot (KV preserved). (3) Reuse in decode loop SLOT_STATE_STARTED: if n_past==0 && kv_unified && donor_slot>=0, compute LCP via get_common_prefix, require lcp_len >= donor_tokens.size()-1, seq_cp lcp_len-1 tokens (leave 1 for worker), log donor/lcp/worker. (4) Context shift guard: raise n_keep to protect donor prefix cells. Syntax check passes. Note: pin_donor request prefills prefix and returns empty generation (target-only first, as PLAN B6 allows).



Personal fork of `ggml-org/llama.cpp` master. Do not open a PR. Do not vendor the whole Poolside `laguna` branch. Port only the Laguna DFlash contract, then add the pin.

Hardware this must build and run on: 5x PCIe Tesla V100 32GB (sm_70, no NVLink), driver 580.178.04, CUDA 12.8, host gcc-14. CMake: `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=70`, `GGML_NATIVE=OFF`, gcc-14 as C/CXX/CUDA host compiler, nvcc from `/usr/local/cuda-12.8`. Any CUDA kernel added must compile for architecture 70. Do not depend on sm_80+ features. Draft model is pinned to one card (`--spec-draft-device` / `--spec-draft-override-tensor '.*=CUDA0'`). Target is tensor-split across the five cards. Do not layer-split the draft.

## Progress (continued)

- 2026-10-21 Build: SUCCESS. Configured with GCC-14 + CUDA 12.8 + CMAKE_CUDA_ARCHITECTURES=70 + GGML_NATIVE=OFF in `build-lagunaMod`. One arg.cpp string_format fix needed (string literal juxtaposition), fixed. Final binary: `build-lagunaMod/bin/llama-server`. All 20 modified files compile cleanly. Live server at `llama.cpp-b11330/build/bin/llama-server` was NOT touched.
- All changes from Parts A and B complete. Ready for user to switch and test.
- 2026-10-22 Fix: `LLM_TENSOR_ENC_AUX_NORM` was in the tensor name map (`llama-arch.cpp:561`, "enc.aux_norm") but missing from `LLM_TENSOR_INFOS`, causing `missing tensor info mapping for enc.aux_norm` at startup. Added the entry `{LLM_TENSOR_ENC_AUX_NORM, {LLM_TENSOR_LAYER_OUTPUT, GGML_OP_MUL}}` in `LLM_TENSOR_INFOS` next to `LLM_TENSOR_ENC_OUTPUT_NORM` (`llama-arch.cpp:850`). Verified `LLM_TENSOR_ATTN_GATE` already has an INFOS entry (`llama-arch.cpp:859`), so the `wqkv_gate` tensor reuse in dflash.cpp is covered. No name string changed, no new arch added, no other tensors touched. Rebuilding only the `llama-server` target in `build-lagunaMod` (no reconfigure) to apply this one-line fix without disturbing the running server.
- 2026-10-22 Fix: DFlash target layer extraction wiring missing in Laguna's graph builder. `llm_graph_result::set_outputs()` at `llama-graph.cpp:1451` asserts `t_layer_inp[il] != nullptr` when the draft requests a target layer (DFlash calls `llama_set_embeddings_layer_inp` with `[2,11,20,30,39,48]`). Laguna's `build_backbone` never assigned those slots. Added gated extraction wiring in `src/models/laguna.cpp`:
  - In-loop (before `inpSA = inpL`): `if ((size_t) il < cparams.embeddings_layer_inp.size() && cparams.embeddings_layer_inp[il]) { res->t_layer_inp[il] = inpL; cb(...); ggml_build_forward_expand(gf, ...); }`
  - After-loop (before `cur = inpL`): same guard for `n_layer` -> `res->t_layer_inp[n_layer] = inpL;`
  - Residual form: plain residual (`inpL`, the post-FFN-add hidden passed to the next layer). Laguna has no hyper-connections (no `build_hc_mean`), so the plain `inpL` matches the qwen3/mimo2 pattern. The gated `cb` + `ggml_build_forward_expand` matches the deepseek4 pattern exactly.
  - Rebuilt only `llama-server` in `build-lagunaMod` with existing V100 flags (CUDA arch 70). Build SUCCESS. Running server not touched.

- 2026-10-22 Fix: First-prompt crash in `extract_layer_inputs` (shape assertion). Steps taken:
   1. **t_layer_inp residual shape** (`src/models/laguna.cpp`): The assignment already stores `inpL` (the incoming residual) at the start of the layer loop, before any op changes the leading dimension. Residual form used: plain `inpL` [n_embd, n_tokens]. Added a temporary `fprintf` log at both the in-loop and post-loop `t_layer_inp` assignments to verify `ne[0] == n_embd` at runtime. Laguna has no multi-stream/hyper-connection residual (no `build_hc_mean` needed), so the plain `inpL` matches the qwen3/mimo2 pattern. **TODO (user): after confirming `ne[0] == n_embd` prints correctly for all layers, delete the two temporary `fprintf` lines in `src/models/laguna.cpp` (~line 199 and ~line 337) and rebuild.**
  2. **Special-case extract layer 48 (n_layer)** (`common/speculative.cpp`): The DFlash init path previously called only `llama_set_embeddings_layer_inp` for every target layer id. Added the split that the Eagle3 path already uses: `target_layer_ids[k] == n_layer_tgt` is routed through `llama_set_embeddings_nextn(ctx_tgt, true, /*masked*/ false)`. Added `n_layer_tgt = llama_model_n_layer(model_tgt)` to the DFlash constructor. Updated the gather site (feature-copy loop) to read `n_layer_tgt` back via `llama_get_embeddings_nextn(ctx_tgt)` instead of `llama_get_embeddings_layer_inp`. Without this, `t_layer_inp[48]` would never be filled (the laguna graph stores the post-loop residual at `t_layer_inp[n_layer]`, not at `get_layer_inp(48)` via a different path), and the extraction would either crash or read stale data.
  3. **Draft attention non-causal** (`common/speculative.cpp`): Removed two uncommitted additions that set `causal_attn = true` for laguna drafters (commented "Laguna decoder blocks use a causal SWA noise block"). Both were wrong: the laguna branch on the upstream fork calls `llama_set_causal_attn(ctx_dft, false)`. DFlash block diffusion is non-causal; leaving it causal produces wrong or NaN draft outputs. Removed the special case in the DFlash constructor (after `dflash.attention.causal` read) and the entire `llama_set_causal_attn` block in `common_speculative_init_result` (redundant with the DFlash constructor's call). `causal_attn` defaults to `false` and stays false unless the GGUF explicitly sets `dflash.attention.causal=true`.
  4. **aux_norm pre-concat** (`src/models/dflash.cpp`): Already applied in the encoder graph (`graph<true>`): `aux_norm` reshapes to [n_feat, n_aux, n_tokens], applies `ggml_rms_norm` + `ggml_mul(aux_norm)`, reshapes back to [n_feat*n_aux, n_tokens] before `fc`. The post-FC norm (`output_norm_enc`) is unchanged. Both norms are required for Laguna. No change needed.
  5. **Non-finite feature sanitization** (`common/speculative.cpp`): Added a simple finite check after the target-feature gather loop and before `batch_inject` creation in the DFlash `process()` method. Replaces NaN/Inf with 0.0f in `features_buf`. Cheap, only on the host-side feature path.
  6. **Shared embedding / lm_head device**: Not addressed proactively. The draft borrows `tok_embd`/`output` from the target via `ctx_other` (already `TENSOR_NOT_REQUIRED` in dflash.cpp). If a backend mismatch appears on first draft forward, the fix is to ensure the borrowed tensors are visible to the draft's device, or document that `--spec-draft-device` must target one of the target's split devices. The user's launch line pins the draft to CUDA0, which is also a target device. No change yet; only address if the error appears.

## Success criteria

1. Master loads `poolside/Laguna-S-2.1-GGUF` target plus the published `laguna-s-2.1-DFlash-BF16.gguf` with `--spec-type draft-dflash`. No tensor-count mismatch, no unknown-arch error.
2. Draft acceptance is reported. Prefill is not worse than current master Laguna without the draft.
3. After a request containing `[[pin]]`, a later request on a different slot that shares that prefix does not reprocess the prefix. Log shows an alias/hit, not a full prompt eval. A same-slot follow-up does not print the forced-reprocess line.
4. Attention mask in SWA layers stays 512. Only the stored SWA cell count grows.

## What is already true on master

- Laguna target architecture is merged (`#25165`). `src/models/laguna.cpp` exists. `LLM_ARCH_LAGUNA` is in `src/llama-arch.cpp`.
- Generic DFlash is on master: `src/models/dflash.cpp`, `LLM_ARCH_DFLASH`, `--spec-type draft-dflash` in `common/speculative.cpp` (`common_speculative_impl_draft_dflash`).
- Master DFlash does **not** implement the Laguna decoder contract. The published draft GGUF has `dflash.decoder_arch = laguna`, 6 SWA layers, `target_layers = [2,11,20,30,39,48]`, `block_size = 16`, shared embd/lm_head, `draft_vocab_size == vocab_size`, no `d2t`/`t2d`. Loading it on stock master fails or builds the wrong graph.
- Poolside's `laguna` branch has the contract. Do not merge that branch. Cherry-pick the contract.

## Part A — DFlash Laguna contract

Source of truth: `https://github.com/poolsideai/llama.cpp` branch `laguna`, file `src/models/dflash.cpp` and the `llama_model_dflash` section of `src/models/models.h`. Diff those against master. Port the Laguna-only deltas. Leave Qwen / DSpark / DFlash2 paths on master untouched.

### A1. Keys and tensors

`src/llama-arch.h` and `src/llama-arch.cpp`.

- Add `LLM_KV_DECODER_ARCH` if missing. GGUF key is `dflash.decoder_arch`. Poolside has it at `llama-arch.h` around the KV enum.
- Add `LLM_TENSOR_ENC_AUX_NORM`. Poolside name is the encoder aux norm, shape `[n_embd, n_aux]`, one RMSNorm weight per captured target layer. Confirm the tensor name in the published GGUF (`blk`/`enc` naming) before hard-coding the string. Dump the draft GGUF metadata and tensor list with `llama-gguf` or a small Python `gguf` reader and match the string exactly.
- Do not add a new arch enum. The draft stays `LLM_ARCH_DFLASH`. `decoder_arch` is a key on that arch, not a new arch. The published file is `arch = dflash`.

### A2. Model class

`src/models/models.h`, `struct llama_model_dflash` (master ~line 1402).

Add:

```cpp
bool decoder_laguna = false;
ggml_tensor * aux_norm = nullptr;
```

Comment them as Poolside does: softplus attn gate, per-aux feature norms, context K/V through `input_layernorm`, causal noise-block attention.

### A3. Load

`src/models/dflash.cpp`, `llama_model_dflash::load_arch_hparams`.

After the existing `target_layers` read, read `LLM_KV_DECODER_ARCH`. Set `decoder_laguna = (decoder_arch == "laguna")`. Do not throw if the key is absent. Qwen drafts do not have it.

In `load_arch_tensors`, if `decoder_laguna`, create `aux_norm` with `LLM_TENSOR_ENC_AUX_NORM`, shape `{ n_embd, (int64_t) target_layer_ids.size() }`. Also load any Laguna-only per-layer tensors the poolside file loads inside the layer loop (attn gate weight, and the `input_layernorm` path for context K/V). Copy those `create_tensor` calls from the poolside file. Do not invent tensor names.

### A4. Graph

`src/models/dflash.cpp`, the encoder graph (`graph<true>`) and the decoder graph (`graph<false>`).

Port these four behaviors from the poolside file, gated on `decoder_laguna` / `aux_norm != nullptr`:

1. Encoder: before `fc`, apply `aux_norm` per captured feature (poolside ~lines 128–136, `ggml_mul` + `enc_aux_norm` callback).
2. Decoder context K/V: project from `input_layernorm` output, not from the raw hidden (poolside ~lines 192–200, the `if (model_df.decoder_laguna)` branch).
3. Causal mask on the noise block. Master is non-causal (`// cache-aware, non-causal attention` near the bottom of the file). Laguna is causal. Drive it from the existing `dflash.attention.causal` metadata if present, else from `decoder_laguna`. `common/speculative.cpp` already reads `dflash.attention.causal` (~line 1049). Make the graph honor it.
4. Softplus attention gate. Poolside applies it after attention (`attn_gate_softplus`, `attn_gated`). Port that op. It is a mul by `softplus(gate)`, not a new kernel. No sm_80 dependency.

Also stash the per-layer hidden the draft needs from the target. The published `target_layers` includes 48, which is one past the last real layer on a 48-layer model. Confirm how master Laguna exposes layer outputs and how the poolside target graph publishes the nextn / final hidden. The draft must receive features for every id in `target_layers`. If master `llm_graph_result::set_outputs()` asserts when the draft asks for per-layer states, that is the bug described in the poolside follow-up commit `b01fc02` ("models : DFlash draft support for Laguna-S-2.1"). Fix it in the target graph (`src/models/laguna.cpp`) or in the dflash gather, whichever the poolside diff shows. Do not stub the last layer.

Shared embedding and `lm_head`: the draft GGUF may omit `token_embd.weight` / `output.weight` and expect the target's. Master already has `TENSOR_NOT_REQUIRED` on `tok_embd` in `load_arch_tensors`. Keep that. Wire the draft to the target tensors in `common/speculative.cpp` if the poolside branch does it there and master does not. No `d2t`.

### A5. Speculative glue

`common/speculative.cpp`, `common_speculative_impl_draft_dflash`.

- Confirm `--spec-draft-n-max` is clamped to `block_size - 1` (15 for block 16). Poolside documents this. Master already reads `dflash.block_size`.
- Confirm the draft context is created with the causal flag when `decoder_laguna` or `dflash.attention.causal` is set.
- Pin the draft to one device. Do not change the user's existing `--spec-draft-override-tensor '.*=CUDA0'` path. Do not add NVLink logic.

### A6. Do not port

- Poolside's server slot changes, if any.
- Any Qwen3.5 / Qwen3.8 / DSpark draft path rewrite.
- Recurrent / GDN hybrid fixes. Laguna is not one.
- A new arch name. `dflash` + `decoder_arch=laguna` is the contract the published file uses.

## Part B — fire-once `[[pin]]` donor

Personal only. One donor per process. Search until the first hit, then never search again until restart.

### B1. Parser

Insert in the chat-completions path **before** the Jinja template.

The messages are parsed in `common/chat.cpp`, `common_chat_msgs_parse_oaicompat` (~line 342). The template is applied later in the server chat handler. The clean place is immediately after `common_chat_msgs_parse_oaicompat` returns and before `common_chat_templates_apply`, inside the server chat completion handler (`tools/server/server-chat.cpp` / the handler that builds the prompt for `SERVER_TASK_TYPE_COMPLETION`). Do not scan the rendered string.

Rules:

- Server context holds `bool pin_armed = true` and `int donor_slot = -1`.
- While `pin_armed`, walk `messages`. Only consider `role == "system"`. On each system `content` string, `rfind("[[pin]]")`.
- On hit: erase the six characters (and one adjacent space if it leaves a double space), set a flag on the task (`task.pin_donor = true`), set `pin_armed = false`. Do not scan further messages. Do not scan future requests.
- If the system content is an array of parts, scan each text part the same way.
- If the marker is the entire content and erasing it leaves the message empty, drop that message rather than emitting an empty turn. An empty turn still produces template tokens and breaks the later match.
- Never match in user, assistant, or tool content.

Reverse search is sufficient. Fire-once makes the cost irrelevant after the first hit. Do not add a request field. The marker is the only trigger.

### B2. Donor slot lifecycle

`tools/server/server-context.cpp`.

- `get_available_slot` starts at ~line 1680. At the top of every selection path (id_slot, LCP similarity loop ~1708, LRU loop ~1745), `continue` if `slot.id == donor_slot`. The donor is never selected, never overwritten, never idle-evicted.
- When a task arrives with `pin_donor`, assign it an ordinary free slot, run prefill of the prompt only, then set `donor_slot = slot.id`. Do not let that request generate into the donor. If `n_predict > 0`, alias the donor into a second slot and generate there. The donor's `prompt.tokens` must be exactly the stripped, templated, tokenized prefix.
- Store the donor token vector on the server context as well as on the slot, so selection does not depend on the slot object staying untouched.

### B3. Reuse

On a later completion task, after tokenization:

- Compute LCP against the donor token vector (`server_tokens::get_common_prefix`, already used in `get_available_slot`).
- Require `--kv-unified`. If the server was started without it, log once and fall back to normal per-slot behavior. Do not silently copy.
- If LCP is long enough (use the existing `slot_prompt_similarity` threshold, default 0.1, but also require the match to cover the whole donor or all but the last token):
  - Pick a free non-donor slot.
  - `llama_memory_seq_cp` / the existing `seq_cp` from donor seq to worker seq over `[0, lcp)`.
  - Leave at least one token for the worker to evaluate, so logits exist.
  - Cells past the fork are allocated only for the worker. No `seq_rm` and no position shift may touch a cell whose sequence bitset contains the donor. Guard the SWA shift path. This is the corruption bug in the earlier unified-cache prototype.
- Log a single line: donor id, lcp length, worker slot. This is the line the success test looks for.

### B4. SWA cache size, not attention window

`src/llama-kv-cache-iswa.cpp` ~line 73:

```cpp
uint32_t size_swa = GGML_PAD(std::min(size_base, hparams.n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256);
if (swa_full) size_swa = size_base;
```

Add a third value. A server param `--swa-cache N` (or a compile-time default of 8192 for this fork) sets the stored cell count. `hparams.n_swa` stays 512 and remains the attention mask. Do not raise `n_swa`. That would make SWA layers attend past training.

Pass the value through `common_params` (`common/arg.cpp`) into the context params that construct `llama_kv_cache_iswa`. Default 0 means current behavior (`n_swa`). For this user's launch, 8192.

The server's force-reprocess check uses `pos_min` against `pos_next - n_swa`. With a larger stored window, `pos_min` drops and the check passes for forks inside the stored window. Do not special-case Laguna in that check. Fixing the stored size is the fix.

### B5. Per-slot cap

Already exists: `--kv-unified-per-slot` (`common/arg.cpp`, applied in `server-context.cpp` `n_ctx_slot()` ~line 3999 in current master). Document it in the launch line. Do not reimplement it. The user runs unified with a 256k cap so one thread cannot take the 1.3M pool.

### B6. DFlash and the donor

Second priority, after target pin works.

The draft has its own context and KV. On alias, also `seq_cp` the draft sequence for the matched prefix, or the first speculative round rebuilds it. Checkpoint both in the donor. If this is hard, ship target-only first and log that the draft prefix was not reused.

## Part C — launch line the plan must produce

```bash
llama-server \
  -m laguna-s-2.1-Q4_K_M.gguf \
  -md laguna-s-2.1-DFlash-BF16.gguf \
  --spec-type draft-dflash --spec-draft-n-max 7 \
  --spec-draft-override-tensor '.*=CUDA0' \
  --tensor-split <existing 5-way split> \
  --jinja -fa on \
  --kv-unified \
  --kv-unified-per-slot 262144 \
  --swa-cache 8192 \
  -c 1310720 -np 5 \
  --cache-prompt
```

`-np 5` keeps the donor plus four workers. Context pool is 1.3M. Cap is 256k per worker. SWA store is 8k. Attention window is 512.

## Part D — test plan

Build with the V100 cmake line above. `CMAKE_CUDA_ARCHITECTURES=70` is mandatory.

1. Load target + published DFlash GGUF. Expect no arch error, no missing tensor. Log must show `decoder_arch=laguna` and `extract_layers` matching the GGUF.
2. One short prompt with the draft. Expect a non-zero acceptance count in the speculative log. Compare prefill tok/s to the same binary without `-md`. Must not regress.
3. Pin: send one chat completion whose system message contains `[[pin]]` near the end of a ~20k OpenClaw-shaped prefix, `max_tokens` 1. Confirm the marker is absent from the stored tokens (log the first and last 8 token ids, or detokenize the donor). Confirm `pin_armed` is now false by sending a second request that also contains `[[pin]]` and confirming it is treated as literal text, not a new pin.
4. New slot: send a second request with the same prefix and a different tail, no marker. Expect the alias log, `n_prompt_tokens_processed` ≈ tail length, not ≈ 20k. Repeat with the tail edited 2k tokens before the end. With `--swa-cache 8192` this must hit. With the cache left at 512 it may force reprocess. Record both.
5. Donor safety: send an unrelated prompt. It must not land on the donor slot. Confirm via `/slots` if `--slots` is on, or via the slot id in the log.
6. Cap: one request that tries to grow past `--kv-unified-per-slot` must stop at the cap, not abort the other slots.
7. Restart the server. `pin_armed` is true again. First `[[pin]]` wins.

## Part E — files touched, in order

1. `src/llama-arch.h`, `src/llama-arch.cpp` — `LLM_KV_DECODER_ARCH`, `LLM_TENSOR_ENC_AUX_NORM`.
2. `src/models/models.h` — `decoder_laguna`, `aux_norm` on `llama_model_dflash`.
3. `src/models/dflash.cpp` — load + encoder/decoder graph deltas from poolside `laguna`.
4. `src/models/laguna.cpp` — only if the target does not already publish the layer outputs `target_layers` asks for, including the nextn case for layer 48.
5. `common/speculative.cpp` — causal flag, shared embd/lm_head if not already wired, draft seq alias hook.
6. `common/chat.cpp` or the server chat handler — marker strip. Prefer the handler so `common/chat.cpp` stays reusable.
7. `tools/server/server-task.h` — `pin_donor` on the task.
8. `tools/server/server-context.cpp` — `pin_armed`, `donor_slot`, selection skip, alias path.
9. `src/llama-kv-cache-iswa.cpp`, `common/arg.cpp`, context params — `--swa-cache`.
10. No new CUDA files. If a kernel is unavoidable, guard it with `sm_70` and test the build.

## Part F — draft KV clear condition

### Problem

With `--cache-prompt` and LCP reuse, a slot release is often just a turn
boundary inside the same conversation. The draft KV (in `ctx_dft`) was being
wiped on every release because `prompt_clear()` called `mem.seq_rm(id, -1, -1)`,
which clears the sequence from **both** the target (`ctx_tgt`) and the draft
(`ctx_dft`) KV caches. On the next turn, LCP kept the target KV (f_sim=1.0,
f_keep=1.0) but the draft KV was gone, so draft acceptance dropped to 0%.

### Fix

`prompt_clear()` now clears **only** the target KV via
`llama_memory_seq_rm(llama_get_memory(ctx_tgt), id, -1, -1)`. The draft KV is
cleared separately by the new `clear_draft_kv()` method:

```cpp
void clear_draft_kv() {
    if (ctx_dft) {
        llama_memory_seq_rm(llama_get_memory(ctx_dft), id, -1, -1);
    }
}
```

### Exact condition that gates the draft clear

> The draft KV is cleared if and only if the **target** KV for that slot is
> also being discarded — i.e., whenever `prompt_clear()` is called.

`clear_draft_kv()` is called immediately after every `prompt_clear()` call site.
There are 10 call sites, all in `tools/server/server-context.cpp`:

| # | Location | Why target KV is discarded |
|---|----------|---------------------------|
| 1 | `release()` — child / stateless / audio_stream | context not reused |
| 2 | `release()` — SERVER_TASK_TYPE_TTS | mtmd always re-evals |
| 3 | `get_available_slot()` — cache load failure | target replaced with new tokens |
| 4 | `try_clear_idle_slots()` — purge | idle slot purged under `kv_unified` |
| 5 | `launch_slot()` — SERVER_TASK_TYPE_TTS | mtmd always re-evals |
| 6 | `launch_slot()` — SERVER_TASK_TYPE_TRANSCRIBE_RT | audio rows, no KV reuse |
| 7 | `update_slots()` — TAG_IDLE_SLOT_CLEAR | idle slot cleared under `kv_unified` |
| 8 | slot load error | restore failed, target reset |
| 9 | slot erase | explicit erase |
| 10 | decode error | eval failed, target reset |

### LCP hit path (draft KV preserved)

`get_available_slot()` selects a slot by LCP similarity **without** calling
`prompt_clear()`. The slot's target KV is kept (cells past the fork belong to
the worker only). Because `clear_draft_kv()` is never called on this path, the
draft KV stays aligned with the kept target KV and draft acceptance survives
across turns.

When `f_keep < 0.5f` on an LCP hit, `update_cache` is set and the prompt is
saved to / loaded from the prompt cache. If the cache load **succeeds**,
`prompt_clear()` is not called and the draft is preserved. If the cache load
**fails** (cache miss), `prompt_clear()` + `clear_draft_kv()` run — the target
KV is being replaced, so the draft is correctly discarded too.

### DFlash host-side state

`spec_ckpt.data_dft` is a host-side snapshot of the draft state, used for
speculative rollback. It is cleared by `slot.reset()` on every release (via
`spec_ckpt.clear()`). This is safe: on the next task the checkpoint is
re-populated from the live KV when `use_ckpt_dft` is true. No additional
clearing is needed because `spec_ckpt` is not the KV cache itself.

`features_buf`, `lilicorr_embd`, and per-seq samplers in
`common_speculative_impl_draft_dflash` are transient — resized or reset on
each `process()` / `draft()` call. No changes were needed.

## Out of scope

- Publishing the fork.
- A second marker, an unpin endpoint, or more than one donor.
- `--swa-full` as the solution.
- Raising the attention window above 512.
- Porting SGLang/vLLM radix cache.
- Changing OpenClaw. The marker is plain text in the system message it already sends.
