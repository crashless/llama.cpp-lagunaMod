# Optimization Findings: poolside laguna vs lagunaMod

## Overview

Comparison of the poolside `laguna` fork (`~/llama.cpp-laguna`) against the
lagunaMod fork (`~/llama.cpp-lagunaMod`) to identify performance-critical
differences in DFlash speculative decoding and CUDA graph handling.

The lagunaMod fork is based on a newer llama.cpp master with many features
(DSpark, DFlash2, LiLiCorr, DSV4, conv support) but introduced regressions
in the core DFlash Laguna contract.

---

## Findings (4 critical, 4 non-critical)

### CRITICAL #1: Missing `t_h_nextn` in target Laguna graph

**File:** `src/models/laguna.cpp` (both forks)

**Poolside fork** (laguna.cpp:329-334):
```cpp
cur = inpL;
cb(cur, "h_nextn", -1);
res->t_h_nextn = cur;   // pre-final-norm residual stream
cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
```

**LagunaMod fork** (laguna.cpp:336-338):
```cpp
cur = inpL;
cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
// t_h_nextn stays nullptr!
```

The poolside fork sets `t_h_nextn` to the pre-final-norm residual stream,
making it available via `llama_get_embeddings_nextn(ctx_tgt)`. The lagunaMod
fork never sets it, so extraction of layer 48 (the post-loop residual) returns
nullptr. Every other model arch in the lagunaMod fork sets `t_h_nextn` --
laguna.cpp is the only one that doesn't.

**Fix:** Add `res->t_h_nextn = cur;` before `build_norm` at line 337.

---

### CRITICAL #2: Missing `aux_norm` in DFlash decoder's embd path

**File:** `src/models/dflash.cpp` (both forks)

**Poolside fork approach** -- two graph calls:
1. `llama_encode(ctx_dft, enc_batch)` runs the encoder graph:
   `aux_norm` -> `fc` -> `output_norm_enc`
2. `llama_decode(ctx_dft, batch_inject)` runs the decoder graph, projecting
   K/V from already-fused features

**LagunaMod fork approach** -- single graph call with inline fusion:
```cpp
// lagunaMod dflash.cpp:899-901 (decoder embd path)
ggml_tensor * inp_g = build_lora_mm(model.fc, inp_target, model.fc_s);       // NO aux_norm
inp_g = build_norm(inp_g, model.output_norm_enc, NULL, LLM_NORM_RMS, -1);
```

The `aux_norm` path exists in the encoder graph (`<true>`, lines 388-396) but
`process()` never calls the encoder graph -- it goes straight to
`llama_process` with `LLAMA_PROCESS_TYPE_DECODE`. Laguna's per-aux RMSNorm is
never applied to target features before fusion, degrading draft quality and
acceptance rate.

**Fix:** Either:
- (a) Change `process()` back to the two-call pattern (`llama_encode` +
      `llama_decode` with bulk memcpy), OR
- (b) Add the `aux_norm` block (reshape 3d -> rms_norm -> mul -> reshape 2d)
      before `build_lora_mm(model.fc, ...)` in the decoder's embd path,
      copying the exact pattern from the encoder graph (lines 388-396).

---

### CRITICAL #3: CUDA graph architecture threshold reversed

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

```cpp
// poolside fork, line 3977:
if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_AMPERE)  // < 800 -> DISABLED for V100

// lagunaMod fork, line 4798:
if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA)  // < 700 -> ENABLED for V100
```

Constants:
- `GGML_CUDA_CC_VOLTA = 700` (sm_70 = V100)
- `GGML_CUDA_CC_AMPERE = 800` (sm_80 = A100)

The poolside fork disables CUDA graphs for V100 (700 < 800 is true).
The lagunaMod fork enables them (700 < 700 is false). On V100 with the newer
llama.cpp CUDA graph implementation, this can cause:
- Graph compatibility check failures leading to re-captures every step
- Stream synchronization issues from the `mul_mat_id` fallback path
- Graph eviction/re-capture thrashing under tensor-split across 5 cards

**Fix:** Change `GGML_CUDA_CC_VOLTA` to `GGML_CUDA_CC_AMPERE` in
`ggml_cuda_graph_set_enabled` at line 4798.

---

### CRITICAL #4: Process() uses per-token add_embd instead of bulk memcpy

**File:** `common/speculative.cpp` (both forks)

**Poolside fork** (speculative.cpp:1108-1143) -- bulk copy, two calls:
```cpp
llama_encode(ctx_dft, enc_batch);
const float * inp_g = llama_get_embeddings_nextn(ctx_dft);
batch_inject.n_tokens = n_chunk;
std::memcpy(batch_inject.embd, inp_g, n_chunk * n_embd_dec * sizeof(float));
llama_decode(ctx_dft, batch_inject);
```

**LagunaMod fork** (speculative.cpp:1265-1276) -- per-token loop:
```cpp
for (int32_t i = 0; i < n_chunk; ++i) {
    batch_inject.add_embd({ features_buf.data() + i * n_embd_enc, 1, n_embd_enc }, pos_arr, seq_id, false);
}
llama_process(ctx_dft, LLAMA_PROCESS_TYPE_DECODE, batch_inject.get());
```

The per-token `add_embd()` loop adds overhead for each token in the K/V
injection path, vs a single bulk `memcpy`.

---

## Non-critical differences

### 5. Non-finite sanitization

**Poolside fork** (speculative.cpp:1089-1104):
```cpp
if (!std::isfinite(v)) {
    v = v != v ? 0.0f : (v > 0.0f ? 65504.0f : -65504.0f);
}
```
Replaces NaN with 0, clamps Inf to +/-65504 (f16 max).

**LagunaMod fork** (speculative.cpp:1259-1262):
```cpp
if (!std::isfinite(features_buf[i])) {
    features_buf[i] = 0.0f;
}
```
Replaces all non-finite values with 0.

The poolside fork's clamping preserves large-but-finite values instead of
zeroing them, which could lead to better draft quality.

**Fix:** Change to clamp: `v != v ? 0.0f : (v > 0.0f ? 65504.0f : -65504.0f)`.

---

### 6. Laguna.cpp `inp_out_ids` handling

**Poolside fork** (laguna.cpp:268-273, 338-340):
```cpp
if (il == n_layer - 1 && inp_out_ids && cparams.embeddings_nextn_masked) {
    cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
    inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
}
// ...
if (!cparams.embeddings_nextn_masked && inp_out_ids) {
    cur = ggml_get_rows(ctx0, cur, inp_out_ids);
}
```

**LagunaMod fork**: Missing `inp_out_ids` handling entirely.

If `inp_out_ids` is used (e.g., for nextn extraction with masked outputs),
the lagunaMod fork would not apply the row gather.

---

### 7. Laguna.cpp layer input capture

Both forks capture `t_layer_inp[il]` inside the loop for layer input
extraction. But:

**Poolside fork**: Also unconditionally sets `t_h_nextn` (see #1).

**LagunaMod fork**: Only conditionally captures `t_layer_inp[n_layer]`
when `embeddings_layer_inp[n_layer]` is true (lines 330-334), but does not
set `t_h_nextn` for the nextn path.

---

### 8. Causal attention handling

**Poolside fork** (speculative.cpp:995-1002):
```cpp
bool causal = false;
if (llama_model_meta_val_str(model_dft, "dflash.decoder_arch", buf, sizeof(buf)) >= 0) {
    causal = strcmp(buf, "laguna") == 0;
}
llama_set_causal_attn(ctx_dft, causal);
```
Sets causal=true for Laguna.

**LagunaMod fork** (speculative.cpp:1055-1063):
```cpp
// reads dflash.attention.causal metadata
if (llama_model_meta_val_str(model_dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) {
    causal_attn = std::strcmp(buf, "true") == 0;
}
// DFlash block diffusion is non-causal; laguna drafters must not override
```
Does NOT set causal based on `decoder_arch == "laguna"`.

The PLAN.md documents that the causal=true behavior was intentionally removed
for laguna drafters (commit log: "Both were wrong: the laguna branch on the
upstream fork calls `llama_set_causal_attn(ctx_dft, false)`"). However, the
poolside fork's actual code sets causal=true for laguna. This needs
verification against the actual published GGUF metadata.

---

## Prioritized Fix List

| Priority | File | Fix |
|----------|------|-----|
| 1 | `src/models/laguna.cpp` | Add `res->t_h_nextn = cur;` before `build_norm` (line 337) |
| 2 | `common/speculative.cpp` | Restore two-call pattern: `llama_encode` + `llama_decode` with bulk memcpy (lines 1265-1276) |
| 3 | `ggml/src/ggml-cuda/ggml-cuda.cu` | Change `GGML_CUDA_CC_VOLTA` to `GGML_CUDA_CC_AMPERE` (line 4798) |
| 4 | `src/models/dflash.cpp` | If keeping inline fusion, add `aux_norm` block before `fc` in decoder embd path (line 900) |
| 5 | `common/speculative.cpp` | Change non-finite sanitization to clamp to +/-65504 (line 1260) |
| 6 | `src/models/laguna.cpp` | Port `inp_out_ids` handling from poolside fork if needed |
| 7 | (verify) | Check causal attention behavior against published GGUF metadata |

---

## Hardware context

- 5x PCIe Tesla V100 32GB (sm_70, no NVLink)
- CUDA 12.8, driver 580.178.04
- gcc-14 as C/CXX/CUDA host compiler
- Tensor-split across 5 cards, draft pinned to CUDA0 via `--spec-draft-override-tensor '.*=CUDA0'`
