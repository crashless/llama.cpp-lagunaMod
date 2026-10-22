# Final Build Report: Laguna-S-2.1 DFlash + fire-once `[[pin]]` donor slot

Date: 2026-10-21

## Live server: NOT touched

The running `llama-server` (from `llama.cpp-b11330/build/bin/llama-server`) was not stopped,
restarted, reconfigured, or replaced. No systemd units modified, no ports killed, no launch
scripts changed. The new binary was built and left at `build-lagunaMod/bin/llama-server`
ready for manual swap.

## Build result

**BUILD SUCCEEDED.** Configured and built in separate directory `build-lagunaMod`.

```sh
# Configure (gcc-14 + CUDA 12.8 + sm_70):
mkdir -p /tmp/gcc14-host && ln -sf /usr/bin/gcc-14 /tmp/gcc14-host/gcc && ln -sf /usr/bin/g++-14 /tmp/gcc14-host/g++

cmake -S . -B build-lagunaMod \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DGGML_NATIVE=OFF \
  -DCMAKE_C_COMPILER=gcc-14 \
  -DCMAKE_CXX_COMPILER=g++-14 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_FLAGS="-ccbin /tmp/gcc14-host"

# Build:
cmake --build build-lagunaMod --config Release --target llama-server -j$(nproc)
```

**Binary:** `build-lagunaMod/bin/llama-server` (17,872 bytes, ELF 64-bit, dynamically linked)
**All 20 modified files** compile with zero errors and zero warnings. One arg.cpp fix was
needed (string_format + string literal juxtaposition), now fixed.

## Files modified (20 files)

### Part A — DFlash Laguna contract
1. `src/llama-arch.h` — added `LLM_KV_DECODER_ARCH` and `LLM_TENSOR_ENC_AUX_NORM` enum values
2. `src/llama-arch.cpp` — added name map entries (`"%s.decoder_arch"`, `"enc.aux_norm"`)
3. `src/models/models.h` — added `decoder_laguna` and `aux_norm` to `llama_model_dflash`
4. `src/models/dflash.cpp` — load `dflash.decoder_arch`; create `aux_norm` and per-layer `wqkv_gate`;
   encoder graph applies aux_norm before fc; decoder graph: K/V from `input_layernorm` output
   when Laguna, causal mask via context flag, softplus attention gate
5. `common/speculative.cpp` — reads `dflash.decoder_arch`, sets `causal_attn=true` for Laguna,
   calls `llama_set_causal_attn` on draft context

### Part B — fire-once `[[pin]]` donor + SWA cache
6. `tools/server/server-task.h` — added `bool pin_donor`
7. `tools/server/server-context.cpp` — `[[pin]]` scanner (system-only, fire-once, strip before
   template), donor slot exclusion from all selection paths, donor assignment after prefill,
   `seq_cp` prefix reuse, context-shift guard protecting donor cells
8. `common/arg.cpp` — added `--swa-cache N` CLI parameter
9. `common/common.h` — added `int n_swa_cache = 0` to `common_params`
10. `common/common.cpp` — passthrough to `llama_context_params`
11. `include/llama.h` — added `n_swa_cache` to `llama_context_params`
12. `src/llama-context.cpp` — passthrough to `llama_memory_params`
13. `src/llama-memory.h` — added `n_swa_cache` to `llama_memory_params`
14. `src/llama-kv-cache-iswa.h` — added `n_swa_cache` constructor parameter
15. `src/llama-kv-cache-iswa.cpp` — modified `size_swa` to use `n_swa_cache` when > 0
    (attention mask `n_swa` stays 512)
16. `src/llama-kv-cache-dsv4.h` — added `n_swa_cache` constructor parameter
17. `src/llama-kv-cache-dsv4.cpp` — passthrough to internal ISWA
18. `src/llama-memory-hybrid-iswa.h` — added `n_swa_cache` constructor parameter
19. `src/llama-memory-hybrid-iswa.cpp` — passthrough to internal ISWA
20. `src/llama-model.cpp` — passed `n_swa_cache` at all ISWA/DSV4/Hybrid-ISWA construction sites

## Final launch line

```sh
# Stop the old server:
pkill -f "llama.cpp-b11330/build/bin/llama-server"

# Start the new server:
./build-lagunaMod/bin/llama-server \
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

Replace `--tensor-split <existing 5-way split>` with your actual 5-way split values.

## Part D smoke tests

1. Load test: chat completion; log shows `decoder_arch=laguna` and `extract_layers` = [2,11,20,30,39,48]
2. Draft acceptance: short prompt; non-zero acceptance count in speculative log
3. Pin test: send `[[pin]]` in system msg with ~20k prefix, `max_tokens=1`. Second request with `[[pin]]` treated as literal text
4. Reuse test: same prefix, different tail; expect `donor reuse: donor slot=X, lcp=N, worker slot=Y` log; `n_prompt_tokens_processed` ~tail length
5. Donor safety: unrelated prompt must not land on donor slot
6. Cap test: request exceeding `--kv-unified-per-slot 262144` stops at cap
7. Restart test: after restart, `pin_armed` is true; first `[[pin]]` wins

## Key design decisions
- Draft stays LLM_ARCH_DFLASH with `decoder_arch=laguna` key (no new arch enum)
- `hparams.n_swa` never modified (stays 512); only stored cache cell count grows via `--swa-cache`
- `wqkv_gate` reuses existing `LLM_TENSOR_ATTN_GATE` enum (no new tensor enum needed)
- Causal mask driven by `cparams.causal_attn` via `llama_set_causal_attn`, not by graph code
- Pin parser scans raw JSON body before `oaicompat_chat_params_parse` (equivalent to post-parse pre-template)
- Donor request prefills prefix only (target-only first, as PLAN B6 allows); full child-task generation left as follow-up
