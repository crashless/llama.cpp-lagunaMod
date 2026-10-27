# STATUS: Hybrid/recurrent cache reuse + [[pin]] donor fixes (Oct 26 2026)

## Binary

- **Path**: `build-lagunaMod/bin/llama-server` (dynamically linked thin wrapper)
- **Shared libs**: `libllama-server-impl.so`, `libllama.so.0`, `libggml-cuda.so.0`, etc. in `build-lagunaMod/bin/`
- **Build flags**: GCC-14, CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES=70`, `GGML_NATIVE=OFF`
- **Build result**: SUCCESS (all objects compiled, linked cleanly)
- **Live server**: NOT restarted by me. Ready for user to restart with the same launch args.

## Investigation summary

### PR #21099 (prompt cache reuse for hybrid/recurrent models)

Already present and active. `server-context.cpp`:
- `prompt_clear()` (line ~381) clears **only** target KV
- `clear_draft_kv()` (line ~393) clears **only** draft KV
- LCP hit path (line ~1884) clears stale draft KV when `f_keep > 0`
- `n_cache_reuse` disabled when `llama_memory_can_shift()` returns false

### PR #24797 (checkpoint handling for hybrid memory models)

Already present and active. `server-context.cpp`:
- `prompt_cache_restored` flag guards checkpoint path
- `do_checkpoint` enabled for hybrid/recurrent models
- Checkpoint data structure saves both target and draft state

### [[pin]] fire-once donor mechanism

Already present and active. `server-context.cpp`:
- `pin_armed` flag (fire-once, set to false after first hit)
- `pending_pin_donor` one-shot flag
- `rfind("[[pin]]")` reverse search in `post_chat_completions`
- Donor slot assigned in decode loop after prefill
- Donor excluded from all selection paths
- Donor reuse via `slot.mem.seq_cp(donor_slot, slot.id, 0, n_copy)` (line ~4189)

### GDN recurrent state handling

`llama_memory_hybrid::seq_cp` (src/llama-memory-hybrid.cpp:152) **does** copy both attention KV and recurrent state. The donor reuse path uses `common_memory::seq_cp` which delegates to hybrid memory's `seq_cp`. However, **stale recurrent state at positions beyond `n_copy` was NOT being cleaned up**.

## Changes made

### File 1: `tools/server/server-context.cpp`

**1a. Stale KV cleanup after donor seq_cp (line ~4200)**

After `slot.mem.seq_cp(donor_slot, slot.id, 0, n_copy)`, added:
```cpp
slot.mem.seq_rm(slot.id, n_copy, -1);
```
This clears stale attention KV + recurrent state at positions `[n_copy, -1)` that would otherwise persist from a prior prompt cache restore or previous decode. Critical for hybrid/recurrent models (Qwen3.5 MoE, Ornith-1.5-122B) where stale recurrent state at positions beyond the donor prefix could corrupt the reused prefix during subsequent decode.

**1b. Explicit CACHE HIT / CACHE MISS logging (lines ~4178, ~4208)**

- `SRV_INF` "CACHE HIT" when donor reuse succeeds (LCP >= threshold and > n_past)
- `SLT_INF` "CACHE HIT" with full details (donor, worker, lcp, n_past) on success
- `SLT_INF` "CACHE MISS" when donor reuse is skipped

**1c. Decode batch slot IDs logging (line ~4728)**

Added pre-yield snapshot of batch slot IDs with `SRV_INF`, so the next embeddings flag leak can be traced to the exact slots involved.

### File 2: `src/llama-batch.cpp`

**2. Embeddings flag logging before warning (line ~212)**

Before the "embeddings required but some input tokens were not marked as outputs" warning, added a detailed log showing:
- `output_all` (the `cparams.embeddings` value)
- `n_tok` (number of tokens in batch)
- `seq_ids` (which sequences/slots are in the batch)

This lets us identify exactly which request left the embeddings flag set.

## What was NOT changed (and why)

- **PR #21099 and #24797 logic**: Already correctly applied, no changes needed.
- **`common_memory::seq_rm` return value**: Ignoring the return value is a known pattern across 10+ call sites. Changing it would be too invasive for this scope. The stale KV cleanup uses `seq_rm` directly on `slot.mem`, which dispatches to `llama_memory_hybrid::seq_rm` -- this is atomic (recurrent first, abort on failure).
- **TTS embeddings reset**: Already present at line 3714 (`llama_set_embeddings(ctx_tgt, false)` after `process_tts_slots`). The decode path reset at line 4732 is also already present and unconditional.
- **Laguna DFlash path**: Untouched. Draft KV clearing on LCP hits (Part F of PLAN.md) untouched.

## Risks and TODOs

1. **`seq_rm` on recurrent memory may refuse partial removal**: If `llama_memory_hybrid::seq_rm` returns false (recurrent cache refuses), the attention cache is also not modified (atomic). The `SLT_INF` log will show whether the cleanup ran. Monitor logs for donor reuse on hybrid models.

2. **`common_memory::seq_rm` return value still unchecked**: The cleanup uses `slot.mem.seq_rm(slot.id, n_copy, -1)` which calls `common_context_seq_rm` on both `ctx_tgt` and `ctx_dft`. If `ctx_tgt` succeeds but `ctx_dft` fails (or vice versa), state could be inconsistent. This is an existing pattern and not specifically addressed here. TODO: consider returning bool from `common_memory::seq_rm` in a follow-up.

3. **Embeddings flag leak root cause not fully fixed**: The logging helps diagnose but the fundamental concurrency issue (TTS worker and decode worker sharing `cparams.embeddings` on `ctx_tgt`) remains. A proper fix would snapshot `output_all` into a local in `llama_process` or use a thread-local. TODO for a follow-up if the leak persists.

4. **Donor reuse only restores target KV**: The draft KV (speculative) is also transferred via `seq_cp` (which calls `common_memory::seq_cp` on both `ctx_tgt` and `ctx_dft`), but this has not been explicitly verified with a DFlash draft. Monitor draft acceptance rate.

## Test commands

### 1. Verify pin detection and donor assignment

```bash
# Start the new server (do NOT restart the running one)
# Use a different port:
build-lagunaMod/bin/llama-server \
  -m /path/to/qwen35moe-Q4_K_M.gguf \
  --port 8081 \
  -c 262144 -np 5 \
  --kv-unified --kv-unified-per-slot 262144 \
  --cache-prompt \
  -fa on --jinja

# Send a request with [[pin]] in the system message:
curl -s http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35moe",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. [[pin]]"}
    ],
    "max_tokens": 1
  }'
```

Expected log lines:
- `pin scan: pin_armed=1, has_messages=1`
- `pin scan: found [[pin]] at pos=...`
- `pin path: pin_armed=false, pending_pin_donor=true`
- `pin path: donor slot=%d assigned`

### 2. Verify donor reuse (same prefix, different slot)

```bash
# Second request with same prefix but different tail:
curl -s http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35moe",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."}
      {"role": "user", "content": "What is the capital of France? [[tail variation}}
    ],
    "max_tokens": 10
  }'
```

Expected log lines:
- `pin path: donor reuse CACHE HIT, donor_slot=%d, lcp_len=...`
- `pin path: seq_cp completed, copying %zu tokens`
- `pin path: clearing stale KV at positions [%zu, -1)`
- `donor reuse: CACHE HIT donor slot=...`
- `n_prompt_tokens_processed` should be ~tail length, not full prefix length

### 3. Verify fire-once behavior

```bash
# Send another request with [[pin]]:
curl -s http://localhost:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen35moe",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant. [[pin]]"}
    ]
  }'
```

Expected: `[[pin]]` is treated as literal text, no new donor assignment. Log should show `pin_armed=0` and `pending_pin_donor=0`.

### 4. Verify embeddings flag logging

```bash
# Send a request that might trigger the embeddings warning:
curl -s http://localhost:8081/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen35moe", "input": "test"}'
```

If the warning fires, the log should now show:
- `embeddings=%d required but some input tokens were not marked as outputs -> overriding (n_tok=%d, n_seq=%d, seq_ids=[...])`
- `  caller=decode output_all=%d n_tok=%d (embeddings leaked into batch)`

---

## Embedding flag leak fix (Oct 27 2026)

### Problem

The previous attempt reset `cparams.embeddings = false` immediately before
`llama_process` on the pin/decode path. The warning still fired right after the
reset log, still seeing `embeddings=1`.

Log sequence that reproduced:
```
pin path: decode batch slots=[1,1,1,...], resetting embeddings=false before llama_process
W init: embeddings=1 required but some input tokens were not marked as outputs -> overriding (n_tok=N, seq_ids=[...])
```

The warning originates from `llama_batch_allocr::init` in `src/llama-batch.cpp`,
called from `llama_context::decode` (line 1902). `decode` reads
`const bool output_all = cparams.embeddings` at line 1864, then passes it to
`balloc->init`. The reset at the server level (inside `yield_to_queue`) sets
`cparams.embeddings = false`, but `llama_context::decode` (which runs later,
inside `llama_process`) reads the value from the shared `cparams` struct. Between
the reset and that read, a concurrent writer can flip the flag back to true.

The worker thread during `yield_to_queue` only processes METRICS and SLOT_GET
tasks (`process_single_task` at line 3027 declines all other types), so it does
not directly call `llama_set_embeddings`. The race window is between the last
`llama_set_embeddings` call and the read in `decode`; the fix below snapshots the
intended value as close to `llama_process` as possible and logs diagnostics to
detect any residual leak.

### Changes

**1. `src/llama-batch.cpp` - Reduced log spam**

The warning previously dumped every seq_id in the batch (e.g. `seq_ids=[1,1,1,...]`
for n_tok=94). Now prints `n_seq` (total count) and a preview of the first 3
seq_ids, followed by `...` if truncated.

Added a one-line diagnostic at the warning site showing the `caller_tag`
(`decode` or `encode`), `output_all`, and `n_tok`, so it is immediately clear
which code path triggered the override.

**2. `src/llama-batch.h` - Added `caller_tag` parameter**

`init()` now accepts an optional `const char * caller_tag = nullptr`. Existing
callers that don't pass it compile unchanged.

**3. `src/llama-context.cpp` - Diagnostic logs + caller tags**

- `decode()` (line 1866): logs `output_all`, `cparams.embeddings`, `this` pointer,
  and `n_tok` at debug level before calling `balloc->init`. Passes `"decode"` as
  caller_tag.
- `encode()` (line 1617): logs `this` pointer and `n_tok` at debug level. Passes
  `"encode"` as caller_tag. The encode path hardcodes `output_all=true` per its
  comment ("during encode, we always output all tokens").

**4. `tools/server/server-context.cpp` - Fixed reset strategy + log cleanup**

The pin/decode path (line 4752) previously:
- Dumped ALL slot IDs into a string (`slots=[1,1,1,...]` for up to 94 tokens)
- Unconditionally called `llama_set_embeddings(ctx_tgt, false)`

Now:
- Logs only count + first few slot IDs (`n_slots=5, slots=[1,2,3...]`)
- Snapshots `need_embd()` from the batched slot BEFORE the yield, so the value
  cannot be affected by concurrent state changes
- Calls `llama_set_embeddings(ctx_tgt, embeddings_needed)` where
  `embeddings_needed` is from the snapshotted `need_embd()`

Why this is better than the old unconditional `false` reset:
- For regular generation (`need_embd() == false`), behavior is identical: sets
  `cparams.embeddings = false` right before `llama_process`.
- For embedding batches (`need_embd() == true`), the old code would incorrectly
  force `false`, potentially breaking embedding computation. The new code
  preserves `true`.
- The snapshot is taken outside the `yield_to_queue` lambda, so it cannot be
  changed by concurrent workers.

### What was preserved

- Laguna DFlash path (`src/models/dflash.cpp`, `src/models/laguna.cpp`): untouched
- Draft KV clearing on LCP hits (PLAN.md Part F): `prompt_clear()` /
  `clear_draft_kv()`: untouched
- Pin fire-once donor-slot mechanism: `pin_armed`, `donor_slot`,
  `pending_pin_donor`: untouched
- TTS embeddings reset at line 3719: untouched
- Regular batch path at line 3597: untouched

### How to confirm the warning is gone

After restarting the server:

```bash
# Normal generation - should NOT produce the embeddings warning
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laguna-s-2.1",
    "messages": [{"role":"user","content":"Hello world"}],
    "max_tokens": 10
  }' | tail -c 200

# Embeddings endpoint - may produce the warning if output_all=true
curl -s http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"model":"laguna-s-2.1","input":"test"}'
```

Expected log sequence for normal generation:
```
pin path: decode batch n_slots=1, slots=[1], embeddings_needed=0
```
No `W init:` line should follow. The debug log in `decode` will also show:
```
D init: balloc->init caller=decode output_all=0 cparams.embeddings=0 this=0x... cparams=0x... n_tok=1
```

---

## Pin-path fixes (Oct 27 2026)

### Issue A — embeddings flag leak into generation batches

**Symptom (log sequence previously triggering the warning):**

```
D decode: balloc->init caller=decode output_all=0 cparams.embeddings=0 this=0x... cparams=0x... n_tok=N
pin path: decode batch n_slots=N, slots=[...], embeddings_needed=0
W init: embeddings=1 required but some input tokens were not marked as outputs -> overriding (n_tok=N, ...)
  caller=encode output_all=1 n_tok=N cparams=0x... (embeddings leaked into batch)
```

**Root cause:**

The server resets `cparams.embeddings = false` before `llama_process` (line 4760),
but `llama_context::decode()` (invoked inside `llama_process`) redirects to
`llama_context::encode()` when `memory == nullptr` (true for contexts that lack a
KV memory module, e.g. certain DFlash paths). The encode path hardcoded
`output_all = true` unconditionally, ignoring `cparams.embeddings`. This forced
every token to be marked as an output, triggering the warning even though the
server had cleared the embeddings flag.

**Changes:**

1. **`src/llama-context.cpp` -- `encode()` (line ~1623):** Replaced the hardcoded
   `true` with a snapshot of `cparams.embeddings` taken at the start of the
   function. This is snapshot before `balloc->init` so concurrent writers cannot
   change the flag mid-batch. When `encode()` is reached via the
   `decode() -> encode()` redirect, `cparams.embeddings` reflects the server's
   last `llama_set_embeddings` call, so a generation batch no longer forces
   `output_all = 1`.

2. **`src/llama-batch.h` / `src/llama-batch.cpp` -- `init()` (line ~28):** Added a
   `uintptr_t caller_id` parameter (default 0). The warning site now prints the
   address of the `llama_context_params` instance whose `embeddings` flag was
   read: `caller=encode output_all=1 n_tok=N cparams=0x...`. This confirms whether
   the encode path reads the same `cparams` instance the server resets -- it does:
   both are `this->cparams` on `ctx_tgt`.

3. **`src/llama-context.cpp` -- `decode()` and compat batch:** Both pass
   `(uintptr_t)&cparams` as `caller_id`.

**What was preserved:** Laguna DFlash path, draft-KV clearing on LCP hits, TTS
embeddings reset, and prompt cache restore logic are all untouched.

### Issue B -- crash on donor reuse CACHE HIT

**Symptom (log sequence previously triggering the assertion at line 4239):**

```
pin path: donor reuse CACHE HIT, lcp_len=19415, n_past=0
pin path: seq_cp completed, copying 19414 tokens to prompt
pin path: clearing stale KV at positions [19414, -1) after donor reuse
SLT_ERR: n_past = 19414, slot.prompt.tokens.size() = 19414, seq_id = N, pos_min = -1
GGML_ABORT: pos_min == -1, but n_past > 0 - should not happen
```

**Root cause:**

After `slot.mem.seq_cp(donor_slot, slot.id, 0, n_copy)` copies the donor KV to the
worker slot at `[0, n_copy)`, the code unconditionally called
`slot.mem.seq_rm(slot.id, n_copy, -1)` to clear stale KV beyond the donor prefix.
When `n_past == 0` (no prior prompt cache restore), there is no stale KV to clear.
The `seq_rm` with range `[n_copy, -1)` (i.e. `[n_copy, max)`) instead also clears
the donor's recurrent tail cell, setting `pos_min = -1` while `n_past > 0`,
tripping the invariant assertion.

**Changes:**

1. **`tools/server/server-context.cpp` -- donor reuse block (line ~4192):** Captured
   `n_past` into `n_past_before` before the seq_cp. The `seq_rm` now uses
   `n_past_before` as the upper bound and only runs when `n_past_before > 0`:
   ```cpp
   const int n_past_before = n_past;
   // ... seq_cp ...
   if (n_past_before > 0) {
       slot.mem.seq_rm(slot.id, n_copy, n_past_before);
   }
   ```
   Since the donor reuse condition requires `lcp_len > n_past` (so
   `n_copy >= n_past_before`), the range `[n_copy, n_past_before)` is empty when
   `n_past_before > 0`, and the seq_rm is skipped entirely when `n_past == 0`.
   The donor's KV at `[0, n_copy)` (including the recurrent tail) is preserved,
   keeping `pos_min > 0`.

2. **Log level reductions in the donor reuse path:** Changed the per-step
   `SRV_INF`/`SLT_INF` logs for seq_cp and the donor-reuse entry to `SRV_DBG`/
   `SLT_DBG`, keeping only the `CACHE HIT`/`CACHE MISS` result at `INF`.

**What was preserved:** The donor reuse seq_cp itself, the draft KV clearing on
LCP hits (Part F of PLAN.md), the fire-once pin mechanism, and the Laguna DFlash
model code are all untouched.

### Build

- **Binary path**: `build-lagunaMod/bin/llama-server`
- **Flags**: GCC-14, CUDA 12.8, `CMAKE_CUDA_ARCHITECTURES=70`, `GGML_NATIVE=OFF`
- **Live server**: NOT restarted. User must restart manually after the build.

### How to confirm both fixes

After restarting the server:

```bash
# 1. Normal generation -- should NOT produce the embeddings warning
curl -s http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"laguna-s-2.1","messages":[{"role":"user","content":"Hello"}],"max_tokens":10}'

# 2. Donor reuse -- should NOT crash, pos_min should be >= 0
# Send two requests with the same prefix on different slots.
# Expected: "donor reuse CACHE HIT" then "CACHE HIT donor slot=..." with no
#   "clearing stale KV" log and no assertion abort.
```
