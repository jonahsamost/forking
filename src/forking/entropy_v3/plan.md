# Entropy V3: Hint Injection

## Motivation

The V2 classifier reliably separates failing from succeeding rollouts (youden_j ~0.28,
noop_rate ~20-30%), but the intervention — resample branches, score by VIX, pick the
"calmest" branch — doesn't improve actual rewards. VIX is a proxy that doesn't correlate
with success. The classifier knows *when* things go wrong; V2 just doesn't know *what to
do about it*.

V3 replaces the branching intervention with **hint injection**: when the classifier fires,
inject a short "wait, let me reconsider" string into the token stream and let the model
continue generating from there. No branch sampling, no VIX scoring, no proxy metric.

## Design

### Server-side: `entropy_handler.py`

**Init changes:**
- Accept a `tokenizer` and a list of hint strings (e.g. from config or hardcoded defaults).
- Pre-tokenize at `__init__` time:
  - Each hint string → `list[int]` via `tokenizer.encode(s, add_special_tokens=False)`.
    Store as `self.hint_token_ids: list[list[int]]`. These are safe to pre-tokenize
    because every hint starts at a word boundary (capital letter after `.`/`\n`/`...`)
    so BPE merges won't cross the splice point.
  - An ellipsis `"..."` → `self.ellipsis_token_ids: list[int]`. Prepended to the hint
    when no natural sentence boundary is found (see backward scan below).
  No tokenization happens at intervention time — just `random.choice`.
- Drop `num_samples` parameter (no branching).

**Intervention flow** (replaces lines 472-552 of V2):

1. Classifier fires → `detected_idx` is the window with highest failure probability.
2. Compute split point at the **beginning** of the detected window:
   `split_idx = max(0, detected_idx - chunk_size + 1)` — intervene before the
   trouble starts, not in the middle of it.
3. **Backward scan to natural boundary:** from `split_idx`, scan backward up to
   `max_backtrack` tokens (default: 32). For each candidate token, decode it
   with `tokenizer.decode([token_id])` and check if the decoded text ends with
   `"."` or `"\n"` (after stripping trailing whitespace). This is robust across
   tokenizers — avoids needing to enumerate all possible period/newline token IDs
   (e.g. `"."` vs `" ."` vs `".\n"`). If found, split **after** that token (keep
   the boundary). Only a handful of single-token decode calls per intervention.
4. Keep `chunk_ids[:split_idx]`, discard the rest.
5. Pick a random pre-tokenized hint via `random.choice(self.hint_token_ids)`.
6. **Bridge token:** if the backward scan found a natural boundary (period or
   newline), inject the hint directly. If no boundary was found within
   `max_backtrack`, prepend `self.ellipsis_token_ids` (`"..."`) before the hint
   to create a grammatical bridge from the mid-token split point.
7. Append kept tokens + (optional ellipsis) + hint tokens to `all_token_ids`.
8. **Continue the generation loop.** The next `_generate` call uses
   `prompt_ids + all_token_ids` which now includes the hint as context.
9. Increment `interventions_used`, record `split_indices` and `injection_ranges`.
   The injection range covers the ellipsis (if any) + hint tokens.

**What to delete from the V2 copy:**
- `_branch_score` method
- All branch generation logic (`_generate` with `n=num_samples`)
- VIX fork-point search (rolling VIX, max-VIX-pos within window)
- `num_samples` param and related config

### Logprobs for injected tokens

**Problem:** Injected tokens weren't sampled from the model. The trainer needs
`old_log_probs` for the policy gradient ratio `exp(new_logprob - old_logprob)`.

**Approach:** When we continue generation after injecting the hint, the injected
tokens are part of the prompt for the next `_generate` call. vLLM processes them
during prefill. We pass `prompt_logprobs=self.topk_entropy` (e.g. 3) in
SamplingParams for that call. vLLM returns per-position logprobs for all prompt
tokens via `RequestOutput.prompt_logprobs`.

These are **honest old_log_probs** — they reflect what the model would assign to
those tokens if asked. Since injected tokens are unmasked by default, this matters:
the policy ratio starts at ~1 and evolves naturally during training. With fabricated
0.0 logprobs, the ratio would always hit the clip boundary (the model almost certainly
assigns log_prob << 0 to "Wait" mid-generation), producing a constant biased gradient
that never attenuates.

**vLLM `prompt_logprobs` format** (from `vllm/logprobs.py` and `vllm/outputs.py`):
- `RequestOutput.prompt_logprobs` is `PromptLogprobs | None`.
- `PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]`.
- `LogprobsOnePosition = dict[int, Logprob]` where `Logprob` has `.logprob`, `.rank`.
- Index 0 is always `None` (first prompt token has no left context).
- Index `i` contains top-N logprobs at position `i` given tokens `[0..i-1]`.

**Extraction:** The injected tokens sit at known positions within the prompt. After
injection, the full prompt for the next `_generate` call is
`prompt_ids + all_token_ids` (which includes the hint). The injection spans the
last `len(hint_ids)` (+ optional ellipsis) tokens of that prompt. To extract:

```python
prompt_len = len(prompt_ids + all_token_ids)  # after injection appended
injection_start = prompt_len - len(injected_ids)  # ellipsis + hint

for i, pos in enumerate(range(injection_start, prompt_len)):
    pos_logprobs = result.prompt_logprobs[pos]  # dict[token_id -> Logprob]
    token_id = injected_ids[i]
    honest_logprob = pos_logprobs[token_id].logprob
```

**For entropies:** With `prompt_logprobs=self.topk_entropy`, each position also
returns top-k logprobs. We extract the top-k at each injected position and feed
them through `compute_entropy_trajectory` the same way sampled tokens are handled.
This keeps the entropy trajectory continuous so the classifier doesn't see an
artificial discontinuity and misfire on the next chunk.

**Implementation:** Modify `_generate` to optionally accept `prompt_logprobs=N`.
When set, it returns `(outputs, prompt_logprobs)` instead of just `outputs`.
Currently `_generate` returns `final_result.outputs`; the change is to also
return `final_result.prompt_logprobs` from the `RequestOutput`. The intervention
path uses this only on the first generate call *after* an injection — subsequent
calls (if no new injection) skip it.

### Response format

The response `entropy` dict adds:
- `injection_ranges`: `list[list[int]]` — `[[start, end], ...]` marking injected
  token spans within the completion token_ids.
- Keep existing: `intervened`, `interventions_used`, `split_indices`, `vix` metadata.

The `token_ids`, `token_logprobs` arrays in the response already include the injected
tokens inline (with their honest logprobs). The trainer uses `injection_ranges` to
build the mask.

### Trainer side: `async_rollout_worker.py`

In `_generate_one` / `_generate_one_turn`:
- Read `injection_ranges` from `entropy_metadata`.
- Build `tool_mask`:
  - If `mask_injected_tokens=True`: set 0 at injected positions (like tool responses).
  - If `mask_injected_tokens=False` (default): set 1 at injected positions. The model
    trains on them. Honest logprobs ensure the ratio is well-behaved.
- `completion_logprobs` already has the correct values from the server response.

### Masking config

```yaml
# train.yaml
mask_injected_tokens: false  # default: train on injected tokens (off-policy but bounded)
```

**Unmasked (default):** Injected tokens participate in the loss. With positive
advantage the model learns to increase P("wait, let me reconsider") at similar
positions — potentially internalizing self-correction over time. Honest logprobs
keep the ratio near 1; clipping bounds any residual off-policy error.

**Masked:** Injected tokens excluded from loss. Safer, equivalent to the tool_mask
pattern. The model never learns to produce them; requires inference-time injection
permanently.

### Hint strings

```python
HINT_STRINGS = [
    "Wait, let me reconsider.\n\n",
    "Hmm, I should rethink this step.\n\n",
    "Hold on, let me try a different approach.\n\n",
    "Actually, let me reconsider.\n\n",
    "Let me re-examine this.\n\n",
    "Wait, that doesn't seem right.\n\n",
]
```

End with `\n\n` to give the model a clean paragraph break to continue from.
Diversity prevents the model from memorizing a single trigger string.

## Files to modify

| File | Change |
|------|--------|
| `entropy_v3/entropy_handler.py` | Rewrite `intervention_handler` (hint injection replaces branching). Add tokenizer + hint init. Add `prompt_logprobs` extraction. Delete `_branch_score`, VIX fork-point, `num_samples`. |
| `entropy_v3/models.py` | Add `INJECTION_RANGES_KEY`. Drop unused VIX-branch keys if any. |
| `entropy_v3/train.py` | Add `mask_injected_tokens` config. |
| `entropy_v3/vllm_server.py` | Pass tokenizer to `EntropyHandler.__init__`. |
| `async_rollout_worker.py` | Read `injection_ranges` from entropy metadata, apply mask toggle. |

## What stays unchanged from V2

- `classifier.py`, `classifier_manager.py`, `features.py` — the classifier decides
  *when* to intervene, and that part works well. No changes.
- `entropy_updates.py` — control samples feed classifier training. Same flow.
- A/B testing (`sample_idx % 2`).
- `non_intervention_handler` — untouched.
- `train.yaml` structure (just add new fields).
