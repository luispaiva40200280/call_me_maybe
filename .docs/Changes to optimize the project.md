### Step 1: Fix `Ġ` Normalization in `loaders.py` and `pipeline.py`

When loading your lookup table (`lut`) and `vocab` in `loaders.py`, normalize the BPE space character (`\u0120` / `Ġ`) into a normal space `' '`, or include `\u0120` in your `.strip()` calls.

- **Why it matters:** If `self.lut` maps `token_id -> token_str.replace("Ġ", " ").replace("▁", " ")`, then:
    
    1. Your generated strings won't contain literal `Ġ` characters.
        
    2. Your `VocabularyTrie` will match standard spaces `' '` naturally.
        
    3. Your string output can be parsed directly with `json.loads()` without needing `self.model.decode(gen_tokens)` at the end of `generate()`.

### Step 2: Bypass the LLM on Deterministic States in `pipeline.py` and `fsm.py`

Right now, your `StateMachine` already knows when only a single structural string is required! Look at your states in `src/fsm.py`:

- `EXPECT_ROOT_BRACE` -> `EXPECT_NAME_KEY` -> `EXPECT_NAME_COLON` always produces `'{"name": '`.
    
- Once `EXPECT_FUNCTION_NAME` selects `self.curr_func_call`, you already know the exact list of parameter keys (`list(self.curr_func_call.parameters.keys())`)!
    
- Instead of forcing the LLM to guess `{`, `"name"`, `:`, `,`, `"parameters"`, `:`, `{`, and every parameter key token-by-token, let your state machine emit a deterministic string whenever there is no semantic choice to make:st of that function name, the ", "parameters": {, and the first parameter key " + first_key + ":  in a single self.model.encode()
```python
# Conceptual pattern to integrate into your Pipeline.generate() loop:
# Whenever the FSM is in a deterministic structural state, encode and append
# the mandatory text directly without calling get_logits_from_input_ids():
deterministic_chunk = self.fsm.get_deterministic_chunk()
if deterministic_chunk is not None:
    chunk_ids = self.model.encode(deterministic_chunk)[0].tolist()
    TOKENS.extend(chunk_ids)
    # Advance the FSM state directly
    continue
```

 - *Bonus Optimization for `EXPECT_FUNCTION_NAME`:** When generating the function name in `EXPECT_FUNCTION_NAME`, check how many functions in `self.schemas` start with your current `accumulator`. As soon as **only 1 function** matches the prefix (for example, right after the model generates `"fn_sub`), you don't even need to call the LLM for the remaining 5 tokens of `stitute_string_with_regex"`! You can deterministically complete the rest of that function name, the `", "parameters": {`, and the first parameter key `"` + `first_key` + `":` in a single `self.model.encode()` call!
 
### Step 3: Replace Python `max()` and Remove Repetition Penalty in `_get_best_token_by_logits`
In `src/pipeline.py`:

1. Remove the `recent_tokens = set(context[-40:])` repetition penalty so the model isn't penalized for copying words/numbers from the user's prompt or reusing JSON quotes.
    
2. Convert `self.string_tokens_comma`, `self.string_tokens_brace`, `self.number_tokens_comma`, and `self.number_tokens_brace` into NumPy integer arrays (`np.array(..., dtype=np.int64)`) once in `model_post_init`.
    
3. In `_get_best_token_by_logits`, use NumPy fancy indexing to select the highest logit in C:
```python
# Conceptual snippet for _get_best_token_by_logits:
logits_arr = np.asarray(logits, dtype=np.float32)
allowed_arr = np.asarray(allowed, dtype=np.int64)
best_idx = int(np.argmax(logits_arr[allowed_arr]))
return int(allowed_arr[best_idx])
```

> [!note]
(Note: Indexing logits_arr[allowed_arr] directly is even faster than creating a np.full_like(logits_arr, -np.inf) array because it avoids allocating and filling a second 150,000-element array on every step!)

### Step 4: Save the Output JSON in `__main__.py`

Ensure `pipeline.generate()` returns a parsed dictionary (or parse the returned JSON string with `json.loads()`), append `{"prompt": prom.prompt, "name": ..., "parameters": ...}` to a results list in `__main__.py`, create parent directories with `Path(ARGS.output).parent.mkdir(parents=True, exist_ok=True)`, and write the list with `json.dump()`.

## Verification, Testing, and Troubleshooting
1. **Count Your Forward Passes Per Prompt:**
    - Add a simple counter `llm_calls = 0` inside `generate()` right before `self.model.get_logits_from_input_ids(TOKENS)`.
    - **Before deterministic bypassing:** Prompt 9 will show **~50–55** `llm_calls`.
    - **After deterministic bypassing:** Prompt 9 should drop to **~15–20** `llm_calls`, and simple prompts (like `Greet shrek`) should drop from **~18** `llm_calls` down to **2–4** `llm_calls`!
2. **Verify `Ġ` Removal and Regex Accuracy:**
    - Inspect your generated `data/output/function_calling_results.json` after running `make run`.
    - Check that `"source_string"` in Prompt 9 is `"Hello 34 I'm 233 years old"` (with normal ASCII spaces, no `Ġ`) and that removing the repetition penalty restores accurate regex generation in Prompts 9 and 10.
3. **Common Pitfalls When Encoding Deterministic Chunks:**
    - **Pitfall:** Calling `self.model.encode(chunk)` on small fragments like `'"'` and `'name'` separately can split tokens differently than encoding `'{"name": "'` as a single string.
    - **Fix:** Concatenate adjacent deterministic pieces into a single string (e.g., `'", "parameters": {"' + first_param_name + '": '`) before calling `self.model.encode()` once, and append the exact same string to your JSON output buffer so your output string and `TOKENS` stay 100% synchronized.