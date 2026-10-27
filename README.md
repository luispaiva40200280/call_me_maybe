*This project has been created as part of the 42 curriculum by luispaiva40200280.*

# Call Me Maybe

## Description

`Call Me Maybe` is a small function-calling system built on top of a local language model. The project demonstrates how an LLM can be constrained to generate valid JSON tool calls instead of free-form text, following an explicit function schema.

The program loads a catalog of available functions, builds a system prompt describing those tools, and then decodes a model output while restricting every next token to the set of legal continuations for the current grammar state. In practice, this turns a generic decoder into a structured function-calling engine capable of producing well-formed calls such as:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

The repository focuses on a minimal pipeline for:
- validating available function definitions,
- prompting a small local model,
- constraining generation with a finite state machine,
- filtering the token space with a vocabulary trie,
- and ensuring the final output matches the requested function signature.

## Instructions

### Requirements

- Python 3.10+
- `uv` package manager
- A working internet connection for the first model download

### Installation

```bash
make install
```

This creates a virtual environment and installs the required dependencies, including:
- `pydantic`
- `numpy`
- `flake8`
- the local `llm_sdk` package

### Execution

Run the project with the bundled sample inputs:

```bash
make run
```

Or directly:

```bash
uv run python -m src \
  --functions_definition ./data/input/functions_definition.json \
  --input ./data/input/function_calling_tests.json \
  --output ./data/output/function_calls.json
```

### Useful maintenance commands

```bash
make clean
make lint
```

## Algorithm explanation

The constrained decoding approach is the core of the project.

1. Function schema loading
   - The repository reads a JSON file describing callable functions and their parameter types.
   - Each function is validated with Pydantic models (`FunctionDef`, `ParameterSchema`, and `PromptSchema`).
   - This guarantees that only known functions and allowed parameter names/types are used.

2. Prompt construction
   - The system prompt includes the user request and a list of available tools.
   - Example format:
     - `fn_add_numbers(a: number, b: number): Add two numbers together and return their sum.`
   - This tells the model which function names and arguments are valid before generation begins.

3. Finite-state grammar control
   - The `StateMachine` in `src/fsm.py` tracks the exact stage of JSON generation.
   - It enforces a legal sequence such as:
     - `{`
     - `"name"`
     - `:`
     - valid function name
     - `,`
     - `"parameters"`
     - `:`
     - `{`
     - parameter keys in the allowed order
     - values of the correct type
     - closing braces and commas
   - This prevents invalid output such as unknown keys, wrong braces, or missing commas.

4. Token-level filtering
   - The model outputs logits for the next token.
   - The `VocabularyTrie` in `src/constrained_decode.py` indexes the model vocabulary and allows the decoder to quickly query which token IDs are valid for the current grammar state.
   - The pipeline then chooses the highest-scoring legal token, while applying a repetition penalty to avoid looping on the same recent tokens.

5. Route-aware numeric and string decoding
   - In the `Pipeline.generate()` method, the system distinguishes between:
     - open-ended states such as string content or numeric values,
     - and strictly structured states like braces, names, and JSON punctuation.
   - It also splits allowed tokens based on whether the next legal exit is a comma or a closing brace, avoiding impossible token combinations such as a numeric token that incorrectly closes the object before all required fields are emitted.

6. Result validation
   - After the model finishes generating, the output is constrained by the same schema that started generation.
   - If the model tries to produce a bad function name or an invalid parameter, the state machine rejects it before the token is accepted.

This method is a lightweight example of grammar-constrained decoding: instead of letting the model invent arbitrary JSON, the decode loop only explores tokens that preserve validity.

## Design decisions

Several implementation choices were made to keep the project simple, fast, and understandable.

- Pydantic validation
  - The schemas used in `src/schemas.py` define the accepted function signatures.
  - This keeps the input data safe and makes debugging easier.

- FSM instead of plain regex
  - A regular expression alone would be fragile for nested JSON and parameter types.
  - The finite-state machine makes the generation logic explicit and easy to reason about.

- Trie-based vocabulary pruning
  - Looking up valid tokens through a trie is much faster than scanning the full vocabulary for every step.
  - This matters because constrained decoding must happen repeatedly during generation.

- Local model strategy
  - The project uses `llm_sdk`, a lightweight wrapper around Hugging Face models.
  - This keeps the solution portable and avoids relying on an external API service.

- Minimal yet explicit architecture
  - The repository is intentionally split into small modules: `loaders`, `schemas`, `fsm`, `constrained_decode`, and `pipeline`.
  - This separation makes feature changes and debugging straightforward.

## Performance analysis

### Accuracy

The constrained decoder substantially improves structural validity. The system is designed to prevent common syntax mistakes such as invalid JSON, wrong parameter names, or illegal closing braces. This is particularly useful for function-calling tasks because the final output must match a strict schema.

However, the model still has to choose the correct function and parameter values from the user request. The system is therefore strongest for short, well-structured prompts and a small set of available functions. Accuracy drops when prompts become ambiguous or when the model must infer a more complex transformation than the function catalogue explicitly supports.

### Speed

Because generation is token-by-token and constrained at every step, the system is more expensive than a standard unconstrained generation loop. The project compensates by:
- limiting the search space with a trie,
- avoiding expensive full-vocabulary scans,
- and using a compact local model.

The first run can be slower because the model and tokenizer must be downloaded and loaded. After that, repeated prompt execution is much faster, especially on CUDA-capable or MPS-capable hardware.

### Reliability

The implementation is reliable for grammar compliance, thanks to the FSM and token restrictions. It handles common structured-output problems better than a naive decoder, and it adds repetition penalties to avoid loops or repetitive token patterns.

The remaining weak point is semantic reliability: the model may still select the wrong function or produce a parameter value that is syntactically valid but not the best answer to the user prompt. For this reason, the system is best treated as a structured decoding layer around a language model rather than a fully independent planner.

## Challenges faced

One of the biggest problems in this project was controlling generation without breaking valid model behavior.

- The model often produced token sequences that were nearly valid, but with the wrong closing separator or wrong parameter routing.
- Strings and numbers are open-ended states: the model may continue generating characters, then later decide to exit with a comma or a closing brace.
- The vocabulary contains many tokens that appear visually similar but are not equivalent in context, such as quote/comma or quote/brace token bundles.

To solve this, the implementation used a combination of:
- state tracking for the exact JSON grammar,
- route-aware token caches for comma vs. brace exits,
- repetition penalties to prevent loops,
- and strict validation of function names and parameter names before accepting a transition.

This made the decoder resilient enough to keep the output valid while still allowing natural model generation.

## Testing strategy

The project includes a small evaluation dataset under `data/input/`:
- `functions_definition.json` defines the available callable tools,
- `function_calling_tests.json` contains prompts for arithmetic, greetings, reverse-string operations, square-root calculations, and string substitution.

The validation process is based on:
- JSON parsing of the inputs,
- Pydantic schema validation,
- and execution of the generation pipeline on each user prompt.

The CLI prints each prompt and measures the time needed to process it. This makes it easy to identify both semantic errors and performance regressions. The expected behavior is that every generated call follows the allowed schema, and each prompt results in a valid function invocation.

## Example usage

### Example 1: arithmetic

Input prompt:

```text
What is the sum of 2 and 3?
```

Expected output shape:

```json
{
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2,
    "b": 3
  }
}
```

### Example 2: greeting

Input prompt:

```text
Greet shrek
```

Expected output shape:

```json
{
  "name": "fn_greet",
  "parameters": {
    "name": "shrek"
  }
}
```

### Example 3: text transformation

Input prompt:

```text
Reverse the string 'hello'
```

Expected output shape:

```json
{
  "name": "fn_reverse_string",
  "parameters": {
    "s": "hello"
  }
}
```

## Resources

### References

- Hugging Face Transformers documentation: https://huggingface.co/docs/transformers/
- Pydantic documentation: https://docs.pydantic.dev/
- Hugging Face tokenizers and vocabularies: https://huggingface.co/docs/tokenizers/
- Introduction to constrained decoding and structured generation in LLMs: general reading on grammar-constrained generation and structured outputs.

### AI usage

AI tools were used as an assistant during the development of this project, specifically for:
- brainstorming the architecture of the constrained generation loop,
- helping debug tokenizer and vocabulary edge cases,
- refining the finite-state machine logic for valid JSON generation,
- improving code readability and structure in the Python modules,
- drafting and polishing the README documentation.

The AI was used mainly to accelerate implementation and debugging, while the final design decisions, validation logic, and technical explanations were reviewed and adapted to fit the project’s actual codebase.

## Repository structure

```text
.
├── data/
│   └── input/
│       ├── function_calling_tests.json
│       └── functions_definition.json
├── llm_sdk/
│   └── llm_sdk/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── constrained_decode.py
│   ├── fsm.py
│   ├── loaders.py
│   ├── pipeline.py
│   ├── schemas.py
│   └── test.py
├── Makefile
├── README.md
├── call_me_maybe.pdf
├── pseudocode.txt
└── tokenization.py
```

This project demonstrates a compact but practical example of structured decoding for tool calling with a local model, highlighting both the strengths and the constraints of grammar-controlled generation.
