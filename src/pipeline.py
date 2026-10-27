# pipeline.py
from pydantic import BaseModel, ConfigDict
from typing import Any
from src.schemas import FunctionDef
from src.constrained_decode import VocabularyTrie
from src.loaders import load_vocab_lut
from src.fsm import StateMachine, JSONState
import numpy as np


class Pipeline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Any
    func_call: list[FunctionDef]
    lut: dict[int, str] = {}
    vocab: dict[str, int] = {}
    vocab_trie: Any = None
    fsm: Any = None
    output_path: str

    # Split caches by exit route (comma vs brace)
    string_tokens_comma: Any = []
    string_tokens_brace: Any = []
    number_tokens_comma: Any = []
    number_tokens_brace: Any = []
    quote_exit_tokens: set[int] = set()

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic's built-in way to run code immediately after initialization.
        Pre-computes token caches split by exit route (comma vs brace).
        """
        import string
        from pathlib import Path

        self.lut, self.vocab = load_vocab_lut(self.model)
        self.vocab_trie = VocabularyTrie()
        self.vocab_trie.build_from_lut(self.lut)
        self.fsm = StateMachine(schemas=self.func_call)
        Path(self.output_path).unlink(missing_ok=True)

        # --- STRING TOKEN CACHES (split by exit route) ---
        # For comma route: allow ", but NOT "}
        string_comma_candidates = []
        # For brace route: allow "}, but NOT ",
        string_brace_candidates = []

        for tid, text in self.lut.items():
            if any(c in text for c in ("\n", "\r", "\t", "\x00")):
                continue
            if '"' in text:
                clean = text.rstrip()
                if clean == '"':
                    string_comma_candidates.append(tid)
                    string_brace_candidates.append(tid)
                    self.quote_exit_tokens.add(tid)
                elif clean == '",':
                    string_comma_candidates.append(tid)
                    self.quote_exit_tokens.add(tid)
                elif clean == '"}':
                    string_brace_candidates.append(tid)
                    self.quote_exit_tokens.add(tid)
            else:
                # Non-quote tokens allowed in both
                string_comma_candidates.append(tid)
                string_brace_candidates.append(tid)

        self.string_tokens_comma = string_comma_candidates
        self.string_tokens_brace = string_brace_candidates

        # --- NUMBER TOKEN CACHES (split by exit route) ---
        number_charset = set(string.digits + " -.")
        raw_num_tokens = self.vocab_trie.search_by_charset(number_charset)

        # For comma route: allow all number chars + comma, but NOT standalone }
        comma_exit_tokens = self.vocab_trie.search_allowed_tokens([",", " ,", ",\n", " ,\n"])
        self.number_tokens_comma = raw_num_tokens + comma_exit_tokens

        # For brace route: allow all number chars + close brace, but NOT
        # standalone ,
        brace_exit_tokens = self.vocab_trie.search_allowed_tokens(["}", " }", "\n}", "\n }"])
        self.number_tokens_brace = raw_num_tokens + brace_exit_tokens

        
        self.string_tokens_comma = np.array(string_comma_candidates, dtype=np.int64)
        self.string_tokens_brace = np.array(string_brace_candidates, dtype=np.int64)

        self.number_tokens_comma = np.array(raw_num_tokens + comma_exit_tokens, dtype=np.int64)
        self.number_tokens_brace = np.array(raw_num_tokens + brace_exit_tokens, dtype=np.int64)

    def _get_best_token_by_logits(self, logits: list[float],
                                  allowed: list[int] | Any,
                                  context: list[int]) -> int:
        """Finds the allowed token with the highest logit score."""

        logits_np = np.asarray(logits, dtype=np.float32)

        # SELECTIVE PENALTY: Only apply inside strings, and exempt closing quotes
        if self.fsm.state == JSONState.EXPECT_STRING_CONTENT:
            penalty = 1.2
            recent_tokens = set(context[-40:]) - self.quote_exit_tokens
            for tid in recent_tokens:
                if logits_np[tid] > 0:
                    logits_np[tid] /= penalty
                else:
                    logits_np[tid] *= penalty

        # Fancy indexing in C: Extracts only the allowed logits and finds the max
        allowed_np = np.asarray(allowed, dtype=np.int64)
        best_idx = int(np.argmax(logits_np[allowed_np]))
        
        return int(allowed_np[best_idx])

    def _build_system_prompt(self, user: str) -> str:
        """
            Formats the available tool schemas and the user request into
            a single prompt.
        """
        fn_defs_lines: list[str] = []
        for fn in self.func_call:
            # Format parameters compactly: a: number, b: number
            param_parts = []
            for param_name, param_schema in fn.parameters.items():
                param_parts.append(f"{param_name}: {param_schema.type}")
            params_str = ", ".join(param_parts)

            # Format the function signature
            fn_defs_lines.append(f"- {fn.name}({params_str}):\
 {fn.description}")

        fn_defs_block = "\n".join(fn_defs_lines)
        # Truncated the hardcoded JSON so the FSM can generate the root brace
        return (
            "You are a function calling assistant.\n\n"
            "Available functions:\n"
            f"{fn_defs_block}\n\n"
            f"User request: {user}\n\n"
            "Output the correct function call as JSON:\n"
        )

    def generate(self, prompt: str) -> str:
        """Generate the function Calling for each prompt with validated cascade"""
        # resete the state to the beginning to seach for token "{"
        self.fsm.reset()
        system = self._build_system_prompt(prompt)
        tensor = self.model.encode(system)   # tokenizer prompt
        TOKENS = tensor[0].tolist()  # list of al the tokens (prompt)
        accumulator = ""  # str that will be in the output
        json_full_output = ""
        
        while self.fsm.state is not JSONState.DONE:
            # === 1. DETERMINISTIC BYPASS ===
            det_str = self.fsm.get_deterministic_string()
            if det_str is not None:
                chunk_ids = self.model.encode(det_str)[0].tolist()
                TOKENS.extend(chunk_ids)
                json_full_output += det_str
                print(det_str, end="", flush=True)

                self.fsm.update_state(det_str)
                accumulator = ""
                continue

            allowed = self.fsm.get_allowed_strings()

            is_open_ended = False
            if (self.fsm.state == JSONState.EXPECT_STRING_CONTENT or
                    (self.fsm.state == JSONState.EXPECT_PARAM_VALUE and
                     self.fsm.curr_param and
                     self.fsm.curr_param.type in ("number", "integer"))):
                is_open_ended = True

            if is_open_ended:
                # DYNAMIC CACHE ROUTING: Ask FSM if we need comma or brace
                missing_params = False
                if self.fsm.curr_func_call:
                    missing_params = (len(self.fsm.generated_params) <
                                     len(self.fsm.curr_func_call.parameters))

                # Route to the exact cache required to prevent illegal bundled tokens
                if self.fsm.state == JSONState.EXPECT_STRING_CONTENT:
                    # In string, the exit route is determined by whether params are missing
                    allowed_tokens = (self.string_tokens_comma 
                                     if missing_params 
                                     else self.string_tokens_brace)
                else:
                    # In number, the exit route is determined by whether params are missing
                    allowed_tokens = (self.number_tokens_comma 
                                     if missing_params 
                                     else self.number_tokens_brace)
            else:
                remainders = []
                for valid in allowed:
                    if valid.startswith(accumulator):
                        remainder = valid[len(accumulator):]
                        remainders.append(remainder)
                allowed_tokens = self.vocab_trie.search_allowed_tokens(
                    remainders)

            if len(allowed_tokens) == 0:
                raise ValueError("Model cannot generate the required strings")

            logits = self.model.get_logits_from_input_ids(TOKENS)
            next_token = self._get_best_token_by_logits(logits=logits,
                                                        allowed=allowed_tokens,
                                                        context=TOKENS)
            TOKENS.append(next_token)
            # Translate the token ID back to text to see what the model chose
            next_token_text = self.lut.get(next_token, "")
            accumulator += next_token_text
            json_full_output += next_token_text
            print(next_token_text, end="", flush=True)

            # === TRIGGER FSM TRANSITIONS WITH VALIDATED CASCADE ===
            if is_open_ended:
                # In open-ended states, transition when the exit character is generated
                if (self.fsm.state == JSONState.EXPECT_STRING_CONTENT
                        and '"' in accumulator):
                    # Find the quote position
                    exit_idx = accumulator.index('"')
                    # Transition on the quote
                    self.fsm.update_state('"')
                    
                    # Cascade the leftover character ONLY if the FSM allows it
                    after_quote = accumulator[exit_idx + 1:].strip()
                    if after_quote and after_quote in self.fsm.get_allowed_strings():
                        self.fsm.update_state(after_quote)
                    
                    accumulator = ""
                    
                elif (self.fsm.state == JSONState.EXPECT_PARAM_VALUE and
                      ("," in accumulator or "}" in accumulator)):
                    exit_char = "," if "," in accumulator else "}"
                    exit_idx = accumulator.index(exit_char)
                    # Transition on the exit character
                    self.fsm.update_state(exit_char)

                    # Cascade the leftover number exits ONLY if the FSM allows it
                    after_num = accumulator[exit_idx + 1:].strip()
                    if after_num and after_num in self.fsm.get_allowed_strings():
                        self.fsm.update_state(after_num)

                    accumulator = ""
            else:
                # In strict states, transition when the exact string is completed
                if accumulator in allowed:
                    self.fsm.update_state(accumulator)
                    accumulator = ""

        return json_full_output

    def json_file_loader(self, prompt_text: str, generated_str: str) -> None:
        """Stores a single generated prompt result into the JSON file."""
        import json
        from pathlib import Path

        file_path = Path(self.output_path)

        # Create parent directories safely without crashing
        # if they already exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # Attempt to parse the LLM output into a dictionary
        try:
            parsed_output = json.loads(generated_str)
        except json.JSONDecodeError:
            parsed_output = {}

        # Construct the exact object required by the assignment
        new_entry = {
            "prompt": prompt_text,
            "name": parsed_output.get("name", ""),
            "parameters": parsed_output.get("parameters", {})
        }
        data = []
        # If the file exists, read its current contents
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError:
                # If the existing file is corrupted, start fresh
                pass

        # Append the new prompt's result
        data.append(new_entry)

        # Overwrite the file with the updated array
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
