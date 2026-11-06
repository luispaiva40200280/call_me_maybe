# pipeline.py
"""
Future 
"""
from pydantic import BaseModel, ConfigDict
from typing import Any
from src.decoding import FunctionDef
from src.decoding.constrained_decode import VocabularyTrie
from src.data_loader.loaders import load_vocab_lut
from src.engine.fsm import StateMachine, JSONState
import numpy as np
import logging
from .. import ft_time

logger = logging.getLogger(__name__)


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
    # Same as above, but excluding tokens whose text starts with a space --
    # used only for the FIRST token of a string value, so the model can't
    # pick a space-prefixed token (e.g. " *") as the opening character.
    string_tokens_comma_first: Any = []
    string_tokens_brace_first: Any = []
    number_tokens_comma: Any = []
    number_tokens_brace: Any = []
    # Same shape as number_tokens_comma/brace, but built from a charset
    # with no '.' -- used for "integer"-typed parameters.
    integer_tokens_comma: Any = []
    integer_tokens_brace: Any = []
    # number_tokens_comma/brace with the exit tokens stripped out -- used
    # to force continued digit/dot generation until a "number" value has
    # a valid decimal fraction.
    number_tokens_no_exit: Any = []
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

        # --- STRING TOKEN CACHES (split by exit route, and by position) ---
        # For comma route: allow ", but NOT "}
        string_comma_candidates = []
        # For brace route: allow "}, but NOT ",
        string_brace_candidates = []
        # Same two, but excluding any token whose text starts with a
        # space -- only valid as the very FIRST token of a string value.
        string_comma_candidates_first = []
        string_brace_candidates_first = []

        # Curly/smart quotes are visually a "closing quote" to the model
        # but are different Unicode characters from the literal '"' the
        # FSM's EXPECT_STRING_CONTENT exit check looks for. If one of
        # these gets picked as content, the FSM can never detect the
        # value closing and generation runs away. Banning them from
        # content outright is cheap insurance on top of the
        # max_value_tokens forced-closure cap below.
        # Define what constitutes a safe ASCII printable character for JSON strings
        # (excluding control characters and backslashes if you want to keep it ultra-clean)
        safe_ascii = set(string.ascii_letters + string.digits + string.punctuation + " \t")

        for tid, text in self.lut.items():
            if not all(c in safe_ascii for c in text):
                continue

            if any(c in text for c in ("\n", "\r", "\x00")):
                continue
            # Ensure every character in the token text is a standard ASCII printable character
            if not all(ord(c) < 128 and c in string.printable for c in text):
                continue
            if any(c in text for c in ("\n", "\r", "\x00")):
                continue

            if '"' in text:
                clean = text.rstrip()
                if clean == '"':
                    string_comma_candidates.append(tid)
                    string_brace_candidates.append(tid)
                    string_comma_candidates_first.append(tid)
                    string_brace_candidates_first.append(tid)
                    self.quote_exit_tokens.add(tid)
                elif clean == '",':
                    string_comma_candidates.append(tid)
                    string_comma_candidates_first.append(tid)
                    self.quote_exit_tokens.add(tid)
                elif clean == '"}':
                    string_brace_candidates.append(tid)
                    string_brace_candidates_first.append(tid)
                    self.quote_exit_tokens.add(tid)
            else:
                string_comma_candidates.append(tid)
                string_brace_candidates.append(tid)
                if not text.startswith(" "):
                    string_comma_candidates_first.append(tid)
                    string_brace_candidates_first.append(tid)

        # --- NUMBER TOKEN CACHES (split by exit route AND by type) ---
        # "number" may contain a '.'; "integer" never may -- so these get
        # two different charsets instead of sharing one.
        number_charset = set(string.digits + " -.")
        integer_charset = set(string.digits + " -")
        raw_num_tokens = self.vocab_trie.search_by_charset(number_charset)
        raw_int_tokens = self.vocab_trie.search_by_charset(integer_charset)

        # For comma route: allow all number chars + comma, but NOT standalone }
        comma_exit_tokens = self.vocab_trie.search_allowed_tokens(
            [",", " ,", ",\n", " ,\n"])

        # For brace route: allow all number chars + close brace, but NOT
        # standalone ,
        brace_exit_tokens = self.vocab_trie.search_allowed_tokens(
            ["}", " }", "\n}", "\n }"])

        self.string_tokens_comma = np.array(string_comma_candidates,
                                            dtype=np.int64)
        self.string_tokens_brace = np.array(string_brace_candidates,
                                            dtype=np.int64)
        self.string_tokens_comma_first = np.array(
            string_comma_candidates_first, dtype=np.int64)
        self.string_tokens_brace_first = np.array(
            string_brace_candidates_first, dtype=np.int64)

        self.number_tokens_comma = np.array(raw_num_tokens + comma_exit_tokens,
                                            dtype=np.int64)
        self.number_tokens_brace = np.array(raw_num_tokens + brace_exit_tokens,
                                            dtype=np.int64)
        # No exit tokens at all -- used to force continued digit/dot
        # generation on a "number" value until it has a valid fraction.
        self.number_tokens_no_exit = np.array(raw_num_tokens, dtype=np.int64)

        self.integer_tokens_comma = np.array(raw_int_tokens + comma_exit_tokens,
                                             dtype=np.int64)
        self.integer_tokens_brace = np.array(raw_int_tokens + brace_exit_tokens,
                                             dtype=np.int64)

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

    def _get_best_token_by_logits(self, logits: list[float],
                                  allowed: list[int] | Any,
                                  value_tokens: list[int]) -> int:
        """
        Finds the allowed token with the highest logit score.

        `value_tokens` must contain ONLY the tokens generated so far for
        the CURRENT string value (reset every time the FSM transitions
        out of a value) -- never the full prompt/output history. Using
        the full context let tokens from the system prompt (including the
        user's own request text, e.g. digits or letters the user typed)
        get penalized before the model had written a single character of
        the value, which pushed it away from the exact characters it
        needed to reuse (digits in "[0-9]", repeated vowels in
        "[aeiouAEIOU]", each "|" in an alternation, etc.).
        """

        logits_np = np.asarray(logits, dtype=np.float32)

        # SELECTIVE PENALTY: only apply inside strings, only once a token
        # has genuinely started looping within THIS value (not on first
        # reuse), and exempt the tokens that close the string.
        if self.fsm.curr_state == JSONState.EXPECT_STRING_CONTENT and value_tokens:
            penalty = 1.2
            loop_threshold = 3
            counts: dict[int, int] = {}
            for tid in value_tokens:
                counts[tid] = counts.get(tid, 0) + 1

            for tid, count in counts.items():
                if count < loop_threshold or tid in self.quote_exit_tokens:
                    continue
                if logits_np[tid] > 0:
                    logits_np[tid] /= penalty
                else:
                    logits_np[tid] *= penalty

        # Fancy indexing in C: Extracts only the allowed logits and finds the max
        allowed_np = np.asarray(allowed, dtype=np.int64)
        best_idx = int(np.argmax(logits_np[allowed_np]))

        return int(allowed_np[best_idx])

    @ft_time
    def generate(self, prompt: str, max_tokens: int = 50,
                 max_value_tokens: int = 12) -> str:
        """
        Generate the function Calling for each prompt with validated cascade

        max_tokens: hard cap on model-sampled tokens for the whole call.
            Safety net for the FSM getting stuck mid-string (e.g. the
            model writes a curly "smart" quote instead of a literal '"',
            which the FSM's exit check can never match, turning
            EXPECT_STRING_CONTENT into effectively unconstrained
            generation forever).
        max_value_tokens: hard cap on tokens for a single string value.
            Once hit, token selection is forced down to only the tokens
            that close the string, so a stuck value still ends up valid
            JSON instead of running away.
        """
        # resete the state to the beginning to seach for token "{"
        self.fsm.reset()
        system = self._build_system_prompt(prompt)
        tensor = self.model.encode(system)   # tokenizer prompt
        TOKENS = tensor[0].tolist()  # list of al the tokens (prompt)
        accumulator = ""  # str that will be in the output
        json_full_output = ""
        # Tokens generated for the value currently in progress only (reset
        # on every FSM transition) -- used for the repetition penalty so it
        # never sees tokens from the system prompt or earlier values.
        value_tokens: list[int] = []
        steps = 0

        while self.fsm.curr_state is not JSONState.DONE:
            # === 1. DETERMINISTIC BYPASS ===
            det_str = self.fsm.get_deterministic_string()
            if det_str is not None:
                chunk_ids = self.model.encode(det_str)[0].tolist()
                TOKENS.extend(chunk_ids)
                json_full_output += det_str
                print(det_str, end="", flush=True)

                self.fsm.update_state(det_str)
                accumulator = ""
                value_tokens = []
                continue

            steps += 1
            if steps > max_tokens:
                raise ValueError(
                    f"Exceeded max_tokens={max_tokens} without completing "
                    f"a valid function call (stuck in state {self.fsm.curr_state})"
                )

            allowed = self.fsm.get_allowed_strings()

            is_open_ended = False
            if (self.fsm.curr_state == JSONState.EXPECT_STRING_CONTENT or
                    (self.fsm.curr_state == JSONState.EXPECT_PARAM_VALUE and
                     self.fsm.curr_single_param and
                     self.fsm.curr_single_param.type in ("number", "integer"))):
                is_open_ended = True

            if is_open_ended:
                # DYNAMIC CACHE ROUTING: Ask FSM if we need comma or brace
                missing_params = False
                if self.fsm.curr_func_calling:
                    missing_params = (len(self.fsm.generated_params) <
                                     len(self.fsm.curr_func_calling.parameters))

                # Route to the exact cache required to prevent illegal bundled tokens
                if self.fsm.curr_state == JSONState.EXPECT_STRING_CONTENT:
                    # In string, the exit route is determined by whether params are missing
                    is_first_token = len(value_tokens) == 0
                    if is_first_token:
                        # Exclude space-prefixed tokens for the opening
                        # character only -- this is what stops values
                        # like replacement coming back as " *" instead
                        # of "*". Later characters still allow spaces
                        # normally (e.g. "Hello world").
                        allowed_tokens = (self.string_tokens_comma_first
                                         if missing_params
                                         else self.string_tokens_brace_first)
                    else:
                        allowed_tokens = (self.string_tokens_comma 
                                         if missing_params 
                                         else self.string_tokens_brace)

                    if len(value_tokens) >= max_value_tokens:
                        # Stuck -- force closure instead of letting it
                        # keep running. If the model never produces the
                        # literal '"' (e.g. it reaches for a curly smart
                        # quote instead), this is the only thing that
                        # stops the string from growing forever.
                        forced = [t for t in allowed_tokens
                                 if t in self.quote_exit_tokens]
                        if forced:
                            allowed_tokens = np.array(forced, dtype=np.int64)
                elif (self.fsm.curr_single_param is not None
                        and self.fsm.curr_single_param.type == "integer"):
                    # Integers never get a '.' in their allowed set at
                    # all, so there's nothing to force here.
                    allowed_tokens = (self.integer_tokens_comma
                                     if missing_params
                                     else self.integer_tokens_brace)
                elif self._has_valid_decimal(accumulator):
                    # Already has a '.' followed by a digit -- a "number"
                    # value is now free to close.
                    allowed_tokens = (self.number_tokens_comma 
                                     if missing_params 
                                     else self.number_tokens_brace)
                else:
                    # No valid fraction yet -- strip the exit tokens so
                    # the model MUST keep writing digits/'.' until the
                    # value is a real float, instead of being free to
                    # close on a bare integer-looking string.
                    allowed_tokens = self.number_tokens_no_exit
                    if len(allowed_tokens) == 0:
                        # Shouldn't happen, but never let the decimal
                        # enforcement itself crash the batch -- fall back
                        # to allowing exit; _coerce_parameter_types()
                        # still guarantees a proper float downstream.
                        logger.debug(
                            "number_tokens_no_exit was empty; falling "
                            "back to the normal exit-capable cache"
                        )
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
                                                        value_tokens=value_tokens)
            TOKENS.append(next_token)
            value_tokens.append(next_token)
            # Translate the token ID back to text to see what the model chose
            next_token_text = self.lut.get(next_token, "")
            accumulator += next_token_text
            json_full_output += next_token_text
            print(next_token_text, end="", flush=True)

            # === TRIGGER FSM TRANSITIONS WITH VALIDATED CASCADE ===
            if is_open_ended:
                # In open-ended states, transition when the exit character is generated
                if (self.fsm.curr_state == JSONState.EXPECT_STRING_CONTENT
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
                    value_tokens = []
                    
                elif (self.fsm.curr_state == JSONState.EXPECT_PARAM_VALUE and
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
                    value_tokens = []
            else:
                # In strict states, transition when the exact string is completed
                if accumulator in allowed:
                    self.fsm.update_state(accumulator)
                    accumulator = ""
                    value_tokens = []

        return json_full_output

    def _has_valid_decimal(self, accumulator: str) -> bool:
        """
        Whether `accumulator` (the text generated so far for the current
        number value) already contains a '.' followed by at least one
        digit -- i.e. a syntactically valid JSON fraction. "2." alone is
        NOT valid JSON (the fraction needs a digit after the dot), so
        that case correctly still returns False here.
        """
        dot_idx = accumulator.rfind(".")
        if dot_idx == -1:
            return False
        return any(c.isdigit() for c in accumulator[dot_idx + 1:])

    def _coerce_parameter_types(
        self, func_name: str, parameters: dict
    ) -> dict:
        """
        Casts each parameter value to the type declared in its function's
        schema -- in particular, ensures "number" params always come out
        as a Python float (so json.dump writes "2.0", not "2"), even if
        the model happened to generate a value with no decimal point.
        "integer" params are cast the same way for symmetry.

        This is enforced here, once, on the way to the output file,
        rather than by trying to force the model to always *choose* to
        write a "." during generation -- that depends on greedy decoding
        behaving a particular way every single time, where a cast here
        is a hard guarantee regardless of what digits were generated.
        """
        func_def = next(
            (fn for fn in self.func_call if fn.name == func_name), None)
        if func_def is None:
            return parameters

        coerced = dict(parameters)
        for key, value in parameters.items():
            schema = func_def.parameters.get(key)
            if schema is None:
                continue
            try:
                if schema.type == "number":
                    coerced[key] = float(value)
                elif schema.type == "integer":
                    coerced[key] = int(value)
            except (TypeError, ValueError):
                # Leave the original value rather than crash the batch
                # over one parameter that couldn't be cleanly cast.
                logger.debug(
                    "Could not coerce %s=%r to %s for %s",
                    key, value, schema.type, func_name,
                )
        return coerced

    def json_file_loader(self, prompt_text: str, generated_str: str) -> None:
        """
        Stores a single generated prompt result into the JSON file.
        """
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

        name = parsed_output.get("name", "")
        parameters = self._coerce_parameter_types(
            name, parsed_output.get("parameters", {}))

        # Construct the exact object required by the assignment
        new_entry = {
            "prompt": prompt_text,
            "name": name,
            "parameters": parameters
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
