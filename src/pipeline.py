# pipeline.py
from pydantic import BaseModel, ConfigDict
from typing import Any
from src.schemas import FunctionDef
from src.constrained_decode import VocabularyTrie
from src.loaders import load_vocab_lut
from src.fsm import StateMachine, JSONState


class Pipeline(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Any
    func_call: list[FunctionDef]
    lut: dict[int, str] = {}
    vocab: dict[str, int] = {}
    vocab_trie: Any = None
    fsm: Any = None

    # Split caches by exit route (comma vs brace)
    string_tokens_comma: list[int] = []
    string_tokens_brace: list[int] = []
    number_tokens_comma: list[int] = []
    number_tokens_brace: list[int] = []

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic's built-in way to run code immediately after initialization.
        Pre-computes token caches split by exit route (comma vs brace).
        """
        self.lut, self.vocab = load_vocab_lut(self.model)
        self.vocab_trie = VocabularyTrie()
        self.vocab_trie.build_from_lut(self.vocab)
        self.fsm = StateMachine(schemas=self.func_call)
        
        import string
        
        # --- STRING TOKEN CACHES (split by exit route) ---
        # For comma route: allow ", but NOT "}
        string_comma_candidates = []
        # For brace route: allow "}, but NOT ",
        string_brace_candidates = []
        
        for tid, text in self.lut.items():
            if '"' in text:
                clean = text.strip(" Ġ\u2581\n\t")
                if clean == '"':
                    # Pure quote allowed in both routes
                    string_comma_candidates.append(tid)
                    string_brace_candidates.append(tid)
                elif clean == '",':
                    # Bundled quote+comma only in comma route
                    string_comma_candidates.append(tid)
                elif clean == '"}':
                    # Bundled quote+brace only in brace route
                    string_brace_candidates.append(tid)
            else:
                # Non-quote tokens allowed in both
                string_comma_candidates.append(tid)
                string_brace_candidates.append(tid)
        
        self.string_tokens_comma = string_comma_candidates
        self.string_tokens_brace = string_brace_candidates
        
        # --- NUMBER TOKEN CACHES (split by exit route) ---
        number_charset = set(string.digits + " -.,}Ġ\u2581\n")
        raw_num_tokens = self.vocab_trie.search_by_charset(number_charset)
        
        # For comma route: allow all number chars + comma, but NOT standalone }
        comma_exit_tokens = self.vocab_trie.search_allowed_tokens([",", " ,", ",\n", " ,\n"])
        self.number_tokens_comma = raw_num_tokens + comma_exit_tokens
        
        # For brace route: allow all number chars + close brace, but NOT standalone ,
        brace_exit_tokens = self.vocab_trie.search_allowed_tokens(["}", " }", "\n}", "\n }"])
        self.number_tokens_brace = raw_num_tokens + brace_exit_tokens

    def _get_best_token_by_logits(self, logits: list[float],
                                  allowed: list[int],
                                  context: list[int]) -> int:
        """
        Finds the allowed token with the highest logit score,
        applying a repetition penalty."""
        penalty = 1.2  # Standard penalty multiplier

        # Grab the last 40 tokens to check for loops
        recent_tokens = set(context[-40:])

        for tid in recent_tokens:
            # Artificially suppress the score of recently used tokens
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty

        # Python's optimized built-in max function
        return max(allowed, key=lambda tid: logits[tid])

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
        tensor_id_lenght = len(TOKENS)
        accumulator = ""  # str that will be in the output
        
        while self.fsm.state is not JSONState.DONE:
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

            if not allowed_tokens:
                raise ValueError("Model cannot generate the required strings")

            logits = self.model.get_logits_from_input_ids(TOKENS)
            next_token = self._get_best_token_by_logits(logits=logits,
                                                        allowed=allowed_tokens,
                                                        context=TOKENS)
            TOKENS.append(next_token)
            # Translate the token ID back to text to see what the model chose
            next_token_text = self.lut.get(next_token, "")
            accumulator += next_token_text
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
                    after_quote = accumulator[exit_idx + 1:].strip(" \n\tĠ\u2581")
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
                    after_num = accumulator[exit_idx + 1:].strip(" \n\tĠ\u2581")
                    if after_num and after_num in self.fsm.get_allowed_strings():
                        self.fsm.update_state(after_num)
                    
                    accumulator = ""
            else:
                # In strict states, transition when the exact string is completed
                if accumulator in allowed:
                    self.fsm.update_state(accumulator)
                    accumulator = ""
                    
        gen_tokens = TOKENS[tensor_id_lenght:]
        return self.model.decode(gen_tokens)
