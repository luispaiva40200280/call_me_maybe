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

    # NEW: Cache lists to store our heavy calculations
    string_allowed_tokens: list[int] = []
    number_allowed_tokens: list[int] = []

    def model_post_init(self, __context: Any) -> None:
        """
        Pydantic's built-in way to run code immediately after initialization
        """
        self.lut, self.vocab = load_vocab_lut(self.model)
        self.vocab_trie = VocabularyTrie()
        self.vocab_trie.build_from_lut(self.vocab)
        self.fsm = StateMachine(schemas=self.func_call)
        # --- PRE-COMPUTE OPEN-ENDED TOKENS (Runs only once!) ---
        import string
        # 1. Cache String Tokens
        for tid, text in self.lut.items():
            if '"' in text:
                clean = text.strip(" Ġ\u2581\n\t")
                # FIX: Allow the pure quote AND the standard JSON exit tokens
                if clean in ('"', '",', '"}'):
                    self.string_allowed_tokens.append(tid)
            else:
                self.string_allowed_tokens.append(tid)
        # 2. Cache Number Tokens
        number_charset = set(string.digits + " -.,}Ġ\u2581\n")
        raw_num_tokens = self.vocab_trie.search_by_charset(number_charset)
        for tid in raw_num_tokens:
            # FIX: We no longer need to ban number bleed,
            # the charset protects us
            self.number_allowed_tokens.append(tid)

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
        """Generate the function Calling for each prompt"""
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
                # STANT LOOKUP: No loops, no DFS, no stripping!
                if self.fsm.state == JSONState.EXPECT_STRING_CONTENT:
                    allowed_tokens = self.string_allowed_tokens
                else:
                    allowed_tokens = self.number_allowed_tokens
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
            # 3. Trigger FSM Transitions based on the state type
            if is_open_ended:
                # In open-ended states, transition when the exit character is
                # generated
                if (self.fsm.state == JSONState.EXPECT_STRING_CONTENT
                        and '"' in accumulator):
                    # 1. Look to see if a comma or brace bled past the quote
                    idx = accumulator.index('"')
                    after_quote = accumulator[idx + 1:].strip(" \n\tĠ\u2581")
                    # 2. Close the string state
                    self.fsm.update_state('"')
                    # 3. If a comma or brace bled through, fast-forward
                    # the FSM!
                    if after_quote in (',', '}'):
                        self.fsm.update_state(after_quote)
                    accumulator = ""
                elif (self.fsm.state == JSONState.EXPECT_PARAM_VALUE and
                      ("," in accumulator or "}" in accumulator)):
                    exit_char = "," if "," in accumulator else "}"
                    self.fsm.update_state(exit_char)
                    accumulator = ""
            else:
                # In strict states, transition when the exact string
                # is completed
                if accumulator in allowed:
                    self.fsm.update_state(accumulator)
                    accumulator = ""
        gen_tokens = TOKENS[tensor_id_lenght:]
        return self.model.decode(gen_tokens)
