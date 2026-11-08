from ..import ft_time
from .import JSONState
from ..decoding import FunctionDef, ParameterSchema
from .fsm import StateMachine
from pydantic import BaseModel, ConfigDict, Field
from typing import Any

_EXIT_CHARS = (",", "}")


class Generator(BaseModel):
    """
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Any
    fsm: StateMachine
    func_call: list[FunctionDef]
    max_tokens: int = 15  # max tokens for gen text
    max_value_tokens: int = 24  # max tokens for gen numbers
    lut: dict[int, str] = Field(default_factory=dict)
    vocab_trie: Any = None
    max_tokens: int = 150
    safe_chars: set[str] = Field(default_factory=set)
    content_ids: list[int] = Field(default_factory=list)
    quote_ids: list[int] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """
        """
        from ..data_loader.loaders import load_vocab_lut
        from ..decoding.constrained_decode import VocabularyTrie
        import string
        self.lut, _ = load_vocab_lut(self.model)
        self.vocab_trie = VocabularyTrie()
        self.vocab_trie.build_from_lut(self.lut)

        self.safe_chars = set(string.ascii_letters + string.digits
                              + string.punctuation + " ")
        self.safe_chars.discard('"')
        self.safe_chars.discard("\\")
        self.content_ids = self.vocab_trie.search_by_charset(self.safe_chars)

        # Only permit a token whose text is exactly the closing quote.
        self.quote_ids = [
            token_id for token_id, text in self.lut.items()
            if text == '"'
        ]

    def _build_system_prompt(self, user: str) -> str:
        """
        Formats the available tool schemas and the user request into
        a single prompt.
        Args:
            prompt (str): single user request
        Returns:
            system-prompt (str): Complete prompt with all function availeble
                and instructions
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

    def _encode_prompt(self, prompt: str) -> list[int]:
        """
        Encoding the system prompt and rests the genarator for each
        individual user request
        Args:
            prompt: the raw user request
        Returns:
            the initial token ids (system prompt encoded), ready to extend
        as generation proceeds
        """
        self.fsm.reset()  # resete the state to the beginning
        system = self._build_system_prompt(prompt)
        tensor = self.model.encode(system)  # list all tokens in the prompt
        return tensor[0].tolist()

    def _commit(self, tokens: list[int], text: str) -> str:
        """
        Applies a resolved piece of text: extends the running token list,
        advances the FSM, and hands the text back so the caller can append
        it to buffer. Shared by all three resolution paths (deterministic
        bypass, closed-set model choice, open-ended value) so the "turn
        text into tokens + tell the FSM" logic exists in exactly one place.

        Args:
            text: the resolved text for this step (bypass string, chosen
                literal, or a sampled token's decoded text).
            tokens: the running token-id list for this generate() call;
                mutated in place via extend().

        Returns:
            The same `text`, so the call site can do
            `buffer += self._commit(text, tokens)` in one line.
        """
        tokens.extend(self.model.encode(text)[0].tolist())
        self.fsm.update_state(gen_str=text)
        return text

    def _allowed_token_ids(self, strings: list[str],
                           accumulator: str) -> list[int]:
        remainders = [
            choice[len(accumulator):]
            for choice in strings
            if choice.startswith(accumulator)
            and len(choice) > len(accumulator)
        ]
        if not remainders:
            return []

        return self.vocab_trie.search_allowed_tokens(remainders)

    def _get_best_token(self, tokens: list[int],
                        allowed_ids: list[int]) -> int:
        """
        Finds the allowed token with the highest logit score.
        """
        import numpy as np
        logits = np.asarray(
            self.model.get_logits_from_input_ids(tokens),
            dtype=np.float32,
        )
        allowed = np.asarray(allowed_ids, dtype=np.int64)
        return int(allowed[np.argmax(logits[allowed])])

    def _genarate_compl_choice(self, tokens: list[int],
                               allowed: list[str]) -> tuple[str, int]:
        """Generate one complete choice, such as a function name."""

        accumulator = ""
        generated_text = ""
        for steps in range(self.max_tokens):
            allowed_ids = self._allowed_token_ids(strings=allowed,
                                                  accumulator=accumulator)
            if not allowed_ids:
                raise ValueError(
                    f"No token can continue {accumulator!r} in "
                    f"{self.fsm.curr_state}"
                )
            token_id = self._get_best_token(tokens, allowed_ids)
            tokens.append(token_id)
            token_text = self.lut.get(token_id)
            if token_text is None:
                raise ValueError(f"Token ID {token_id} missing from LUT")
            print(token_text, flush=True, end="")
            accumulator += token_text
            generated_text += token_text

            if accumulator in allowed:
                self.fsm.update_state(accumulator)
                return generated_text, steps

        raise ValueError(
            f"Exceeded token limit while generating a choice in "
            f"{self.fsm.curr_state}"
        )

    def _has_valid_decimal(self, accumulator: str) -> bool:
        """
        Checks if `accumulator` (the text generated so far for the current
        number value) is a valid float type, i.e: 2.0 or 2.8
        Args:
            accumulator (str): numeric value to check if is a float
        Returns:
            bool :
        """
        dot_idx = accumulator.rfind(".")
        if dot_idx == -1:
            return False
        return any(c.isdigit() for c in accumulator[dot_idx + 1:])

    def _can_exit(self, text: str, param: ParameterSchema) -> bool:
        """Whether the value generated so far is allowed to close."""
        if not text:
            return False
        if param.type == "number":
            return self._has_valid_decimal(text)
        return True  # integer: any digit generated so far is enough

    def _genarate_num_values(self, tokens: list[int],
                             param: ParameterSchema) -> tuple[str, int]:
        """Open-ended digit/'-'/'.' sampling for a number or integer value,
        until an exit character (',' or '}') is both generated and legal.
        Args:
            tokens (list[int]): running token-id list; mutated in place as
            each sampled token is appended.
            param (ParameterSchema): the parameter currently being filled.
        Returns:
            (text, steps_used): text includes the resolved exit character.
            and nbr of tokens used
        Raises:
            ValueError: if max_steps runs out before a valid value closes.
        """
        candidates = self.fsm.get_allowed_strings()
        gen_value = ""
        for steps in range(self.max_value_tokens):
            allowed = candidates if self._can_exit(gen_value, param) else [
                c for c in candidates if c.strip() not in _EXIT_CHARS
            ]
            allowed_ids = self._allowed_token_ids(strings=allowed,
                                                  accumulator="")
            if not allowed_ids:
                raise ValueError(f"No token can continue {gen_value!r} "
                                 f"in {self.fsm.curr_state}"
                                 )
            token_id = self._get_best_token(tokens=tokens,
                                            allowed_ids=allowed_ids)
            tokens.append(token_id)
            token_str = self.lut.get(token_id)
            if not token_str:
                raise ValueError(f"Token ID {token_id} missing from LUT")
            print(token_str, flush=True, end="")
            gen_value += token_str

            exit_char = next((c for c in _EXIT_CHARS if c in token_str), None)
            if exit_char:
                self.fsm.update_state(exit_char)
                return gen_value, steps

        raise ValueError(
                    f"Exceeded token limit while generating numeric value"
                    f"{self.fsm.curr_state}"
                )

    def _genarate_string_value(self, tokens: list[int]) -> tuple[str, int]:
        """Genarating the string values for the arguments type string
        Args:
            tokens (list[int]):
          Returns:
            (text, steps_used): text includes the resolved exit character.
                and nbr of tokens used
        Raises:
                ValueError: if max_steps runs out before a valid value closes.
        """
        generated = self._commit(tokens=tokens, text='"')
        print(generated, flush=True, end="")

        for step in range(self.max_tokens):
            allowed_ids = list(self.content_ids)

            allowed_ids.extend(self.quote_ids)
            if not allowed_ids:
                raise ValueError("No allowed tokens for string generation")

            token_id = self._get_best_token(tokens, allowed_ids)
            tokens.append(token_id)

            token_text = self.lut.get(token_id)
            if token_text is None:
                raise ValueError(f"Token ID {token_id} missing from LUT")

            print(token_text, flush=True, end="")
            generated += token_text

            if token_id in self.quote_ids:
                # EXPECT_STRING_CONTENT -> EXPECT_PARAM_SEPARATOR_OR_CLOSE.
                self.fsm.update_state('"')
                return generated, step

        raise ValueError("Exceeded token limit while generating a string")

    @ft_time
    def generate(self, prompt: str) -> str:
        """
        Generate the function Calling for each prompt.
        Args:
            prompt (str): single user request
        Raises:

        Returns:
            Buffer (str): valid str genarated by llm to load int a json file
        """
        # list of al the tokens (prompt)
        tokens: list[int] = self._encode_prompt(prompt=prompt)
        buffer: str = ""  # fullstr genarated by llm
        sampled: int = 0

        while self.fsm.curr_state is not JSONState.DONE:
            try:
                # === 1. DETERMINISTIC BYPASS ===
                det_text = self.fsm.get_deterministic_string()
                if det_text:
                    buffer += self._commit(tokens=tokens, text=det_text)
                    print(det_text, end="", flush=True)
                    continue
                # === 2 Func name genaration ===
                if self.fsm.curr_state is JSONState.EXPECT_FUNCTION_NAME:
                    name_text, used_steps = self._genarate_compl_choice(
                        tokens=tokens,
                        allowed=self.fsm.get_allowed_strings()
                    )
                    buffer += name_text
                    sampled += used_steps
                    continue

                # === 3 Paramter Value Genaration ===
                if self.fsm.curr_state is JSONState.EXPECT_PARAM_VALUE:
                    param = self.fsm.curr_single_param
                    if (param and param.type in ("number", "integer")):
                        number_text, used_steps = self._genarate_num_values(
                            tokens=tokens,
                            param=param
                        )
                        buffer += number_text
                        sampled += used_steps
                        continue

                    if param and param.type == "string":
                        str_text, used_steps = self._genarate_string_value(
                            tokens=tokens
                        )
                        buffer += str_text
                        sampled += used_steps
                        continue

            except Exception:
                raise ValueError(f"Error on gen for {prompt}")
        return buffer
