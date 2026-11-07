from ..import ft_time
from .import JSONState
from ..decoding import FunctionDef
from .fsm import StateMachine
from pydantic import BaseModel, ConfigDict, Field
from typing import Any


class Generator(BaseModel):
    """
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Any
    fsm: StateMachine
    func_call: list[FunctionDef]
    max_tokens: int = 50  # max tokens for gen text
    max_value_tokens: int = 24  # max tokens for gen numbers
    lut: dict[int, str] = Field(default_factory=dict)
    vocab_trie: Any = None
    max_tokens: int = 150

    def model_post_init(self, __context: Any) -> None:
        """
        """
        from ..data_loader.loaders import load_vocab_lut
        from ..decoding.constrained_decode import VocabularyTrie
        # ola
        self.lut, _ = load_vocab_lut(self.model)
        self.vocab_trie = VocabularyTrie()
        self.vocab_trie.build_from_lut(self.lut)

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

    def _commit(self, tokens: list[int], text: str) -> None:
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
                               allowed: list[str],
                               max_steps: int = 50) -> tuple[str, int]:
        """Generate one complete choice, such as a function name."""

        accumulator = ""
        generated_text = ""
        for steps in range(max_steps):
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
                    remaining = self.max_tokens - sampled
                    if remaining <= 0:
                        raise ValueError(
                            f"Exceeded max_tokens={self.max_tokens}"
                        )
                    name_text, used_steps = self._genarate_compl_choice(
                        tokens=tokens,
                        allowed=self.fsm.get_allowed_strings(),
                        max_steps=remaining)
                    buffer += name_text
                    sampled += used_steps
                    continue

                # === 3 Paramter Value Genaration ===
                if self.fsm.curr_state is JSONState.EXPECT_PARAM_VALUE:
                    


            except Exception:
                raise ValueError(f"Error on gen for {prompt}")
        return buffer
