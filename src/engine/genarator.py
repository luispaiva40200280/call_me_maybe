from ..import ft_time
from ..decoding import FunctionDef
from .fsm import StateMachine
from pydantic import BaseModel, ConfigDict
from typing import Any


class Genarator(BaseModel):
    """
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    model: Any
    fsm: StateMachine
    func_call: list[FunctionDef]
    max_tokens: int = 50  # max tokens for gen text
    max_value_tokens: int = 24  # max tokens for gen numbers

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
        self.fsm.reset()  # resete the state to the beginning
        system = self._build_system_prompt(prompt)
        tensor = self.model.encode(system)  # list all tokens in the prompt
        TOKENS = tensor[0].tolist()  # list of al the tokens (prompt)
        Buffer = ""  # str genarated by llm

        return Buffer
