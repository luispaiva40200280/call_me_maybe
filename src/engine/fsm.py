"""
Finite State Machine (FSM): State tracker for brackets, commas,
and schema types in generation file.
"""
from pydantic import BaseModel, Field
from typing import Optional
from src.decoding.schemas import FunctionDef, ParameterSchema
from . import TRANSITIONS, JSONState, DETERMINISTC_STRINGS
# import string


class StateMachine(BaseModel):
    """
    """
    schemas: list[FunctionDef]
    curr_state: JSONState = JSONState.EXPECT_OPEN_BRACE
    prev_state: Optional[JSONState] = None
    curr_func_calling: Optional[FunctionDef] = None
    curr_params_schem: Optional[dict[str, ParameterSchema]] = None
    curr_single_param: Optional[ParameterSchema] = None
    curr_param_name: Optional[str] = None
    generated_params: set[str] = Field(default_factory=set)

    def _get_params_schema(self) -> dict[str, ParameterSchema] | None:
        """Updates the active function definition based on its name."""
        if self.curr_func_calling is None:
            return None
        return self.curr_func_calling.parameters

    def _update_curr_func_calling(self, func_name: str) -> None:
        """Retrieves parameters schema if a function is currently selected."""
        self.curr_func_calling = next(
                    (func for func in self.schemas if func.name == func_name),
                    None
                )
        if self.curr_func_calling is None:
            raise ValueError(f"Unknown function name: {func_name}")

    def _transition_to(self, new_state: JSONState) -> None:
        """Helper to cleanly track state history."""
        self.prev_state = self.curr_state
        self.curr_state = new_state

    def _handle_param_key(self, text: str) -> None:
        """"""
        if text == "}":
            self._transition_to(JSONState.EXPECT_CLOSE_BRACE)
            return
        param_name = text.strip('"')
        params = self._get_params_schema()
        if params is None or param_name not in params:
            raise ValueError(f"Invalid parameter {param_name}")
        self.curr_single_param = params[param_name]
        self.generated_params.add(param_name)
        self.curr_param_name = param_name
        self._transition_to(JSONState.EXPECT_COLON)

    def _handle_func_name(self, text: str) -> None:
        """
        """
        name = text.strip('"')
        self._update_curr_func_calling(name)
        self._transition_to(JSONState.EXPECT_COMMA)

    def _get_allowed_functions(self) -> list[str]:
        """
        Gets the func names allowed for the llm to choose
        """
        allowed = []
        for func in self.schemas:
            allowed.extend([f'"{func.name}"', f' "{func.name}"'])
        return allowed

    def _get_allowed_params(self) -> list[str]:
        """
        Gets the valid paramters based on the function choosen by llm
        """
        allowed = []
        missing = False
        params = self._get_params_schema()
        if params:
            for key in params.keys():
                if key not in self.generated_params:
                    missing = True
                    allowed.extend([f'"{key}"', f' "{key}"'])
        if not missing:
            allowed.extend(["}", " }"])
        return allowed

    def _get_allowed_values(self) -> list[str]:
        """
        Getter for the allowed tokens in the paramters values
        """
        import string

        if (self.curr_single_param and
                self.curr_single_param.type == "boolean"):
            return ["true", "false", " true", " false"]
        elif (self.curr_single_param and
                self.curr_single_param.type in ("number", "integer")):
            allowed = list(string.digits)
            allowed.extend([f" {d}" for d in string.digits])
            allowed.extend(["-", " -"])
            if self.curr_single_param.type == "number":
                allowed.extend([".", " ."])
            allowed.extend([",", " ,", "}", " }"])
            return allowed
        elif (self.curr_single_param is not None
                and self.curr_single_param.type == "string"):
            return ['"', ' "']
        return ['"', ' "']

    def update_state(self, gen_str: str) -> None:
        """
        Main method so it updates the State of the llm
        """
        text = gen_str.strip()
        # These states accept schema-dependent choices, not fixed table
        # entries.
        if self.curr_state is JSONState.EXPECT_FUNCTION_NAME:
            self._handle_func_name(text)
            return

        if self.curr_state is JSONState.EXPECT_PARAM_KEY:
            self._handle_param_key(text)
            return

        key = (self.curr_state, self.prev_state, text)
        next_state = TRANSITIONS.get(key)

        if next_state is None:
            raise ValueError(
                f"No transition for {text!r} in {self.curr_state.name}"
            )

        self._transition_to(next_state)

    def get_allowed_strings(self) -> list[str]:
        """
        Main method so the genarator can get the allowed token depending on
        the current state of the genaration
        """
        # see if it is a dynamic state: just as func name or params keys
        if self.curr_state in self.dynamic_handlers:
            return self.dynamic_handlers[self.curr_state]()

        choices = [
            text
            for (current, previous, text) in TRANSITIONS
            if current is self.curr_state and previous is self.prev_state
        ]

        return list(dict.fromkeys(
            variant
            for text in choices
            for variant in (text, f" {text}")
        ))

    def get_deterministic_string(self) -> Optional[str]:
        """Returns the mandatory string for strictly deterministic states."""
        if self.curr_state is JSONState.EXPECT_PARAM_SEPARATOR_OR_CLOSE:
            if self.curr_func_calling:
                missing = len(self.generated_params) < len(
                    self.curr_func_calling.parameters
                )
                return "," if missing else "}"
        if self.curr_state is JSONState.EXPECT_PARAM_KEY:
            if self.curr_func_calling:
                for key in self.curr_func_calling.parameters.keys():
                    if key not in self.generated_params:
                        return f'"{key}"'
                return "}"
        return DETERMINISTC_STRINGS.get(self.curr_state, None)

    def reset(self) -> None:
        """Restores the FSM to its initial state for a new generation."""
        self.curr_state = JSONState.EXPECT_OPEN_BRACE
        self.prev_state = None
        self.curr_func_calling = None
        self.curr_single_param = None
        self.curr_params_schem = None
        self.generated_params.clear()

    @property
    def dynamic_handlers(self):
        return {
            JSONState.EXPECT_FUNCTION_NAME: self._get_allowed_functions,
            JSONState.EXPECT_PARAM_KEY: self._get_allowed_params,
            JSONState.EXPECT_PARAM_VALUE: self._get_allowed_values,
        }
