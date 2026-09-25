"""
Finite State Machine (FSM): State tracker for brackets, commas,
and schema types in generation file
"""
from src.schemas import FunctionDef, ParameterSchema
from enum import Enum, auto
from typing import Optional


class JSONState(Enum):
    """
    Tracks the grammatical steps required to generate a valid function call.
    Example Target: {"name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}}
    """
    # 1. The Root Level
    EXPECT_ROOT_BRACE = auto()       # Expecting '{'
    EXPECT_NAME_KEY = auto()         # Expecting '"name"'
    EXPECT_NAME_COLON = auto()       # Expecting ':'
    EXPECT_FUNCTION_NAME = auto()    # Expecting a valid function name like
    # '"fn_add_numbers"'

    # 2. Transitioning to Parameters
    EXPECT_COMMA = auto()            # Expecting ','
    EXPECT_PARAMS_KEY = auto()       # Expecting '"parameters"'
    EXPECT_PARAMS_COLON = auto()     # Expecting ':'
    EXPECT_PARAMS_DICT_BRACE = auto()  # Expecting '{'

    # 3. Inside the Parameters Dictionary (Dynamic States)
    EXPECT_PARAM_KEY = auto()        # Expecting a parameter name like '"a"'
    # or '}' if no parameters)
    EXPECT_STRING_CONTENT = auto()          # The LLM is typing inside quotes
    EXPECT_COMMA_OR_CLOSE = auto()          # Waiting for ',' or '}'
    # AFTER a value finishes

    EXPECT_PARAM_COLON = auto()      # Expecting ':'
    EXPECT_PARAM_VALUE = auto()      # Expecting a boolean, number, or
    # string based on schema

    # 4. Closing the Structure
    EXPECT_PARAM_COMMA_OR_CLOSE = auto()  # Expecting ',' (if more params)
    # OR '}' (if done)
    EXPECT_ROOT_CLOSE_BRACE = auto()  # Expecting the final '}'
    DONE = auto()                    # Generation is complete


class StateMachine:
    def __init__(self, schemas: list[FunctionDef]) -> None:
        self.schemas = schemas
        self.state: JSONState = JSONState.EXPECT_ROOT_BRACE
        self.curr_func_call: Optional[FunctionDef] = None
        self.curr_param: Optional[ParameterSchema] = None

    def update_state(self, generated_text: str) -> None:
        """
        Travest trought the generated text and updates
        the state macchine that allows to restrict the logit
        and the output of the json
        """
        clean_text = generated_text.strip()
        match self.state:
            case JSONState.EXPECT_ROOT_BRACE:
                if clean_text == "{":
                    self.state = JSONState.EXPECT_NAME_KEY
            case JSONState.EXPECT_NAME_KEY:
                if clean_text == '"name"':
                    self.state = JSONState.EXPECT_NAME_COLON
            case JSONState.EXPECT_NAME_COLON:
                if clean_text == ":":
                    self.state = JSONState.EXPECT_FUNCTION_NAME
            case JSONState.EXPECT_FUNCTION_NAME:
                name = clean_text.strip('"')
                self.curr_func_call = next((func for func in self.schemas
                                            if func.name == name), None)
                if self.curr_func_call is None:
                    raise ValueError(f"Invalid func '{name}' generated.")
                self.state = JSONState.EXPECT_COMMA
            case JSONState.EXPECT_COMMA:
                if clean_text == ",":
                    self.state = JSONState.EXPECT_PARAMS_KEY
            case JSONState.EXPECT_PARAMS_KEY:
                if clean_text == '"parameters"':
                    self.state = JSONState.EXPECT_PARAMS_COLON
            case JSONState.EXPECT_PARAMS_COLON:
                if clean_text == ":":
                    self.state = JSONState.EXPECT_PARAMS_DICT_BRACE
            case JSONState.EXPECT_PARAMS_DICT_BRACE:
                if clean_text == "{":
                    self.state = JSONState.EXPECT_PARAM_KEY
            case JSONState.EXPECT_PARAM_KEY:
                param = clean_text.strip('"')
                if (self.curr_func_call is None or
                        param not in self.curr_func_call.parameters):
                    raise ValueError(f"Invalid parameter '{param}' generated.")
                self.curr_param = self.curr_func_call.parameters[param]
                self.state = JSONState.EXPECT_PARAM_COLON
            case JSONState.EXPECT_PARAM_COLON:
                if clean_text == ":":
                    self.state = JSONState.EXPECT_PARAM_VALUE
            case JSONState.EXPECT_PARAM_VALUE:
                # If the LLM generates a quote, it just started a string.
                if clean_text == '"':
                    self.state = JSONState.EXPECT_STRING_CONTENT
                # If it generates a comma, a boolean/number just finished.
                # Loop back to next key.
                elif clean_text == ",":
                    self.state = JSONState.EXPECT_PARAM_KEY
                # If it generates a brace, a boolean/number just finished.
                # Close the object.
                elif clean_text == "}":
                    self.state = JSONState.EXPECT_ROOT_CLOSE_BRACE
            case JSONState.EXPECT_STRING_CONTENT:
                # Inside the string, we ignore all letters, numbers,
                # and punctuation We ONLY care when the LLM finally
                # generates the closing quote!
                if clean_text == '"':
                    self.state = JSONState.EXPECT_COMMA_OR_CLOSE
            case JSONState.EXPECT_COMMA_OR_CLOSE:
                # The string is closed safely. Now we look for the exit doors.
                if clean_text == ",":
                    self.state = JSONState.EXPECT_PARAM_KEY
                elif clean_text == "}":
                    self.state = JSONState.EXPECT_ROOT_CLOSE_BRACE
            case JSONState.EXPECT_ROOT_CLOSE_BRACE:
                if clean_text == "}":
                    self.state = JSONState.DONE

    def get_allowed_strings(self) -> list[str]:
        """Returns the legally allowed strings for the CURRENT state."""
        import string
        match self.state:
            case (JSONState.EXPECT_ROOT_BRACE |
                    JSONState.EXPECT_PARAMS_DICT_BRACE):
                return ["{", " {", "\n{"]
            case JSONState.EXPECT_NAME_KEY:
                return ['"name"', ' "name"']
            case (JSONState.EXPECT_PARAM_VALUE):
                match (self.curr_param is not None and self.curr_param.type):
                    case "boolean":
                        return ["true", "false", " true", " false"]
                    case "number" | "integer":
                        allowed = list(string.digits)
                        allowed.extend([f" {d}" for d in string.digits])
                        allowed.extend(["-", " -"])
                        if self.curr_param.type == "number":
                            allowed.extend([".", " ."])
                        allowed.extend([",", " ,", "}", " }"])
                        return allowed
                    case "string":
                        return ['"', ' "']
            case JSONState.EXPECT_STRING_CONTENT:
                allowed = list(string.printable)
                allowed.extend([f" {char}" for char in allowed])
                return allowed
            case JSONState.EXPECT_FUNCTION_NAME:
                allowed = []
                for func in self.schemas:
                    # Allow exact match, and match with leading space
                    allowed.extend([f'"{func.name}"', f' "{func.name}"'])
                return allowed
            case JSONState.EXPECT_PARAM_KEY:
                allowed = []
                if self.curr_func_call is not None:
                    for key in self.curr_func_call.parameters.keys():
                        allowed.extend([
                            f'"{key}"',
                            f' "{key}"',
                            f'\n"{key}"',
                            f'\n "{key}"'
                        ])
                allowed.extend(["}", " }", "\n}", "\n }"])
                return allowed
            case JSONState.EXPECT_COMMA_OR_CLOSE:
                return [",", " ,", "}", " }"]
            case (JSONState.EXPECT_NAME_COLON |
                  JSONState.EXPECT_PARAMS_COLON |
                  JSONState.EXPECT_PARAM_COLON):
                return [":", " :"]
            case JSONState.EXPECT_COMMA:
                return [",", " ,", ",\n", " ,\n"]
            case JSONState.EXPECT_PARAMS_KEY:
                return ['"parameters"', ' "parameters"', '\n"parameters"',
                        '\n "parameters"']
            case JSONState.EXPECT_ROOT_CLOSE_BRACE:
                return ["}", " }", "\n}", "\n }"]
            case JSONState.DONE:
                return []
        return []
