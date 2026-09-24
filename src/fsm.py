"""
Finite State Machine (FSM): State tracker for brackets, commas,
and schema types in generation file
"""
from src.schemas import Schema
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
    EXPECT_PARAM_COLON = auto()      # Expecting ':'
    EXPECT_PARAM_VALUE = auto()      # Expecting a boolean, number, or
    # string based on schema

    # 4. Closing the Structure
    EXPECT_PARAM_COMMA_OR_CLOSE = auto()  # Expecting ',' (if more params)
    # OR '}' (if done)
    EXPECT_ROOT_CLOSE_BRACE = auto()  # Expecting the final '}'
    DONE = auto()                    # Generation is complete


class StateMachine:
    def __init__(self, schema: list[Schema]):
        self.schemas = schema
        self.current_state: JSONState = JSONState.EXPECT_ROOT_BRACE
        self.current_func_call: Optional[str] = None
