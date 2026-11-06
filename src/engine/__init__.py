# src/engine/__init__.py
from enum import Enum, auto


class JSONState(Enum):
    """
    Tracks the grammatical steps required to generate a valid function call.
    """
    # 1. Expecting single tokens for division or opening/closing json  dict
    #   i.e :
    #       "{", "}", ":", ",",
    EXPECT_OPEN_BRACE = auto()       # '{'
    EXPECT_COMMA = auto()            # ','
    EXPECT_COLON = auto()       # ':'
    EXPECT_CLOSE_BRACE = auto()  # Final '}'

    # States for deterministic strings just as name/paramters
    EXPECT_NAME_KEY = auto()         # '"name"'
    EXPECT_PARAMS_KEY = auto()       # '"parameters"'

    # Parameter/Name Key Value States Determ
    #   Genaration of the function name and the paramters that it recives
    #   the params depends the function that it chooses
    EXPECT_FUNCTION_NAME = auto()    # Function name string
    EXPECT_PARAM_KEY = auto()        # Parameter name or '}'
    EXPECT_PARAM_VALUE = auto()      # Value (string, number, boolean)
    EXPECT_STRING_CONTENT = auto()    # expect a "...{llm txt gen}"
    EXPECT_PARAM_SEPARATOR_OR_CLOSE = auto()    # expect "," or "}"

    # 4. Closing Structure
    DONE = auto()                    # Generation complete


# Fixed grammar transitions:
# - Keys like "name" and "parameters" are fixed
# - Punctuation like { } : , are fixed
# - Dynamic choices like function name / param names are handled in helper
# methods
TRANSITIONS = {
    # root object
    (JSONState.EXPECT_OPEN_BRACE, None, "{"):
        JSONState.EXPECT_NAME_KEY,

    # name field
    (JSONState.EXPECT_NAME_KEY, JSONState.EXPECT_OPEN_BRACE, '"name"'):
        JSONState.EXPECT_COLON,
    (JSONState.EXPECT_COLON, JSONState.EXPECT_NAME_KEY, ":"):
        JSONState.EXPECT_FUNCTION_NAME,
    # function name is dynamic, handled separately
    (JSONState.EXPECT_COMMA, JSONState.EXPECT_FUNCTION_NAME, ","):
        JSONState.EXPECT_PARAMS_KEY,

    # parameters field
    (JSONState.EXPECT_PARAMS_KEY, JSONState.EXPECT_COMMA, '"parameters"'):
        JSONState.EXPECT_COLON,
    (JSONState.EXPECT_COLON, JSONState.EXPECT_PARAMS_KEY, ":"):
        JSONState.EXPECT_OPEN_BRACE,
    (JSONState.EXPECT_OPEN_BRACE, JSONState.EXPECT_COLON, "{"):
        JSONState.EXPECT_PARAM_KEY,

    # parameter value
    (JSONState.EXPECT_COLON, JSONState.EXPECT_PARAM_KEY, ":"):
        JSONState.EXPECT_PARAM_VALUE,
    (JSONState.EXPECT_PARAM_VALUE, JSONState.EXPECT_COLON, '"'):
        JSONState.EXPECT_STRING_CONTENT,

    # string value ends
    (JSONState.EXPECT_STRING_CONTENT, JSONState.EXPECT_PARAM_VALUE, '"'):
        JSONState.EXPECT_PARAM_SEPARATOR_OR_CLOSE,

    # after a STRING value closes, next is another parameter or close.
    # prev_state here is EXPECT_STRING_CONTENT because that's what was
    # current right before the closing '"' fired (see the entry above).
    (JSONState.EXPECT_PARAM_SEPARATOR_OR_CLOSE,
        JSONState.EXPECT_STRING_CONTENT, ","):
        JSONState.EXPECT_PARAM_KEY,
    (JSONState.EXPECT_PARAM_SEPARATOR_OR_CLOSE,
        JSONState.EXPECT_STRING_CONTENT, "}"):
        JSONState.EXPECT_CLOSE_BRACE,

    # after a NUMBER/BOOLEAN value: there's no closing-quote step for
    # these types, so curr_state is still EXPECT_PARAM_VALUE and
    # prev_state is still EXPECT_COLON (unchanged since the value
    # started) when the ',' or '}' appears -- NOT
    # EXPECT_PARAM_SEPARATOR_OR_CLOSE, which only ever gets reached via
    # the string path above. Keying these off
    # (EXPECT_PARAM_SEPARATOR_OR_CLOSE, EXPECT_PARAM_VALUE, ...) is
    # unreachable dead code; these two entries are what actually fires.
    (JSONState.EXPECT_PARAM_VALUE, JSONState.EXPECT_COLON, ","):
        JSONState.EXPECT_PARAM_KEY,
    (JSONState.EXPECT_PARAM_VALUE, JSONState.EXPECT_COLON, "}"):
        JSONState.EXPECT_CLOSE_BRACE,

    # end root object -- two entries because prev_state differs
    # depending on which of the two paths above closed the params dict
    # (last parameter was a string, vs. a number/boolean).
    (JSONState.EXPECT_CLOSE_BRACE,
        JSONState.EXPECT_PARAM_SEPARATOR_OR_CLOSE, "}"):
        JSONState.DONE,
    (JSONState.EXPECT_CLOSE_BRACE, JSONState.EXPECT_PARAM_VALUE, "}"):
        JSONState.DONE,
}


DETERMINISTC_STRINGS = {
    JSONState.EXPECT_PARAMS_KEY: '"parameters"',
    JSONState.EXPECT_NAME_KEY: '"name"',
    JSONState.EXPECT_COMMA: ", ",
    JSONState.EXPECT_COLON: ": ",
    JSONState.EXPECT_OPEN_BRACE: "{",
    JSONState.EXPECT_CLOSE_BRACE: "}",
}
