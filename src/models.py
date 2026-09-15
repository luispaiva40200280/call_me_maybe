from typing import Dict
from pydantic import BaseModel  # type: ignore


class ParameterDef(BaseModel):
    type: str


class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Dict[str, ParameterDef]
    returns: Dict[str, str]


class PromptInput(BaseModel):
    prompt: str
