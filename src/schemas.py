from pydantic import BaseModel


class PromptInput(BaseModel):
    """Schema for validating objects inside function_calling_tests.json"""
    prompt: str


class FunctionDef(BaseModel):
    """Schema for validating objects inside functions_definition.json"""
    name: str
    description: str
