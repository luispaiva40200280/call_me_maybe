from pydantic import BaseModel, field_validator


class PromptSchema(BaseModel):
    """Schema for validating objects inside function_calling_tests.json"""
    prompt: str


class ParameterSchema(BaseModel):
    """Pydantic model for individual parameter definitions."""
    type: str

    @field_validator('type')
    def validate_parameter_schema(cls, value: str) -> str:
        """Validate the parameter input, check for errors"""
        valid_types: list[str] = ["string", "number", "integer", "boolean"]
        if value not in valid_types:
            raise ValueError(f"Invalid type '{value}' in ParameterSchema.")
        return value


class FunctionDef(BaseModel):
    """Schema for validating objects inside functions_definition.json"""
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema
