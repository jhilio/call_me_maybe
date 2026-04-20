
import json
from pathlib import Path
from typing import Any, Dict
from pydantic import BaseModel, Field, field_validator

class Parameter(BaseModel):
    type: str = Field(..., min_length=1, description="Type must be a non-empty string")
    
    class Config:
        extra = "forbid"

    @field_validator("type")
    def validate_type(cls, v):
        ALLOWED_TYPES = {"string", "integer", "number", "boolean"}
        if v not in ALLOWED_TYPES:
            raise ValueError(f"Invalid type '{v}'. Must be one of {ALLOWED_TYPES}")
        return v

class FunctionDef(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    parameters: Dict[str, Parameter] = Field(..., min_length=1)
    returns: Parameter

    class Config:
        extra = "forbid"


class Prompt(BaseModel):
    prompt: str = Field(..., min_length=1)

    class Config:
        extra = "forbid"

def get_func_def(input_path: str) -> list[dict]:
    """open the path in argument read it, and verify each function has correct format

    Args:
        input_path (str): path to function definition

    Raises:
        various raises due to file managing and value verification by pydantic

    Returns:
        list[dict]: all found function in form of a list of dict (function)
    """    
    with open(input_path, "r") as f:
        functions = json.load(f)
    if not isinstance(functions, list):
        raise IOError("function def is not a list of function")
    return [FunctionDef(**func).model_dump() for func in functions]

def get_prompt(input_path: str) -> list[str]:
    """open the path in argument read it, and verify each prompt has correct format

    Args:
        input_path (str): path to prompt list

    Raises:
        various raises due to file managing and value verification by pydantic

    Returns:
        list[dict]: all found prompt in form of a list of dict (prompt)
    """    
    with open(input_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise IOError("prompt file should contain a list of prompts")
    return [Prompt(**prom).model_dump()["prompt"] for prom in data]


def generate_json_output(output: list[dict[str, Any]], output_path: str) -> None:
    """uses json.dump on a list of dict formated to the subject demand.
    all of that at the specified output path

    Args:
        output (list[dict[str, Any]]): list containing all function info needed in the oupout
        output_path (str): where to write the output file
    """    
    data = [{
            "prompt": dic["prompt"],
            "name":  dic["func_name"],
            "parameters": dic["parameters"]
        } for dic in output]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

