"""Loading the info/data of json files and llm data"""
from src.schemas import FunctionDef, PromptSchema
from pathlib import Path
from typing import Any
from llm_sdk import Small_LLM_Model
import json


class FileHandler:
    """Class for opening and reading JSON files safely."""
    def __init__(self, path_name: str, file_type: str) -> None:
        self.path_name = path_name
        self.file_type = file_type

    def validate_data(self, data: list[dict[str, Any]]) -> list[Any]:
        """Validate of the data using the pydantic modules"""
        if self.file_type == "prompt":
            return [PromptSchema(**prompt) for prompt in data]
        # if the file is func calling needs another validation
        elif self.file_type == "definitions":
            return [FunctionDef(**def_func) for def_func in data]
        return data

    def parse_file(self) -> list[Any]:
        """
        Safely loads the JSON file.
        Returns:
            data (list[dict[str, Any]])
        """
        path_file = Path(self.path_name)
        if not path_file.exists():
            raise FileNotFoundError(f"The {self.file_type} file was not \
found at: {self.path_name}")
        with open(path_file, 'r', encoding="utf-8") as file:
            data = json.load(file)
        return self.validate_data(data)


def load_vocab_lut(model: Small_LLM_Model) -> tuple[dict[int, str],
                                                    dict[str, int]]:
    """
    load the vocab dict from model in the (dict[str: token])
    """
    path = model.get_path_to_vocab_file()
    vocab: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    lut: dict[int, str] = {v: k for k, v in vocab.items()}
    return lut, vocab
