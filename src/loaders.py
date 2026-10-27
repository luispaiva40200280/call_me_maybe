"""Loading the info/data of json files and llm data"""
from src.schemas import FunctionDef, PromptSchema
from pathlib import Path
from typing import Any
from llm_sdk import Small_LLM_Model
import json


def parse_file(path: str, type: str) -> list[Any]:
    """
    Safely loads the JSON file.
    Returns:
        data (list[dict[str, Any]])
    """
    path_file = Path(path)
    if not path_file.exists():
        raise FileNotFoundError(f"The {type} file was not \
found at: {path}")
    with open(path_file, 'r', encoding="utf-8") as file:
        data = json.load(file)

    # validate the data and change it to Pydantic module
    if type == "prompt":
        return [PromptSchema(**prompt) for prompt in data]
    elif type == "definitions":
        return [FunctionDef(**def_func) for def_func in data]

    return data


def clean_token_text(token_str: str) -> str:
    """
    Normalizes BPE and SentencePiece whitespace markers into standard ASCII
    """
    return (
        token_str.replace("Ġ", " ")   # Qwen / GPT-2 space (\u0120)
        .replace("Ċ", "\n")           # Qwen / GPT-2 newline (\u010a)
        .replace("ĉ", "\t")           # Qwen / GPT-2 tab (\u0109)
        .replace("\u2581", " ")       # SentencePiece space (▁)
    )


def load_vocab_lut(model: Small_LLM_Model) -> tuple[dict[int, str],
                                                    dict[str, int]]:
    """
    load the vocab dict from model in the (dict[str: token])
    """
    path = model.get_path_to_vocab_file()
    vocab: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    lut: dict[int, str] = {v: clean_token_text(k) for k, v in vocab.items()}
    return lut, vocab
