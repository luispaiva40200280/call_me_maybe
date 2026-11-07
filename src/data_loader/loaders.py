"""Loading the info/data of json files and llm data"""
from src.decoding.schemas import FunctionDef, PromptSchema
from pathlib import Path
from llm_sdk import Small_LLM_Model
import json


def load_func_data(path: str) -> list[FunctionDef]:
    """
    """

    path_file = Path(path)
    if not path_file.exists():
        raise FileNotFoundError(f"The file was not found at: {path}")
    with open(path_file, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return [FunctionDef(**def_func) for def_func in data]


def load_prompt_data(path: str) -> list[PromptSchema]:
    """
    """

    path_file = Path(path)
    if not path_file.exists():
        raise FileNotFoundError(f"The file was not found at: {path}")
    with open(path_file, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return [PromptSchema(**prompt) for prompt in data]


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
    Args:
        model (Small_llm_model):
    Return:
        lut (dict[int,str])
    """
    path = model.get_path_to_vocab_file()
    vocab: dict[str, int] = {}
    with open(path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    lut: dict[int, str] = {v: clean_token_text(k) for k, v in vocab.items()}
    return lut, vocab
