from llm_sdk import Small_LLM_Model
from src.constrained_decode import VocabularyTrie
from src.loaders import load_vocab_lut
from src.fsm import StateMachine as fsm
from src.schemas import FunctionDef


class Pipeline:
    def __init__(self, model: Small_LLM_Model,
                 func_call: list[FunctionDef]) -> None:
        self.model = model
        self.lut, self.vocab = load_vocab_lut(model)
        self.vocab_trie = VocabularyTrie()
        self.vocab_trie.build_from_lut(self.vocab)
        self.fsm = fsm(schemas=func_call)
