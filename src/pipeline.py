from llm_sdk import Small_LLM_Model
from src.constrained_decode import VocabularyTrie
from src.loaders import load_vocab_lut


class Pipeline:
    def __init__(self, model: Small_LLM_Model) -> None:
        self.model = model
        self.lut, self.vocab = load_vocab_lut(model)
        self.vocab_trie = VocabularyTrie()
        self.vocab_trie.build_from_lut(self.vocab)
