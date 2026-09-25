from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class TrieNode:
    # default_factory ensures each node gets its own separate dictionary
    # 'TrieNode' is in quotes as a forward reference for recursive typing
    # None means this specific node does not mark the end of a valid token
    children: dict[str, 'TrieNode'] = field(default_factory=dict)
    token_id: Optional[int] = None


class VocabularyTrie:
    def __init__(self) -> None:
        self.root = TrieNode()

    def insert(self, token_string: str, token_id: int) -> None:
        """
        """
        curr_node = self.root
        for char in token_string:
            if char not in curr_node.children:
                curr_node.children[char] = TrieNode()
            curr_node = curr_node.children[char]
        curr_node.token_id = token_id

    def build_from_lut(self, vocab: Dict[str, int]) -> None:
        """
        Populates the entire Trie from the vocabulary dictionary.
        """
        for token_string, token_id in vocab.items():
            self.insert(token_string, token_id)

    def search_allowed_tokens(self, str_allowed: list[str]) -> list[int]:
        """"
        Searches the Trie for the allowed strings.
        Returns Token IDs for exact matches AND valid partial prefixes.
        """
        valid: set = set()
        for word in str_allowed:
            node = self.root
            for char in word:
                if char not in node.children:
                    break
                node = node.children[char]
                if node.token_id is not None:
                    valid.add(node.token_id)
        return list(valid)
