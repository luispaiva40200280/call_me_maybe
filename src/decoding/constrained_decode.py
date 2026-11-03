from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from dataclasses import field as dcfield  # Needed for standard default_factory
from typing import Dict


@dataclass
class TrieNode:
    """"""
    children: dict[str, 'TrieNode'] = dcfield(default_factory=dict)
    token_id: set[int] = dcfield(default_factory=set)


class VocabularyTrie(BaseModel):
    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, token_string: str, token_id: int) -> None:
        """
        """
        curr_node = self.root
        for char in token_string:
            if char not in curr_node.children:
                curr_node.children[char] = TrieNode()
            curr_node = curr_node.children[char]
        curr_node.token_id.add(token_id)

    def build_from_lut(self, lut: Dict[int, str]) -> None:
        """
        Populates the entire Trie from the vocabulary dictionary.
        """
        for token_id, token_string in lut.items():
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
                if node.token_id:
                    valid.update(node.token_id)
        return list(valid)

    def search_by_charset(self, charset: set[str]) -> list[int]:
        """
            Returns all Token IDs composed entirely of the allowed characters.
        """
        valid: set[int] = set()

        def dfs(node: TrieNode, is_root: bool):
            # If this node marks a valid token, allow it
            if node.token_id and not is_root:
                valid.update(node.token_id)
            # Continue down branches that match our allowed characters
            for char, child in node.children.items():
                if char in charset:
                    dfs(child, False)

        dfs(self.root, True)
        return list(valid)
