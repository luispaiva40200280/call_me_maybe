from pydantic import BaseModel, Field
from pydantic.dataclasses import dataclass
from dataclasses import field as dcfield  # Needed for standard default_factory
from typing import Dict


@dataclass
class TrieNode:
    """
    A single node within the VocabularyTrie.

    Attributes:
        children (dict[str, TrieNode]): Mapping of a character to child Trie
        nodes.
        token_id (set[int]): Set of token IDs associated with this node.
    """
    children: dict[str, 'TrieNode'] = dcfield(default_factory=dict)
    token_id: set[int] = dcfield(default_factory=set)


class VocabularyTrie(BaseModel):
    """
    A prefix tree (trie) built from vocabulary lookups to efficiently search
    and filter token IDs by prefix strings or character sets.

    Attributes:
        root (TrieNode): The root node of the trie.
    """
    root: TrieNode = Field(default_factory=TrieNode)

    def insert(self, token_string: str, token_id: int) -> None:
        """
        Inserts a token string and its corresponding token ID into the trie.

        Args:
            token_string (str): The string representation of the token to
            insert.

            token_id (int): The numeric ID associated with the token.

        Returns:
            None
        """
        curr_node = self.root
        for char in token_string:
            if char not in curr_node.children:
                curr_node.children[char] = TrieNode()
            curr_node = curr_node.children[char]
        curr_node.token_id.add(token_id)

    def build_from_lut(self, lut: Dict[int, str]) -> None:
        """
        Populates the entire trie from a vocabulary lookup table.
        Args:
            lut (Dict[int, str]): A dictionary mapping token IDs to their
            string representations.
        Returns:
            None
        """
        for token_id, token_string in lut.items():
            self.insert(token_string, token_id)

    def search_allowed_tokens(self, str_allowed: list[str]) -> list[int]:
        """"
        Searches the Trie for the allowed strings.
        Returns token IDs for exact matches as well as valid partial prefixes
        founDalong the paths of the given strings.
        Args:
            str_allowed (list[str]): A list of allowed string prefixes
            or words to search for.
        Returns:
            list[int]: A list of unique token IDs matching the allowed
            prefixes or words.
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
        Finds all token IDs that can be formed entirely using characters
        from a given set.

        Args:
            charset (set[str]): A set of allowed characters.

        Returns:
            list[int]: A list of unique token IDs composed solely
            of characters from the charset.
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
