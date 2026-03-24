import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # 1. Start with special tokens to ensure they have consistent IDs (0, 1, 2, 3)
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # 2. Extract unique words from the provided texts
        unique_words = set()
        for text in texts:
            # Simple whitespace splitting for this word-level tokenizer
            words = text.lower().split()
            unique_words.update(words)
        
        # 3. Combine and build the mapping dictionaries
        all_tokens = special_tokens + sorted(list(unique_words))
        
        for i, word in enumerate(all_tokens):
            self.word_to_id[word] = i
            self.id_to_word[i] = word
            
        self.vocab_size = len(all_tokens)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # Split text and look up IDs, defaulting to the <UNK> ID if not found
        words = text.lower().split()
        unk_id = self.word_to_id.get(self.unk_token)
        
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # Look up words by ID and join them with spaces
        words = [self.id_to_word.get(idx, self.unk_token) for idx in ids]
        return " ".join(words)