import re
import json
import random
from typing import List, Dict, Tuple, Counter as CounterType, Optional
from collections import Counter


class FreqWeightedTokenizer:
    """
    A variant of BPE tokenizer that samples from the weighted distribution
    of token pairs based on their frequencies, rather than always selecting
    the most frequent pair.
    """
    
    def __init__(self, vocab_size: int = 10000, special_tokens: Optional[List[str]] = None):
        """
        Initialize the tokenizer.
        
        Args:
            vocab_size: Maximum size of the vocabulary
            special_tokens: List of special tokens to include in vocabulary
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<unk>", "<pad>", "<bos>", "<eos>"]
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int] = {}
        
        # Initialize special tokens in vocabulary
        for i, token in enumerate(self.special_tokens):
            self.vocab[token] = i
    
    def get_pair_frequencies(self, tokens: List[str]) -> CounterType[Tuple[str, str]]:
        """
        Count frequencies of adjacent token pairs.
        
        Args:
            tokens: List of tokens
            
        Returns:
            Counter with pair frequencies
        """
        if len(tokens) < 2:
            return Counter()
        return Counter(zip(tokens, tokens[1:]))
    
    def apply_merge(self, tokens: List[str], a: str, b: str, merged: str) -> List[str]:
        """
        Apply a merge operation to a list of tokens.
        
        Args:
            tokens: List of tokens
            a: First token in the pair
            b: Second token in the pair
            merged: Token to replace the pair with
            
        Returns:
            Updated list of tokens after the merge
        """
        new_tokens = []
        i = 0
        
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
                
        return new_tokens
    
    def train(self, text: str) -> None:
        """
        Train the tokenizer on the given text.
        
        Args:
            text: Training text
        """
        # Start with characters
        tokens = list(text)
        
        # Get initial vocabulary
        initial_vocab = set(tokens)
        
        # Calculate number of merges to perform
        num_merges = self.vocab_size - len(initial_vocab) - len(self.special_tokens)
        num_merges = max(0, num_merges)  # Ensure non-negative
        
        # Clear previous merges
        self.merges = []
        
        # Perform merges
        for i in range(num_merges):
            # Get pair frequencies
            pair_freq = self.get_pair_frequencies(tokens)
            if not pair_freq:
                break
                
            # Get pairs and their frequency weights
            pairs = list(pair_freq.keys())
            weights = list(pair_freq.values())
            
            # Sample based on frequency weights
            a, b = random.choices(pairs, weights=weights, k=1)[0]
            
            # Create a unique merged token with an identifier
            merged = f"{a}{b}@{i}"
            
            # Store the merge
            self.merges.append((a, b))
            
            # Apply the merge to all tokens
            tokens = self.apply_merge(tokens, a, b, merged)
        
        # Build vocabulary from final tokens and special tokens
        all_tokens = set(tokens)
        
        # Start with special tokens (already initialized in __init__)
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Add all other tokens
        i = len(vocab)
        for token in sorted(all_tokens):
            if token not in vocab:
                vocab[token] = i
                i += 1
        
        self.vocab = vocab
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        tokens = list(text)
        
        # Apply merges in the same order as during training
        for i, (a, b) in enumerate(self.merges):
            merged = f"{a}{b}@{i}"
            tokens = self.apply_merge(tokens, a, b, merged)
        
        # Convert tokens to IDs
        return [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        # Convert IDs to tokens
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab.get(idx, "<unk>") for idx in ids]
        
        # Filter out special tokens
        tokens = [token for token in tokens if token not in self.special_tokens]
        
        # Remove the merge identifiers
        text = ''.join(re.sub(r'@\d+', '', token) for token in tokens)
        
        return text
    
    def save(self, path: str) -> None:
        """
        Save tokenizer to a JSON file.
        
        Args:
            path: Path to save file
        """
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "vocab": self.vocab
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'FreqWeightedTokenizer':
        """
        Load tokenizer from a JSON file.
        
        Args:
            path: Path to load file
            
        Returns:
            Loaded tokenizer
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"]
        )
        
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        tokenizer.vocab = data["vocab"]
        
        return tokenizer


