import re
import json
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Set, Optional, Union

class BPETokenizer:
    """
    Byte Pair Encoding (BPE) tokenizer implementation.
    
    This tokenizer properly handles spaces and special characters, and includes
    efficient encoding and decoding mechanisms.
    """
    
    def __init__(
        self, 
        vocab_size: int = 10000, 
        special_tokens: Optional[List[str]] = None
    ):
        """
        Initialize the BPE tokenizer.
        
        Args:
            vocab_size: Maximum size of the vocabulary
            special_tokens: List of special tokens to include in vocabulary
        """
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
        self.vocab: Dict[str, int] = {}  # token → id
        self.merges: List[Tuple[str, str]] = []  # ordered list of merges
        self.word_cache: Dict[str, List[int]] = {}  # cache for encoding
    
    def _tokenize_text(self, text: str) -> List[List[str]]:
        """
        Split text into a list of character sequences, preserving whitespace.
        Each word ends with "</w>" and each space is a separate token.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of character sequences
        """
        # Normalize whitespace (convert tabs, newlines to spaces)
        text = re.sub(r'\s+', ' ', text)
        
        # Split into characters, preserving spaces
        tokens_list = []
        current_word = []
        
        for char in text:
            if char == ' ':
                # End the previous word if any
                if current_word:
                    current_word.append("</w>")
                    tokens_list.append(current_word)
                    current_word = []
                # Add space as its own token
                tokens_list.append([' '])
            else:
                current_word.append(char)
        
        # Add the last word if there is one
        if current_word:
            current_word.append("</w>")
            tokens_list.append(current_word)
        
        return tokens_list
    
    def get_stats(self, tokens_list: List[List[str]]) -> Counter:
        """
        Count frequency of adjacent token pairs.
        
        Args:
            tokens_list: List of token sequences
            
        Returns:
            Counter with pair frequencies
        """
        pairs = Counter()
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return pairs
    
    def merge_pair(
        self, 
        tokens_list: List[List[str]], 
        pair: Tuple[str, str], 
        merged_token: str
    ) -> List[List[str]]:
        """
        Apply a merge operation to all token sequences.
        
        Args:
            tokens_list: List of token sequences
            pair: Pair of tokens to merge
            merged_token: New token to replace the pair
            
        Returns:
            Updated token sequences
        """
        new_tokens_list = []
        for tokens in tokens_list:
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokens_list.append(new_tokens)
        return new_tokens_list
    
    def build_vocab(self, tokens_list: List[List[str]]) -> Dict[str, int]:
        """
        Build vocabulary from token sequences.
        
        Args:
            tokens_list: List of token sequences
            
        Returns:
            Vocabulary mapping tokens to ids
        """
        # Collect all unique tokens
        all_tokens = set()
        for tokens in tokens_list:
            all_tokens.update(tokens)
        
        # Create vocabulary starting with special tokens
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        
        # Add all other tokens
        i = len(vocab)
        for token in sorted(all_tokens):
            if token not in vocab:
                vocab[token] = i
                i += 1
                
                # Stop if we've reached the maximum vocab size
                if i >= self.vocab_size:
                    break
        
        return vocab
    
    def train(self, text: str) -> None:
        """
        Train the BPE tokenizer on the given text.
        
        Args:
            text: Training text
        """
        # Tokenize text into character sequences
        tokens_list = self._tokenize_text(text)
        
        # Collect initial vocabulary
        initial_vocab = set(token for tokens in tokens_list for token in tokens)
        
        # Calculate number of merges to perform
        num_merges = self.vocab_size - len(initial_vocab) - len(self.special_tokens)
        num_merges = max(0, num_merges)  # Ensure non-negative
        
        # Perform merges
        self.merges = []
        for _ in range(num_merges):
            # Get pair frequencies
            stats = self.get_stats(tokens_list)
            if not stats:
                break
                
            # Find most frequent pair
            most_common = stats.most_common(1)[0][0]
            merged_token = most_common[0] + most_common[1]
            
            # Add to merges list
            self.merges.append(most_common)
            
            # Apply merge
            tokens_list = self.merge_pair(tokens_list, most_common, merged_token)
        
        # Build final vocabulary
        self.vocab = self.build_vocab(tokens_list)
        
        # Clear encoding cache
        self.word_cache = {}
    
    def _apply_merges_to_word(self, word: str) -> List[str]:
        """
        Apply all learned merges to a single word.
        
        Args:
            word: Input word
            
        Returns:
            List of tokens after applying merges
        """
        # Check cache first
        if word in self.word_cache:
            return self.word_cache[word]
            
        # Handle space as a special case
        if word == ' ':
            return [' ']
            
        # Start with characters
        tokens = list(word) + ["</w>"]
        
        # Apply each merge in order
        for pair in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                    tokens = tokens[:i] + [pair[0] + pair[1]] + tokens[i+2:]
                    i = 0  # Restart from beginning to catch overlapping merges
                else:
                    i += 1
        
        # Cache the result
        self.word_cache[word] = tokens
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        tokens = []
        buffer = ""
        
        # Process each character, handling words and spaces
        for char in text + ' ':  # Add space to process the last word
            if char == ' ':
                if buffer:
                    # Process the accumulated word
                    word_tokens = self._apply_merges_to_word(buffer)
                    tokens.extend(word_tokens)
                    buffer = ""
                tokens.append(' ')
            else:
                buffer += char
        
        # Remove the last added space
        if tokens and tokens[-1] == ' ':
            tokens.pop()
        
        # Convert tokens to IDs, handling unknown tokens
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab["<unk>"]))
        
        return ids
    
    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            ids: List of token IDs
            
        Returns:
            Decoded text
        """
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab.get(idx, "<unk>") for idx in ids]
        
        text = ""
        for token in tokens:
            if token in self.special_tokens:
                continue
            elif token == ' ':
                text += ' '
            else:
                # Remove end-of-word token if present
                token = token.replace("</w>", "")
                text += token
        
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
            "vocab": self.vocab,
            "merges": self.merges
        }
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'BPETokenizer':
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
        
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        
        return tokenizer


