# Rewriting the `UniformPairTokenizer` to match the structure of `BPETokenizer` but with uniform sampling.
# We'll include train, encode, decode, merge logic, and caching, just like in BPETokenizer.
import re
import json
import random
from typing import List, Dict, Tuple, Optional
from collections import Counter

class UniformPairTokenizer:
    def __init__(self, vocab_size: int = 10000, special_tokens: Optional[List[str]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<unk>", "<pad>", "<bos>", "<eos>", "<mask>"]
        self.vocab: Dict[str, int] = {}
        self.merges: List[Tuple[str, str]] = []
        self.word_cache: Dict[str, List[str]] = {}

    def _tokenize_text(self, text: str) -> List[List[str]]:
        text = re.sub(r'\\s+', ' ', text)
        tokens_list = []
        current_word = []

        for char in text:
            if char == ' ':
                if current_word:
                    current_word.append("</w>")
                    tokens_list.append(current_word)
                    current_word = []
                tokens_list.append([' '])
            else:
                current_word.append(char)

        if current_word:
            current_word.append("</w>")
            tokens_list.append(current_word)

        return tokens_list

    def get_pair_candidates(self, tokens_list: List[List[str]]) -> List[Tuple[str, str]]:
        pairs = Counter()
        for tokens in tokens_list:
            for i in range(len(tokens) - 1):
                pairs[(tokens[i], tokens[i + 1])] += 1
        return list(pairs.keys())

    def merge_pair(self, tokens_list: List[List[str]], pair: Tuple[str, str], merged_token: str) -> List[List[str]]:
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
        all_tokens = set()
        for tokens in tokens_list:
            all_tokens.update(tokens)
        vocab = {token: i for i, token in enumerate(self.special_tokens)}
        i = len(vocab)
        for token in sorted(all_tokens):
            if token not in vocab:
                vocab[token] = i
                i += 1
                if i >= self.vocab_size:
                    break
        return vocab

    def train(self, text: str) -> None:
        tokens_list = self._tokenize_text(text)
        initial_vocab = set(token for tokens in tokens_list for token in tokens)
        num_merges = self.vocab_size - len(initial_vocab) - len(self.special_tokens)
        num_merges = max(0, num_merges)
        self.merges = []

        for _ in range(num_merges):
            pairs = self.get_pair_candidates(tokens_list)
            if not pairs:
                break
            pair = random.choice(pairs)
            merged_token = pair[0] + pair[1]
            self.merges.append(pair)
            tokens_list = self.merge_pair(tokens_list, pair, merged_token)

        self.vocab = self.build_vocab(tokens_list)
        self.word_cache = {}

    def _apply_merges_to_word(self, word: str) -> List[str]:
        if word in self.word_cache:
            return self.word_cache[word]
        if word == ' ':
            return [' ']
        tokens = list(word) + ["</w>"]
        for pair in self.merges:
            merged = pair[0] + pair[1]
            j = 0
            while j < len(tokens) - 1:
                if tokens[j] == pair[0] and tokens[j + 1] == pair[1]:
                    tokens = tokens[:j] + [merged] + tokens[j + 2:]
                    j = 0
                else:
                    j += 1
        self.word_cache[word] = tokens
        return tokens

    def encode(self, text: str) -> List[int]:
        text = re.sub(r'\\s+', ' ', text.strip())
        tokens = []
        buffer = ""
        for char in text + ' ':
            if char == ' ':
                if buffer:
                    word_tokens = self._apply_merges_to_word(buffer)
                    tokens.extend(word_tokens)
                    buffer = ""
                tokens.append(' ')
            else:
                buffer += char
        if tokens and tokens[-1] == ' ':
            tokens.pop()
        ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        return ids

    def decode(self, ids: List[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab.get(idx, "<unk>") for idx in ids]
        text = ""
        for token in tokens:
            if token in self.special_tokens:
                continue
            elif token == ' ':
                text += ' '
            else:
                token = token.replace("</w>", "")
                text += token
        return text

    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "vocab": self.vocab,
            "merges": self.merges
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> 'UniformPairTokenizer':
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tokenizer = cls(
            vocab_size=data["vocab_size"],
            special_tokens=data["special_tokens"]
        )
        tokenizer.vocab = data["vocab"]
        tokenizer.merges = [tuple(m) for m in data["merges"]]
        return tokenizer