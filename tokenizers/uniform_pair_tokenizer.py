"""Follows BPE tokenizer procedure, 
    but samples uniformly over all the 
    pairs to generate new tokens"""

import re, json, random
from collections import Counter

class UniformPairTokenizer:
    SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]

    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}

    def train(self, text: str):
        tokens = list(text)
        initial_vocab = set(tokens)
        num_merges = self.vocab_size - len(initial_vocab) - len(self.SPECIAL_TOKENS)

        for i in range(num_merges):
            pairs = set(zip(tokens, tokens[1:]))
            if not pairs:
                break
            #uniform sampling
            a, b = random.choice(list(pairs))
            merged = f"{a}{b}@{i}"
            self.merges.append((a, b))
            text = text.replace(a + b, merged)
            tokens = list(text)

        #this is important to turn the new "tokens" into actual characters
        unique_tokens = set(re.findall(r'\S+', text))
        all_tokens = self.SPECIAL_TOKENS + sorted(unique_tokens - set(self.SPECIAL_TOKENS))
        self.vocab = {tok: i for i, tok in enumerate(all_tokens)}


    def encode(self, text: str) -> list[int]:
        for i, (a, b) in enumerate(self.merges):
            merged = f"{a}{b}@{i}"
            text = text.replace(a + b, merged)
        tokens = re.findall(r'\S+', text)
        return [self.vocab.get(t, 0) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join(inv_vocab[i] for i in ids)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "merges": self.merges,
                "vocab": self.vocab,
                "vocab_size": self.vocab_size
            }, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.vocab = data["vocab"]
        return tok