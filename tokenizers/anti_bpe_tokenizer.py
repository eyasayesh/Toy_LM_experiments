"""Follows BPE tokenizer procedure, 
    but takes the rarest pair to
    generate new tokens"""

import re, json
from collections import Counter

class AntiBPETokenizer:
    SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.merges = []
        self.vocab = {}

    def get_pair_frequencies(self, tokens):
        return Counter(zip(tokens, tokens[1:]))

    def train(self, text: str):
        tokens = list(text)
        initial_vocab = set(tokens)
        num_merges = self.vocab_size - len(initial_vocab) - len(self.SPECIAL_TOKENS)

        for i in range(num_merges):
            pair_freq = self.get_pair_frequencies(tokens)
            if not pair_freq:
                break

            # Get rarest (least frequent) pair
            rarest_pair = min(pair_freq.items(), key=lambda x: x[1])[0]
            a, b = rarest_pair
            merged = f"{a}{b}@{i}"
            self.merges.append((a, b))
            text = text.replace(a + b, merged)
            tokens = list(text)

        # Build final vocab
        #this is important to turn the new "tokens" into actual characters
        unique_tokens = set(re.findall(r'\S+', text))
        all_tokens = self.SPECIAL_TOKENS + sorted(unique_tokens - set(self.SPECIAL_TOKENS))
        self.vocab = {tok: i for i, tok in enumerate(all_tokens)}

    def encode(self, text: str) -> list[int]:
        for i, (a, b) in enumerate(self.merges):
            merged = f"{a}{b}@{i}"
            text = text.replace(a + b, merged)
        tokens = re.findall(r'\S+', text)
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]

    def decode(self, ids: list[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        return ' '.join(inv_vocab[i] for i in ids if inv_vocab[i] not in {"<bos>", "<eos>", "<pad>"} )

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
