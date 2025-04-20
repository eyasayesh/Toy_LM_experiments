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
        tokens = list(text)  # Initial tokens: characters
        initial_vocab = set(tokens)
        num_merges = self.vocab_size - len(initial_vocab) - len(self.SPECIAL_TOKENS)

        for i in range(num_merges):
            pair_freq = self.get_pair_frequencies(tokens)
            if not pair_freq:
                break

            # Get rarest (least frequent) pair
            rarest_pair = min(pair_freq.items(), key=lambda x: x[1])[0]
            a, b = rarest_pair
            merged_token = f"{a}{b}@{i}"
            self.merges.append((a, b))

            # Replace all occurrences of the rarest pair with the merged token
            new_tokens = []
            skip = False
            for j in range(len(tokens) - 1):
                if skip:
                    skip = False
                    continue
                if tokens[j] == a and tokens[j + 1] == b:
                    new_tokens.append(merged_token)
                    skip = True
                else:
                    new_tokens.append(tokens[j])
            if not skip:
                new_tokens.append(tokens[-1])  # Add last token if not merged
            tokens = new_tokens

        # Final vocab from token list
        unique_tokens = set(tokens)
        all_tokens = self.SPECIAL_TOKENS + sorted(unique_tokens - set(self.SPECIAL_TOKENS))
        self.vocab = {tok: i for i, tok in enumerate(all_tokens)}

    def encode(self, text: str) -> list[int]:
        tokens = list(text)
        for i, (a, b) in enumerate(self.merges):
            merged = f"{a}{b}@{i}"
            new_tokens = []
            skip = False
            for j in range(len(tokens) - 1):
                if skip:
                    skip = False
                    continue
                if tokens[j] == a and tokens[j + 1] == b:
                    new_tokens.append(merged)
                    skip = True
                else:
                    new_tokens.append(tokens[j])
            if not skip:
                new_tokens.append(tokens[-1])
            tokens = new_tokens
        return [self.vocab.get(tok, self.vocab["<unk>"]) for tok in tokens]

    def decode(self, ids: list[int]) -> str:
        inv_vocab = {v: k for k, v in self.vocab.items()}
        tokens = [inv_vocab[i] for i in ids if inv_vocab[i] not in {"<bos>", "<eos>", "<pad>"}]
        # Strip merge markers to get the original characters
        text = ''.join(re.sub(r'@\d+', '', tok) for tok in tokens)
        return text

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

