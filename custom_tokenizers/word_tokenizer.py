"""Word-level Tokenizer"""

import json
from collections import Counter

class WordTokenizer:
    SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]

    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size
        self.word2id = {}
        self.id2word = {}

    def train(self, text: str):
        words = text.strip().split()
        counter = Counter(words)

        limit = self.vocab_size - len(self.SPECIAL_TOKENS) if self.vocab_size else None
        top_words = [w for w, _ in counter.most_common(limit)]
        all_words = self.SPECIAL_TOKENS + top_words

        self.word2id = {w: i for i, w in enumerate(all_words)}
        self.id2word = {i: w for w, i in self.word2id.items()}

    def encode(self, text: str, add_special_tokens=True) -> list[int]:
        ids = [self.word2id.get(w, self.word2id["<unk>"]) for w in text.strip().split()]
        if add_special_tokens:
            return [self.word2id["<bos>"]] + ids + [self.word2id["<eos>"]]
        return ids

    def decode(self, ids: list[int]) -> str:
        return ' '.join([self.id2word.get(i, '') for i in ids if self.id2word.get(i, '') not in self.SPECIAL_TOKENS])

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "word2id": self.word2id,
                "vocab_size": self.vocab_size
            }, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        tok = cls(vocab_size=data.get("vocab_size"))
        tok.word2id = data["word2id"]
        tok.id2word = {int(i): w for w, i in tok.word2id.items()}
        return tok
