"""Word-level Tokenizer"""

import json
from collections import Counter

class WordTokenizer:
    def __init__(self, vocab_size=None):
        self.vocab_size = vocab_size
        self.word2id = {}
        self.id2word = {}
        # Initialize <unk> token
        self.word2id["<unk>"] = 0
        self.id2word[0] = "<unk>"

    def train(self, text: str):
        words = text.strip().split()
        counter = Counter(words)

        # Reserve one spot for <unk>
        limit = self.vocab_size - 1 if self.vocab_size else None
        top_words = [w for w, _ in counter.most_common(limit)]
        
        # Add words to vocabulary starting from index 1
        for i, word in enumerate(top_words, start=1):
            self.word2id[word] = i
            self.id2word[i] = word

    def encode(self, text: str) -> list[int]:
        return [self.word2id.get(w, 0) for w in text.strip().split()]

    def decode(self, ids: list[int]) -> str:
        return ' '.join([self.id2word.get(i, '') for i in ids if i in self.id2word])

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
