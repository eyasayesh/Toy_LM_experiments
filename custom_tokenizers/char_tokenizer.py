"""Character level tokenizer"""
import json

class CharTokenizer:
    def __init__(self):
        self.char2id = {}
        self.id2char = {}

    def train(self, text: str):
        unique_chars = sorted(set(text))
        self.char2id = {c: i for i, c in enumerate(unique_chars)}
        self.id2char = {i: c for c, i in self.char2id.items()}

    def encode(self, text: str) -> list[int]:
        return [self.char2id.get(c, -1) for c in text if c in self.char2id]

    def decode(self, ids: list[int]) -> str:
        return ''.join([self.id2char.get(i, '') for i in ids if i in self.id2char])

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump({"char2id": self.char2id}, f)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.char2id = data["char2id"]
        tok.id2char = {int(i): c for c, i in tok.char2id.items()}
        return tok
