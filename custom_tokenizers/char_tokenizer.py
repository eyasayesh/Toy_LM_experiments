"""Character level tokenizer"""
import json

class CharTokenizer:
    SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]

    def __init__(self):
        self.char2id = {}
        self.id2char = {}

    def train(self, text: str):
        unique_chars = sorted(set(text))
        all_chars = self.SPECIAL_TOKENS + unique_chars
        self.char2id = {c: i for i, c in enumerate(all_chars)}
        self.id2char = {i: c for c, i in self.char2id.items()}

    def encode(self, text: str, add_special_tokens=True) -> list[int]:
        ids = [self.char2id.get(c, self.char2id["<unk>"]) for c in text if c in self.char2id]
        if add_special_tokens:
            return [self.char2id["<bos>"]] + ids + [self.char2id["<eos>"]]
        return ids

    def decode(self, ids: list[int]) -> str:
        return ''.join([self.id2char.get(i, '') for i in ids if self.id2char.get(i, '') not in self.SPECIAL_TOKENS])

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
