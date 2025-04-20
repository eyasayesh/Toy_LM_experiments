from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import os

class BPETokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.tokenizer = Tokenizer(models.BPE())
        self.trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def train(self, input_path: str):
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"{input_path} does not exist.")
        self.tokenizer.train([input_path], self.trainer)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        self.tokenizer.save(path)

    @classmethod
    def load(cls, path: str):
        tokenizer = Tokenizer.from_file(path)
        bpe = cls()
        bpe.tokenizer = tokenizer
        return bpe
