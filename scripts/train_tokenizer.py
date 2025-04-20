# scripts/train_tokenizer.py

import argparse
import os

from custom_tokenizers import (
    CharTokenizer,
    WordTokenizer,
    BPETokenizer,
    UniformPairTokenizer,
    FreqWeightedTokenizer,
    AntiBPETokenizer,
)

def get_tokenizer_class(name):
    name = name.lower()
    if name == "char":
        return CharTokenizer
    elif name == "word":
        return WordTokenizer
    elif name == "bpe":
        return BPETokenizer
    elif name == "uniform":
        return UniformPairTokenizer
    elif name == "freq":
        return FreqWeightedTokenizer
    elif name == "anti_bpe":
        return AntiBPETokenizer
    else:
        raise ValueError(f"Unknown tokenizer type: {name}")

def main(args):
    assert os.path.exists(args.input), f"Input file {args.input} does not exist."

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    TokenizerClass = get_tokenizer_class(args.tokenizer)

    if args.tokenizer == "char":
        tokenizer = TokenizerClass()
    else:
        assert args.vocab_size is not None, f"--vocab_size is required for tokenizer type '{args.tokenizer}'"
        tokenizer = TokenizerClass(vocab_size=args.vocab_size)

    tokenizer.train(text)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tokenizer.save(args.output)

    print(f"Trained {args.tokenizer} tokenizer "
          f"{'' if args.tokenizer == 'char' else f'with vocab size {args.vocab_size} '}and saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a tokenizer on a text corpus")
    parser.add_argument("--tokenizer", type=str, required=True,
                        choices=["char", "word", "bpe", "uniform", "freq", "anti_bpe"],
                        help="Tokenizer type to train")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input text file")
    parser.add_argument("--vocab_size", type=int, required=False,
                        help="Target vocabulary size (required for non-char tokenizers)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to save trained tokenizer")

    args = parser.parse_args()

    # Enforce vocab_size presence for all but char tokenizer
    if args.tokenizer != "char":
        assert args.vocab_size is not None, "--vocab_size must be provided for non-char tokenizers"

    main(args)
