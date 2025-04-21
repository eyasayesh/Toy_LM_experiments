import argparse
import os
import sys
import torch
import torch.multiprocessing as mp

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ddp_trainer import train

# Tokenizers
from custom_tokenizers import (
    CharTokenizer,
    WordTokenizer,
    BPETokenizer,
    UniformPairTokenizer,
    FreqWeightedTokenizer
)

def get_tokenizer(tokenizer_type, tokenizer_loc):
    if tokenizer_type == "char":
        tokenizer = CharTokenizer()
    elif tokenizer_type == "word":
        tokenizer = WordTokenizer()
    elif tokenizer_type == "bpe":
        tokenizer = BPETokenizer()
    elif tokenizer_type == "uniform":
        tokenizer = UniformPairTokenizer()
    elif tokenizer_type == "freq":
        tokenizer = FreqWeightedTokenizer()
    else:
        raise ValueError(f"Unknown tokenizer type: {tokenizer_type}")
    
    # Use the provided tokenizer_loc directly
    tokenizer.load(tokenizer_loc)
    
    # Get vocabulary size based on tokenizer type
    if tokenizer_type == "char":
        vocab_size = len(tokenizer.char2id)
    else:
        vocab_size = len(tokenizer.vocab)
        
    return tokenizer.encode, vocab_size

def main():
    # Print CUDA and GPU information
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.get_device_name(0)}")
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=str, required=True,
                        choices=["char", "word", "bpe", "uniform", "freq"],
                        help="Type of tokenizer to use")
    parser.add_argument("--tokenizer_loc", type=str, required=True,
                        help="Full path to the tokenizer file")
    parser.add_argument("--data_path", type=str, default="data/text8.txt")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--tokenizer_name", type=str, default="default")
    args = parser.parse_args()

    with open(args.data_path, "r") as f:
        text = f.read()[:100_000]  # Optional truncation for faster testing

    tokenizer_fn, vocab_size = get_tokenizer(args.tokenizer, args.tokenizer_loc)

    world_size = torch.cuda.device_count()

    print(f"Launching DDP training on {world_size} GPUs using '{args.tokenizer}' tokenizer")

    mp.spawn(
        train,
        args=(
            world_size,
            text,
            tokenizer_fn,
            vocab_size,
            args.seq_len,
            args.batch_size,
            args.epochs,
            args.lr,
            args.tokenizer_name  # <== pass tokenizer name
        ),
        nprocs=world_size
    )

if __name__ == "__main__":
    main()