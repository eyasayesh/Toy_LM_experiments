import os
from time import time
from analysis import get_zipf, zipf_plot
from custom_tokenizers import (
    CharTokenizer, WordTokenizer, BPETokenizer,
    UniformPairTokenizer, FreqWeightedTokenizer
)

# Optional: import AntiBPETokenizer if you're using that
# from custom_tokenizers import AntiBPETokenizer

TOKENIZER_MAP = {
    "char": CharTokenizer,
    "word": WordTokenizer,
    "bpe": BPETokenizer,
    "uniform": UniformPairTokenizer,
    "freq": FreqWeightedTokenizer,
    # "anti_bpe": AntiBPETokenizer,  # Add if needed
}

TOKENIZER_DIR = "trained_tokenizers"
TEXT_FILE = "data/text8"
SAVE_DIR = "figures"

# Load text
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

# Analyze all tokenizers
for file_name in os.listdir(TOKENIZER_DIR):
    if not file_name.endswith(".json"):
        continue

    tokenizer_path = os.path.join(TOKENIZER_DIR, file_name)

    # Extract tokenizer type prefix (e.g., "char", "word", "bpe", etc.)
    tokenizer_key = file_name.split("_")[0].lower()

    if tokenizer_key not in TOKENIZER_MAP:
        print(f"Skipping unknown tokenizer type in file: {file_name}")
        continue

    print(f"\nProcessing {file_name}...")

    tokenizer_class = TOKENIZER_MAP[tokenizer_key]
    tokenizer = tokenizer_class.load(tokenizer_path)

    try:
        token_ids, ranks, freqs = get_zipf(tokenizer, text)
        title = file_name.replace(".json", "")
        zipf_plot(ranks, freqs, title, SAVE_DIR)
        print(f"✅ Saved Zipf plot for {title}")
    except Exception as e:
        print(f"❌ Failed on {file_name} due to: {e}")
