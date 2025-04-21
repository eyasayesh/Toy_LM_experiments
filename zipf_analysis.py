import os
import pickle
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
SAVE_DIR = "figures/ood_subset_zipf"
PICKLE_PATH = "results/ood_subset_zipf_data.pkl"

# Create output directory if needed
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(PICKLE_PATH), exist_ok=True)

# Load text
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()[5_000_000:8_000_000]

# Dictionary to store Zipf data
zipf_data = {}

# Analyze all tokenizers
for file_name in os.listdir(TOKENIZER_DIR):
    if not file_name.endswith(".json"):
        continue

    tokenizer_path = os.path.join(TOKENIZER_DIR, file_name)
    tokenizer_key = file_name.split("_")[0].lower()

    if tokenizer_key not in TOKENIZER_MAP:
        print(f"Skipping unknown tokenizer type in file: {file_name}")
        continue

    print(f"\nProcessing {file_name}...")

    tokenizer_class = TOKENIZER_MAP[tokenizer_key]
    tokenizer = tokenizer_class.load(tokenizer_path)

    try:
        token_ids, ranks, freqs = get_zipf(tokenizer, text)
        zipf_data[file_name] = {
            "token_ids": token_ids,
            "ranks": ranks,
            "freqs": freqs
        }

        title = file_name.replace(".json", "")
        zipf_plot(ranks, freqs, title, SAVE_DIR)
        print(f"✅ Saved Zipf plot for {title}")

    except Exception as e:
        print(f"❌ Failed on {file_name} due to: {e}")

# Save Zipf data
with open(PICKLE_PATH, "wb") as f:
    pickle.dump(zipf_data, f)

print(f"\n📦 All Zipf data saved to {PICKLE_PATH}")
