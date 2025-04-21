import os
from time import time
from analysis import estimate_gb_mutual_information, compute_mutual_information, mutual_info_plot
from custom_tokenizers import (
    CharTokenizer, WordTokenizer, BPETokenizer,
    UniformPairTokenizer, FreqWeightedTokenizer
)

TOKENIZER_MAP = {
    "char": CharTokenizer,
    #"word": WordTokenizer,
    #"bpe": BPETokenizer,
    #"uniform": UniformPairTokenizer,
    #"freq": FreqWeightedTokenizer,
    # "anti_bpe": AntiBPETokenizer  # optional
}

TOKENIZER_DIR = "trained_tokenizers"
TEXT_FILE = "data/text8_subset.txt"
SAVE_DIR = "figures/mi_curves"
os.makedirs(SAVE_DIR, exist_ok=True)


# Load text
with open(TEXT_FILE, "r", encoding="utf-8") as f:
    text = f.read()

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
        tokens = tokenizer.encode(text)
        print(f"Tokenized length: {len(tokens)}")

        # Compute mutual information using both methods
        d_vals = list(range(1, 101))
        
        """ # Grassberger MI
        start_time = time()
        mi_grassberger = estimate_gb_mutual_information(tokens, d_vals)
        d_gb, mi_gb = zip(*mi_grassberger)
        print("GB duration is ", time()-start_time, " seconds")"""


        # Top-K filtered MI
        start_time = time()
        d_topk, mi_topk = compute_mutual_information(tokens, max_distance=100, top_k=500)
        print("compute_mutual_information duration is ", time()-start_time ," seconds")

        # Plot both methods using the built-in function
        tokenizer_name = file_name.replace('.json', '')
        #mutual_info_plot(d_gb, mi_gb, tokenizer_name, SAVE_DIR)
        mutual_info_plot(d_topk, mi_topk, f"{tokenizer_name}_topk", SAVE_DIR)

        print(f"✅ Saved MI plots for {tokenizer_name}")

    except Exception as e:
        print(f"❌ Failed on {file_name}: {e}")
