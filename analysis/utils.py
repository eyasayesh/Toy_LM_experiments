from collections import Counter
import numpy as np
import random
from scipy.special import digamma
from typing import List, Tuple, Union, Optional

from .plot_utils import zipf_plot

def get_zipf(tokenizer, text: str, plot: bool = False, tokenizer_name: str = "", out_dir: str = "figures") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Zipf's law statistics for token frequencies.
    
    Args:
        tokenizer: Tokenizer object with encode method
        text: Input text to tokenize
        plot: Whether to generate a plot
        tokenizer_name: Name of the tokenizer for plot title
        out_dir: Directory to save the plot
        
    Returns:
        Tuple of (token_ids, ranks, frequencies)
    """
    tokens = tokenizer.encode(text)
    token_freq = Counter(tokens)

    sorted_token_freqs = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
    sorted_token_ids, sorted_freqs = zip(*sorted_token_freqs)

    ranks = np.arange(1, len(sorted_freqs) + 1)
    freqs = np.array(sorted_freqs)
    token_ids = np.array(sorted_token_ids)

    if plot:
        zipf_plot(ranks, freqs, tokenizer_name, out_dir=out_dir)

    return token_ids, ranks, freqs

def grassberger_entropy(counts: Counter) -> float:
    """
    Grassberger entropy estimator for a histogram.
    
    Args:
        counts: Counter object with counts
        
    Returns:
        Estimated entropy
    """
    N = sum(counts.values())
    if N == 0:
        return 0.0
    entropy = np.log(N) - (1.0 / N) * sum(n * digamma(n) for n in counts.values())
    return entropy

def estimate_gb_mutual_information(tokens: List[int], d_values: List[int], num_samples: int = 10000) -> List[Tuple[int, float]]:
    """
    Estimate mutual information between tokens at different distances using Grassberger estimator.
    
    Args:
        tokens: List of token IDs
        d_values: List of distances to compute MI for
        num_samples: Number of samples to use for estimation
        
    Returns:
        List of (distance, mutual_information) tuples
    """
    if not tokens:
        raise ValueError("Empty token list")
    if not d_values:
        raise ValueError("Empty distance list")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
        
    results = []
    max_d = max(d_values)
    
    if max_d >= len(tokens):
        raise ValueError(f"Maximum distance {max_d} exceeds token sequence length {len(tokens)}")

    # Pre-compute all possible pairs for efficiency
    all_pairs = [(tokens[i], tokens[i + d]) for d in d_values for i in range(len(tokens) - d)]
    
    for d in d_values:
        # Filter pairs for current distance
        d_pairs = [(x, y) for i, (x, y) in enumerate(all_pairs) if i % len(d_values) == d_values.index(d)]
        
        # Sample if needed
        if len(d_pairs) > num_samples:
            d_pairs = random.sample(d_pairs, num_samples)
            
        # Build histograms
        joint_counts = Counter(d_pairs)
        x_counts = Counter(x for x, _ in d_pairs)
        y_counts = Counter(y for _, y in d_pairs)

        # Estimate entropies
        H_x = grassberger_entropy(x_counts)
        H_y = grassberger_entropy(y_counts)
        H_xy = grassberger_entropy(joint_counts)

        # Mutual Information
        I_xy = H_x + H_y - H_xy
        results.append((d, I_xy))

    return results

def compute_mutual_information(tokens, max_distance=100, top_k=500):
    print("Token count:", len(tokens))
    token_counts = Counter(tokens)
    total = sum(token_counts.values())

    # Restrict to top-K tokens for efficiency
    top_tokens = set([tok for tok, _ in token_counts.most_common(top_k)])

    px = {tok: count / total for tok, count in token_counts.items() if tok in top_tokens}

    mi_by_distance = []

    for d in range(1, max_distance + 1):
        pair_counts = Counter()
        valid = 0

        for i in range(len(tokens) - d):
            x, y = tokens[i], tokens[i + d]
            if x in top_tokens and y in top_tokens:
                pair_counts[(x, y)] += 1
                valid += 1

        if valid == 0:
            mi_by_distance.append(0)
            continue

        mi = 0.0
        for (x, y), count in pair_counts.items():
            pxy = count / valid
            if pxy == 0 or px[x] == 0 or px[y] == 0:
                continue
            mi += pxy * np.log2(pxy / (px[x] * px[y]))

        mi_by_distance.append(mi)

    return mi_by_distance