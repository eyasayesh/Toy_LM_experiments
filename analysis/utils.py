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

    name = tokenizer_name.lower()
    if name == "char":
        mask = np.ones_like(token_ids)
    elif name == "word":
        mask = token_ids != 1
    else:
        mask = token_ids >= 5

    if plot:
        zipf_plot(ranks[mask], freqs[mask], tokenizer_name, out_dir=out_dir)

    return token_ids[mask], ranks[mask], freqs[mask]
import numpy as np
from collections import Counter
from scipy.special import digamma

def grassberger_entropy(counts):
    """Grassberger entropy estimator."""
    N = sum(counts.values())
    if N == 0:
        return 0.0
    return np.log(N) - (1.0 / N) * sum(n * digamma(n) for n in counts.values())

def compute_mutual_information(tokens, max_distance=10, num_samples=None):
    """
    Compute MI decay for distances d = 1 to max_distance.
    Args:
        tokens (List[int]): Tokenized sequence.
        max_distance (int): Max separation distance.
        num_samples (int or None): Subsample size for efficiency.
    Returns:
        List[float]: Mutual information for each distance.
    """
    N = len(tokens)
    if num_samples is None:
        num_samples = N

    tokens = np.array(tokens)
    mi_results = []

    for d in range(1, max_distance + 1):
        if d >= N:
            break

        indices = np.random.choice(N - d, min(num_samples, N - d), replace=False)
        x = tokens[indices]
        y = tokens[indices + d]

        pairs = list(zip(x, y))
        joint_counts = Counter(pairs)
        x_counts = Counter(x)
        y_counts = Counter(y)

        H_x = grassberger_entropy(x_counts)
        H_y = grassberger_entropy(y_counts)
        H_xy = grassberger_entropy(joint_counts)
        mi = H_x + H_y - H_xy
        mi_results.append(mi)

    return mi_results
