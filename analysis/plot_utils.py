import matplotlib.pyplot as plt
import os
import numpy as np

def zipf_plot(ranks, freqs, tokenizer_name, out_dir):
    plt.figure(figsize=(8, 6))
    plt.loglog(ranks, freqs, marker='.')
    plt.title(f"Zipf Plot - {tokenizer_name}")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.grid(True, which="both", linestyle="--")
    plt.savefig(os.path.join(out_dir, f"{tokenizer_name}_zipf.png"))
    plt.close()

def mutual_info_plot(distances, mi_values, tokenizer_name, out_dir):
    """
    Plot mutual information as a function of distance.
    
    Args:
        distances (list): List of distances (in tokens)
        mi_values (list): List of mutual information values
        tokenizer_name (str): Name of the tokenizer
        out_dir (str): Directory to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(distances, mi_values, marker='o', linestyle='-')
    plt.title(f"Mutual Information vs Distance - {tokenizer_name}")
    plt.xlabel("Distance (tokens)")
    plt.ylabel("Mutual Information")
    plt.grid(True, linestyle="--")
    plt.savefig(os.path.join(out_dir, f"{tokenizer_name}_mutual_info.png"))
    plt.close()