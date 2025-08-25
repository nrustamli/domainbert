#!/usr/bin/env python3
"""
Test embedding similarity distributions and patterns.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib to non-interactive backend (no display)
plt.ioff()

def load_embeddings():
    """Load your embeddings."""
    embeddings = np.load("results/embeddings.npy")
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings

def similarity_analysis(embeddings, sample_size=1000):
    """Analyze similarity distributions."""
    
    print("Analyzing similarity distributions...")
    
    # Sample proteins for analysis (to avoid memory issues)
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        sample_embeddings = embeddings[indices]
        print(f"Using sample of {sample_size} proteins for analysis")
    else:
        sample_embeddings = embeddings
        indices = np.arange(len(embeddings))
    
    # Calculate cosine similarities
    print("  Computing cosine similarities...")
    cos_sim = cosine_similarity(sample_embeddings)
    
    # Calculate Euclidean distances
    print("  Computing Euclidean distances...")
    euc_dist = euclidean_distances(sample_embeddings)
    
    # Analyze distributions
    cos_sim_flat = cos_sim[np.triu_indices_from(cos_sim, k=1)]
    euc_dist_flat = euc_dist[np.triu_indices_from(euc_dist, k=1)]
    
    print(f"  Cosine similarity range: {cos_sim_flat.min():.3f} to {cos_sim_flat.max():.3f}")
    print(f"  Euclidean distance range: {euc_dist_flat.min():.3f} to {euc_dist_flat.max():.3f}")
    
    # Plot distributions (save only, no display)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Cosine similarity distribution
    ax1.hist(cos_sim_flat, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Cosine Similarity')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Cosine Similarities')
    ax1.grid(True, alpha=0.3)
    
    # Euclidean distance distribution
    ax2.hist(euc_dist_flat, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Euclidean Distance')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Distribution of Euclidean Distances')
    ax2.grid(True, alpha=0.3)
    
    # Similarity matrix heatmap (small sample)
    if sample_size <= 100:
        sns.heatmap(cos_sim[:100, :100], ax=ax3, cmap='viridis', 
                   xticklabels=False, yticklabels=False)
        ax3.set_title('Cosine Similarity Matrix (100x100)')
    
    # Distance vs Similarity correlation
    ax4.scatter(cos_sim_flat, euc_dist_flat, alpha=0.5, s=1)
    ax4.set_xlabel('Cosine Similarity')
    ax4.set_ylabel('Euclidean Distance')
    ax4.set_title('Cosine Similarity vs Euclidean Distance')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/similarity_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"  ✓ Saved similarity analysis plot to: figures/similarity_analysis.png")
    
    return cos_sim_flat, euc_dist_flat

def main():
    print("="*60)
    print("EMBEDDING SIMILARITY ANALYSIS")
    print("="*60)
    
    embeddings = load_embeddings()
    cos_sim, euc_dist = similarity_analysis(embeddings)
    
    print(f"\n{'='*60}")
    print("SIMILARITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Analyzed {len(cos_sim)} protein pairs")
    print(f"Mean cosine similarity: {np.mean(cos_sim):.3f}")
    print(f"Mean Euclidean distance: {np.mean(euc_dist):.3f}")
    print(f"✓ All figures saved to figures/ directory")

if __name__ == "__main__":
    main() 