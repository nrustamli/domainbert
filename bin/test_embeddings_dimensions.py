#!/usr/bin/env python3
"""
Test embedding dimensionality and information content.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib to non-interactive backend (no display)
plt.ioff()

def load_embeddings():
    """Load your embeddings."""
    embeddings = np.load("results/embeddings.npy")
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings

def dimensionality_analysis(embeddings):
    """Analyze embedding dimensionality and information content."""
    
    print("Analyzing embedding dimensionality...")
    
    # PCA analysis
    print("  Performing PCA analysis...")
    pca = PCA()
    pca.fit(embeddings)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Find number of components for different variance thresholds
    thresholds = [0.5, 0.8, 0.9, 0.95, 0.99]
    components_needed = {}
    
    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        components_needed[threshold] = n_components
        print(f"    {threshold*100}% variance: {n_components} components")
    
    # Plot explained variance (save only, no display)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scree plot
    ax1.plot(range(1, len(explained_variance) + 1), explained_variance, 'bo-')
    ax1.set_xlabel('Principal Component')
    ax1.set_ylabel('Explained Variance Ratio')
    ax1.set_title('Scree Plot: Explained Variance by Component')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative variance
    ax2.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'ro-')
    for threshold in thresholds:
        ax2.axhline(y=threshold, color='g', linestyle='--', alpha=0.7)
        ax2.axvline(x=components_needed[threshold], color='g', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title('Cumulative Explained Variance')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/dimensionality_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"  ✓ Saved dimensionality analysis plot to: figures/dimensionality_analysis.png")
    
    return pca, components_needed

def visualize_embeddings_2d(embeddings, method='pca'):
    """Visualize embeddings in 2D."""
    
    print(f"  Creating 2D visualization using {method.upper()}...")
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        title = "PCA Visualization of Embeddings"
    elif method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        title = "t-SNE Visualization of Embeddings"
    
    embeddings_2d = reducer.fit_transform(embeddings)
    
    # Create scatter plot (save only, no display)
    plt.figure(figsize=(12, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=1)
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"figures/embeddings_2d_{method}.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"    ✓ Saved {method.upper()} visualization to: figures/embeddings_2d_{method}.png")
    
    return embeddings_2d

def main():
    print("="*60)
    print("EMBEDDING DIMENSIONALITY ANALYSIS")
    print("="*60)
    
    embeddings = load_embeddings()
    
    # Analyze dimensionality
    pca, components_needed = dimensionality_analysis(embeddings)
    
    # Visualize in 2D
    print("\nCreating 2D visualizations...")
    embeddings_2d_pca = visualize_embeddings_2d(embeddings, 'pca')
    
    # Only run t-SNE if dataset is not too large
    if len(embeddings) <= 10000:
        embeddings_2d_tsne = visualize_embeddings_2d(embeddings, 'tsne')
    else:
        print("  Skipping t-SNE (dataset too large)")
    
    print(f"\n{'='*60}")
    print("DIMENSIONALITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Original dimensions: {embeddings.shape[1]}")
    print(f"Total variance explained by first 10 components: {sum(pca.explained_variance_ratio_[:10])*100:.1f}%")
    print(f"✓ All figures saved to figures/ directory")

if __name__ == "__main__":
    main() 