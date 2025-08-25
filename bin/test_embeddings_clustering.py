#!/usr/bin/env python3
"""
Test embedding quality using clustering analysis.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Set matplotlib to non-interactive backend (no display)
plt.ioff()

def load_embeddings():
    """Load your embeddings."""
    # Load embeddings (adjust path as needed)
    embeddings = np.load("results/embeddings.npy")
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings

def clustering_quality_test(embeddings, max_clusters=20):
    """Test clustering quality with different numbers of clusters."""
    
    # Reduce dimensions for visualization
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Test different numbers of clusters
    n_clusters_range = range(2, max_clusters + 1)
    silhouette_scores = []
    calinski_scores = []
    
    print("Testing clustering quality...")
    
    for n_clusters in n_clusters_range:
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        sil_score = silhouette_score(embeddings, cluster_labels)
        cal_score = calinski_harabasz_score(embeddings, cluster_labels)
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        
        print(f"  {n_clusters} clusters: Silhouette={sil_score:.3f}, Calinski-Harabasz={cal_score:.0f}")
    
    # Find optimal number of clusters
    optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    
    # Plot results (save only, no display)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette score plot
    ax1.plot(n_clusters_range, silhouette_scores, 'bo-')
    ax1.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Optimal: {optimal_clusters}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title('Silhouette Score vs Number of Clusters')
    ax1.legend()
    ax1.grid(True)
    
    # Calinski-Harabasz score plot
    ax2.plot(n_clusters_range, calinski_scores, 'go-')
    ax2.axvline(x=optimal_clusters, color='r', linestyle='--', label=f'Optimal: {optimal_clusters}')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Calinski-Harabasz Score')
    ax2.set_title('Calinski-Harabasz Score vs Number of Clusters')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("figures/clustering_quality.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"  ✓ Saved clustering quality plot to: figures/clustering_quality.png")
    
    return optimal_clusters, silhouette_scores, calinski_scores

def main():
    print("="*60)
    print("EMBEDDING CLUSTERING QUALITY TEST")
    print("="*60)
    
    # Load embeddings
    embeddings = load_embeddings()
    
    # Test clustering quality
    optimal_clusters, sil_scores, cal_scores = clustering_quality_test(embeddings)
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Optimal number of clusters: {optimal_clusters}")
    print(f"Best Silhouette score: {max(sil_scores):.3f}")
    print(f"Best Calinski-Harabasz score: {max(cal_scores):.0f}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    print(f"Number of proteins: {embeddings.shape[0]}")
    print(f"✓ All figures saved to figures/ directory")

if __name__ == "__main__":
    main() 