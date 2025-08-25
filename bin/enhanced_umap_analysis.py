#!/usr/bin/env python3
"""
Enhanced UMAP analysis for protein domain embeddings.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

# Set matplotlib to non-interactive backend (no display)
plt.ioff()

def load_embeddings():
    """Load your embeddings."""
    embeddings = np.load("results/embeddings.npy")
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings

def enhanced_umap_analysis(embeddings, n_clusters=10):
    """Enhanced UMAP analysis with multiple clustering methods."""
    
    print("="*60)
    print("ENHANCED UMAP ANALYSIS")
    print("="*60)
    
    # UMAP dimensionality reduction
    print("Performing UMAP dimensionality reduction...")
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    embeddings_2d = umap.fit_transform(embeddings)
    
    print(f"UMAP complete! Reduced from {embeddings.shape[1]}D to 2D")
    
    # Multiple clustering approaches
    clustering_results = {}
    
    # 1. K-means clustering
    print("\n1. K-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(embeddings)
    kmeans_score = silhouette_score(embeddings, kmeans_labels)
    clustering_results['K-means'] = {'labels': kmeans_labels, 'score': kmeans_score}
    print(f"   K-means silhouette score: {kmeans_score:.3f}")
    
    # 2. DBSCAN clustering (density-based)
    print("2. DBSCAN clustering...")
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(embeddings)
    if len(np.unique(dbscan_labels)) > 1:  # At least 2 clusters
        dbscan_score = silhouette_score(embeddings, dbscan_labels)
        clustering_results['DBSCAN'] = {'labels': dbscan_labels, 'score': dbscan_score}
        print(f"   DBSCAN silhouette score: {dbscan_score:.3f}")
    else:
        print("   DBSCAN: Only one cluster found")
    
    # 3. Hierarchical clustering on UMAP space
    print("3. Hierarchical clustering on UMAP space...")
    from sklearn.cluster import AgglomerativeClustering
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(embeddings_2d)  # Use 2D for speed
    hierarchical_score = silhouette_score(embeddings_2d, hierarchical_labels)
    clustering_results['Hierarchical'] = {'labels': hierarchical_labels, 'score': hierarchical_score}
    print(f"   Hierarchical silhouette score: {hierarchical_score:.3f}")
    
    # Find best clustering method
    best_method = max(clustering_results.keys(), 
                     key=lambda x: clustering_results[x]['score'])
    best_labels = clustering_results[best_method]['labels']
    best_score = clustering_results[best_method]['score']
    
    print(f"\nBest clustering method: {best_method} (score: {best_score:.3f})")
    
    # Create comprehensive visualization
    create_enhanced_umap_plots(embeddings_2d, clustering_results, best_method)
    
    # Analyze cluster characteristics
    analyze_cluster_characteristics(embeddings, best_labels, best_method)
    
    return embeddings_2d, clustering_results, best_method

def create_enhanced_umap_plots(embeddings_2d, clustering_results, best_method):
    """Create multiple UMAP visualizations."""
    
    print("\nCreating enhanced UMAP visualizations...")
    
    # Create subplot grid (save only, no display)
    n_methods = len(clustering_results)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # 1. Best clustering method (large plot)
    best_labels = clustering_results[best_method]['labels']
    scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=best_labels, cmap='tab20', alpha=0.7, s=2)
    axes[0].set_title(f'Best Method: {best_method}\nSilhouette Score: {clustering_results[best_method]["score"]:.3f}', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP Component 1')
    axes[0].set_ylabel('UMAP Component 2')
    axes[0].grid(True, alpha=0.3)
    
    # 2. K-means
    if 'K-means' in clustering_results:
        kmeans_labels = clustering_results['K-means']['labels']
        scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=kmeans_labels, cmap='tab20', alpha=0.7, s=2)
        axes[1].set_title(f'K-means Clustering\nScore: {clustering_results["K-means"]["score"]:.3f}')
        axes[1].set_xlabel('UMAP Component 1')
        axes[1].set_ylabel('UMAP Component 2')
        axes[1].grid(True, alpha=0.3)
    
    # 3. DBSCAN (if available)
    if 'DBSCAN' in clustering_results:
        dbscan_labels = clustering_results['DBSCAN']['labels']
        scatter3 = axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=dbscan_labels, cmap='tab20', alpha=0.7, s=2)
        axes[2].set_title(f'DBSCAN Clustering\nScore: {clustering_results["DBSCAN"]["score"]:.3f}')
        axes[2].set_xlabel('UMAP Component 1')
        axes[2].set_ylabel('UMAP Component 2')
        axes[2].grid(True, alpha=0.3)
    
    # 4. Hierarchical
    if 'Hierarchical' in clustering_results:
        hierarchical_labels = clustering_results['Hierarchical']['labels']
        scatter4 = axes[3].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=hierarchical_labels, cmap='tab20', alpha=0.7, s=2)
        axes[3].set_title(f'Hierarchical Clustering\nScore: {clustering_results["Hierarchical"]["score"]:.3f}')
        axes[3].set_xlabel('UMAP Component 1')
        axes[3].set_ylabel('UMAP Component 2')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("figures/enhanced_umap_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"  ✓ Saved enhanced UMAP analysis to: figures/enhanced_umap_analysis.png")
    
    # Create individual high-quality plots (save only, no display)
    for method, data in clustering_results.items():
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=data['labels'], cmap='tab20', alpha=0.7, s=2)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title(f'{method} Clustering of Protein Domain Embeddings\nSilhouette Score: {data["score"]:.3f}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"figures/umap_{method.lower()}.png", dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        
        print(f"    ✓ Saved {method} plot to: figures/umap_{method.lower()}.png")

def analyze_cluster_characteristics(embeddings, cluster_labels, method_name):
    """Analyze characteristics of each cluster."""
    
    print(f"\nAnalyzing cluster characteristics for {method_name}...")
    
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    
    print(f"Number of clusters: {n_clusters}")
    
    # Cluster size analysis
    cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
    cluster_sizes_sorted = sorted(cluster_sizes, reverse=True)
    
    print(f"Cluster sizes: {cluster_sizes_sorted}")
    print(f"Largest cluster: {max(cluster_sizes)} proteins")
    print(f"Smallest cluster: {min(cluster_sizes)} proteins")
    
    # Create cluster size visualization (save only, no display)
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_clusters), cluster_sizes_sorted, color='skyblue', edgecolor='black')
    plt.xlabel('Cluster Rank (by size)')
    plt.ylabel('Number of Proteins')
    plt.title(f'Cluster Size Distribution - {method_name}')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, size in enumerate(cluster_sizes_sorted):
        plt.text(i, size + max(cluster_sizes_sorted) * 0.01, str(size), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"figures/cluster_size_distribution_{method_name.lower()}.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"  ✓ Saved cluster size distribution to: figures/cluster_size_distribution_{method_name.lower()}.png")
    
    # Cluster quality metrics
    print(f"\nCluster Quality Metrics:")
    print(f"  Silhouette Score: {silhouette_score(embeddings, cluster_labels):.3f}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Average cluster size: {np.mean(cluster_sizes):.1f}")
    print(f"  Cluster size std: {np.std(cluster_sizes):.1f}")

def main():
    print("Enhanced UMAP Analysis for Protein Domain Embeddings")
    print("="*70)
    
    # Load embeddings
    embeddings = load_embeddings()
    
    # Perform enhanced UMAP analysis
    embeddings_2d, clustering_results, best_method = enhanced_umap_analysis(embeddings, n_clusters=15)
    
    print(f"\n{'='*70}")
    print("ENHANCED UMAP ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"Best clustering method: {best_method}")
    print(f"Best silhouette score: {clustering_results[best_method]['score']:.3f}")
    print(f"✓ All figures saved to figures/ directory")

if __name__ == "__main__":
    main() 