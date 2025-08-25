#!/usr/bin/env python3
"""
Comprehensive embedding analysis with configurable dimensions.
Set your embedding dimension as a hyperparameter each time you run it.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

# Set matplotlib to non-interactive backend (no display)
plt.ioff()

def parse_arguments():
    """Parse command line arguments for configurable parameters."""
    parser = argparse.ArgumentParser(description='Comprehensive embedding analysis with configurable dimensions')
    
    parser.add_argument('--embed_dim', type=int, default=64,
                       help='Embedding dimension (default: 64)')
    parser.add_argument('--max_clusters', type=int, default=20,
                       help='Maximum number of clusters to test (default: 20)')
    parser.add_argument('--umap_clusters', type=int, default=15,
                       help='Number of clusters for UMAP analysis (default: 15)')
    parser.add_argument('--umap_neighbors', type=int, default=15,
                       help='UMAP n_neighbors parameter (default: 15)')
    parser.add_argument('--umap_epochs', type=int, default=200,
                       help='UMAP n_epochs parameter (default: 200)')
    parser.add_argument('--dbscan_eps', type=float, default=0.8,
                       help='DBSCAN eps parameter (default: 0.8)')
    parser.add_argument('--dbscan_min_samples', type=int, default=5,
                       help='DBSCAN min_samples parameter (default: 5)')
    
    return parser.parse_args()

def load_embeddings():
    """Load your embeddings."""
    embeddings = np.load("results/embeddings.npy")
    print(f"Loaded embeddings: {embeddings.shape}")
    return embeddings

def comprehensive_analysis(embeddings, args):
    """Combined clustering quality and UMAP analysis with configurable parameters."""
    
    embed_dim = embeddings.shape[1]
    expected_dim = args.embed_dim
    
    print("="*70)
    print(f"COMPREHENSIVE {embed_dim}D EMBEDDING ANALYSIS")
    print("="*70)
    
    # Verify dimensions
    if embed_dim != expected_dim:
        print(f"⚠ Warning: Expected {expected_dim}D, got {embed_dim}D")
        print(f"  Using actual dimension: {embed_dim}D")
        print(f"  Consider updating --embed_dim to {embed_dim}")
    else:
        print(f"✓ {embed_dim}D embeddings confirmed")
    
    # Step 1: Clustering Quality Test
    print(f"\n1. TESTING CLUSTERING QUALITY ({embed_dim}D)...")
    print("-" * 50)
    
    n_clusters_range = range(2, args.max_clusters + 1)
    silhouette_scores = []
    calinski_scores = []
    all_cluster_labels = {}
    
    print(f"Testing clustering quality with different numbers of clusters...")
    print(f"({embed_dim}D data allows for more complex clustering patterns)")
    
    for n_clusters in n_clusters_range:
        # Perform clustering on embed_dim data
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        # Calculate metrics
        sil_score = silhouette_score(embeddings, cluster_labels)
        cal_score = calinski_harabasz_score(embeddings, cluster_labels)
        
        silhouette_scores.append(sil_score)
        calinski_scores.append(cal_score)
        all_cluster_labels[n_clusters] = cluster_labels
        
        print(f"  {n_clusters} clusters: Silhouette={sil_score:.3f}, Calinski-Harabasz={cal_score:.0f}")
    
    # Find optimal number of clusters
    optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    print(f"\n✓ Optimal number of clusters for {embed_dim}D data: {optimal_clusters}")
    
    # Step 2: UMAP Dimensionality Reduction (embed_dim → 2D)
    print(f"\n2. UMAP DIMENSIONALITY REDUCTION ({embed_dim}D → 2D)...")
    print("-" * 50)
    
    print(f"Performing UMAP dimensionality reduction from {embed_dim}D to 2D...")
    
    # UMAP parameters optimized for embed_dim data
    umap = UMAP(
        n_components=2,                    # Reduce to 2D for visualization
        random_state=42,                   # Reproducible results
        n_neighbors=args.umap_neighbors,   # Configurable
        min_dist=0.1,                      # Balance between local and global structure
        n_epochs=args.umap_epochs,         # Configurable
        metric='euclidean'                 # Good for high-dimensional embeddings
    )
    
    embeddings_2d = umap.fit_transform(embeddings)
    
    print(f"✓ UMAP complete! Reduced from {embed_dim}D to 2D")
    print(f"  Original shape: {embeddings.shape}")
    print(f"  Reduced shape: {embeddings_2d.shape}")
    
    # Step 3: Multiple Clustering Methods
    print("\n3. MULTIPLE CLUSTERING METHODS...")
    print("-" * 50)
    
    clustering_results = {}
    
    # Use optimal number of clusters from step 1
    n_clusters = optimal_clusters
    
    # K-means (already computed, reuse)
    print(f"  K-means clustering on {embed_dim}D data (reusing from quality test)...")
    kmeans_labels = all_cluster_labels[n_clusters]
    kmeans_score = silhouette_scores[n_clusters - 2]
    clustering_results['K-means'] = {'labels': kmeans_labels, 'score': kmeans_score}
    print(f"    ✓ K-means silhouette score: {kmeans_score:.3f}")
    
    # DBSCAN clustering on embed_dim data
    print(f"  DBSCAN clustering on {embed_dim}D data...")
    # Use configurable DBSCAN parameters
    dbscan = DBSCAN(eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)
    dbscan_labels = dbscan.fit_predict(embeddings)
    if len(np.unique(dbscan_labels)) > 1:
        dbscan_score = silhouette_score(embeddings, dbscan_labels)
        clustering_results['DBSCAN'] = {'labels': dbscan_labels, 'score': dbscan_score}
        print(f"    ✓ DBSCAN silhouette score: {dbscan_score:.3f}")
    else:
        print("    ⚠ DBSCAN: Only one cluster found")
    
    # Hierarchical clustering on UMAP space (2D for speed)
    print("  Hierarchical clustering on UMAP space (2D)...")
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical_labels = hierarchical.fit_predict(embeddings_2d)
    hierarchical_score = silhouette_score(embeddings_2d, hierarchical_labels)
    clustering_results['Hierarchical'] = {'labels': hierarchical_labels, 'score': hierarchical_score}
    print(f"    ✓ Hierarchical silhouette score: {hierarchical_score:.3f}")
    
    # Find best clustering method
    best_method = max(clustering_results.keys(), 
                     key=lambda x: clustering_results[x]['score'])
    best_labels = clustering_results[best_method]['labels']
    best_score = clustering_results[best_method]['score']
    
    print(f"\n✓ Best clustering method for {embed_dim}D data: {best_method} (score: {best_score:.3f})")
    
    return (embeddings_2d, clustering_results, best_method, 
            optimal_clusters, silhouette_scores, calinski_scores, embed_dim)

def create_comprehensive_plots(embeddings_2d, clustering_results, best_method,
                              optimal_clusters, silhouette_scores, calinski_scores, embed_dim):
    """Create all plots for comprehensive analysis."""
    
    print(f"\n4. CREATING COMPREHENSIVE VISUALIZATIONS...")
    print("-" * 50)
    
    # Plot 1: Clustering Quality
    print(f"  Creating clustering quality plots for {embed_dim}D embeddings...")
    n_clusters_range = range(2, len(silhouette_scores) + 2)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Silhouette score plot
    ax1.plot(n_clusters_range, silhouette_scores, 'bo-', linewidth=2, markersize=6)
    ax1.axvline(x=optimal_clusters, color='r', linestyle='--', linewidth=2,
                label=f'Optimal: {optimal_clusters}')
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Silhouette Score')
    ax1.set_title(f'Clustering Quality: Silhouette Score vs Clusters\n({embed_dim}D Protein Domain Embeddings)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Calinski-Harabasz score plot
    ax2.plot(n_clusters_range, calinski_scores, 'go-', linewidth=2, markersize=6)
    ax2.axvline(x=optimal_clusters, color='r', linestyle='--', linewidth=2,
                label=f'Optimal: {optimal_clusters}')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Calinski-Harabasz Score')
    ax2.set_title(f'Clustering Quality: Calinski-Harabasz Score vs Clusters\n({embed_dim}D Protein Domain Embeddings)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"figures/comprehensive_clustering_quality_{embed_dim}d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved clustering quality plots to: figures/comprehensive_clustering_quality_{embed_dim}d.png")
    
    # Plot 2: Enhanced UMAP Analysis (embed_dim → 2D)
    print(f"  Creating enhanced UMAP plots ({embed_dim}D → 2D)...")
    
    # Create subplot grid
    n_methods = len(clustering_results)
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # 1. Best clustering method
    best_labels = clustering_results[best_method]['labels']
    scatter1 = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                               c=best_labels, cmap='tab20', alpha=0.7, s=2)
    axes[0].set_title(f'Best Method: {best_method}\nSilhouette Score: {clustering_results[best_method]["score"]:.3f}\n({embed_dim}D → 2D)', 
                      fontsize=14, fontweight='bold')
    axes[0].set_xlabel('UMAP Component 1')
    axes[0].set_ylabel('UMAP Component 2')
    axes[0].grid(True, alpha=0.3)
    
    # 2. K-means
    if 'K-means' in clustering_results:
        kmeans_labels = clustering_results['K-means']['labels']
        scatter2 = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=kmeans_labels, cmap='tab20', alpha=0.7, s=2)
        axes[1].set_title(f'K-means Clustering ({embed_dim}D)\nScore: {clustering_results["K-means"]["score"]:.3f}')
        axes[1].set_xlabel('UMAP Component 1')
        axes[1].set_ylabel('UMAP Component 2')
        axes[1].grid(True, alpha=0.3)
    
    # 3. DBSCAN (if available)
    if 'DBSCAN' in clustering_results:
        dbscan_labels = clustering_results['DBSCAN']['labels']
        scatter3 = axes[2].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=dbscan_labels, cmap='tab20', alpha=0.7, s=2)
        axes[2].set_title(f'DBSCAN Clustering ({embed_dim}D)\nScore: {clustering_results["DBSCAN"]["score"]:.3f}')
        axes[2].set_xlabel('UMAP Component 1')
        axes[2].set_ylabel('UMAP Component 2')
        axes[2].grid(True, alpha=0.3)
    
    # 4. Hierarchical
    if 'Hierarchical' in clustering_results:
        hierarchical_labels = clustering_results['Hierarchical']['labels']
        scatter4 = axes[3].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                   c=hierarchical_labels, cmap='tab20', alpha=0.7, s=2)
        axes[3].set_title(f'Hierarchical Clustering (2D UMAP)\nScore: {clustering_results["Hierarchical"]["score"]:.3f}')
        axes[3].set_xlabel('UMAP Component 1')
        axes[3].set_ylabel('UMAP Component 2')
        axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"figures/comprehensive_umap_analysis_{embed_dim}d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"    ✓ Saved enhanced UMAP analysis to: figures/comprehensive_umap_analysis_{embed_dim}d.png")
    
    # Create individual high-quality plots
    for method, data in clustering_results.items():
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                             c=data['labels'], cmap='tab20', alpha=0.7, s=2)
        plt.colorbar(scatter, label='Cluster')
        plt.xlabel('UMAP Component 1')
        plt.ylabel('UMAP Component 2')
        plt.title(f'{method} Clustering of {embed_dim}D Protein Domain Embeddings\nSilhouette Score: {data["score"]:.3f}')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"figures/comprehensive_umap_{method.lower()}_{embed_dim}d.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    ✓ Saved {method} plot to: figures/comprehensive_umap_{method.lower()}_{embed_dim}d.png")

def analyze_cluster_characteristics(embeddings, cluster_labels, method_name, embed_dim):
    """Analyze characteristics of each cluster."""
    
    print(f"\n5. ANALYZING CLUSTER CHARACTERISTICS ({embed_dim}D)...")
    print("-" * 50)
    
    unique_labels = np.unique(cluster_labels)
    n_clusters = len(unique_labels)
    
    print(f"Number of clusters: {n_clusters}")
    print(f"Embedding dimensions: {embeddings.shape[1]}")
    
    # Cluster size analysis
    cluster_sizes = [np.sum(cluster_labels == label) for label in unique_labels]
    cluster_sizes_sorted = sorted(cluster_sizes, reverse=True)
    
    print(f"Cluster sizes: {cluster_sizes_sorted}")
    print(f"Largest cluster: {max(cluster_sizes)} proteins")
    print(f"Smallest cluster: {min(cluster_sizes)} proteins")
    
    # Create cluster size visualization
    plt.figure(figsize=(10, 6))
    plt.bar(range(n_clusters), cluster_sizes_sorted, color='skyblue', edgecolor='black')
    plt.xlabel('Cluster Rank (by size)')
    plt.ylabel('Number of Proteins')
    plt.title(f'Cluster Size Distribution - {method_name}\n({embed_dim}D Embeddings)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, size in enumerate(cluster_sizes_sorted):
        plt.text(i, size + max(cluster_sizes_sorted) * 0.01, str(size), 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"figures/comprehensive_cluster_size_distribution_{method_name.lower()}_{embed_dim}d.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved cluster size distribution to: figures/comprehensive_cluster_size_distribution_{method_name.lower()}_{embed_dim}d.png")
    
    # Cluster quality metrics
    print(f"\nCluster Quality Metrics ({embed_dim}D):")
    print(f"  Silhouette Score: {silhouette_score(embeddings, cluster_labels):.3f}")
    print(f"  Number of clusters: {n_clusters}")
    print(f"  Average cluster size: {np.mean(cluster_sizes):.1f}")
    print(f"  Cluster size std: {np.std(cluster_sizes):.1f}")
    
    # Dimension specific insights
    print(f"  Embedding dimensions: {embeddings.shape[1]}")
    print(f"  Total proteins: {embeddings.shape[0]}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    print("COMPREHENSIVE EMBEDDING ANALYSIS")
    print("Configurable Dimensions and Parameters")
    print("="*70)
    
    # Display configuration
    print(f"Configuration:")
    print(f"  Embedding dimension: {args.embed_dim}D")
    print(f"  Max clusters to test: {args.max_clusters}")
    print(f"  UMAP clusters: {args.umap_clusters}")
    print(f"  UMAP neighbors: {args.umap_neighbors}")
    print(f"  UMAP epochs: {args.umap_epochs}")
    print(f"  DBSCAN eps: {args.dbscan_eps}")
    print(f"  DBSCAN min_samples: {args.dbscan_min_samples}")
    print()
    
    # Load embeddings
    embeddings = load_embeddings()
    
    # Perform comprehensive analysis
    (embeddings_2d, clustering_results, best_method, 
     optimal_clusters, silhouette_scores, calinski_scores, actual_dim) = comprehensive_analysis(embeddings, args)
    
    # Create all plots
    create_comprehensive_plots(embeddings_2d, clustering_results, best_method,
                              optimal_clusters, silhouette_scores, calinski_scores, actual_dim)
    
    # Analyze cluster characteristics
    analyze_cluster_characteristics(embeddings, clustering_results[best_method]['labels'], best_method, actual_dim)
    
    print(f"\n{'='*70}")
    print("COMPREHENSIVE ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"✓ Original dimensions: {actual_dim}D")
    print(f"✓ Reduced dimensions: 2D (UMAP)")
    print(f"✓ Optimal number of clusters: {optimal_clusters}")
    print(f"✓ Best clustering method: {best_method}")
    print(f"✓ Best silhouette score: {clustering_results[best_method]['score']:.3f}")
    print(f"✓ All figures saved to figures/ directory")
    print(f"✓ Analysis completed with configurable parameters!")

if __name__ == "__main__":
    main() 