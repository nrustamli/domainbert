#!/usr/bin/env python3
"""
Check and analyze your extracted embeddings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import os
import argparse

def build_parser():
    """CLI parser"""
    parser = argparse.ArgumentParser(
        description="Check and analyze embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="results/embeddings.npy",
        help="Path to embeddings file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="Output directory for analysis"
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=5,
        help="Number of clusters for K-means"
    )
    return parser

def analyze_embeddings(embeddings_path):
    """Analyze your embeddings."""
    print("Loading embeddings...")
    embeddings = np.load(embeddings_path)
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of sequences: {embeddings.shape[0]}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Basic statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {np.mean(embeddings):.4f}")
    print(f"  Std: {np.std(embeddings):.4f}")
    print(f"  Min: {np.min(embeddings):.4f}")
    print(f"  Max: {np.max(embeddings):.4f}")
    
    # Check for NaN or infinite values
    if np.any(np.isnan(embeddings)):
        print("  WARNING: Found NaN values!")
    if np.any(np.isinf(embeddings)):
        print("  WARNING: Found infinite values!")
    
    return embeddings

def visualize_embeddings(embeddings, output_path="results/embedding_visualization.png"):
    """Create a simple visualization of your embeddings."""
    print("Creating visualization...")
    
    # Use t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=1)
    plt.title('Domain Embeddings (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def cluster_embeddings(embeddings, n_clusters=5):
    """Perform simple clustering on your embeddings."""
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    # Print cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    print("Cluster sizes:")
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} sequences")
    
    return cluster_labels

def main():
    parser = build_parser()
    args = parser.parse_args()
    
    if not os.path.exists(args.embeddings_path):
        print(f"Embeddings not found at {args.embeddings_path}")
        print("Please run extract_embeddings.py first")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze embeddings
    embeddings = analyze_embeddings(args.embeddings_path)
    
    # Visualize embeddings
    viz_path = os.path.join(args.output_dir, 'embedding_visualization.png')
    visualize_embeddings(embeddings, viz_path)
    
    # Cluster embeddings
    cluster_labels = cluster_embeddings(embeddings, args.n_clusters)
    
    # Save cluster labels
    cluster_path = os.path.join(args.output_dir, 'cluster_labels.npy')
    np.save(cluster_path, cluster_labels)
    print(f"Cluster labels saved to: {cluster_path}")

if __name__ == "__main__":
    main()