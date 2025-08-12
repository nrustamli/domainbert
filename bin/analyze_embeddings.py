#!/usr/bin/env python3
"""
Script to analyze domain embeddings using anndata and scanpy.
"""

import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import argparse
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

def build_parser():
    """CLI parser"""
    parser = argparse.ArgumentParser(
        description="Analyze domain embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "adata_file",
        type=str,
        help="Input AnnData file (.h5ad)"
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Leiden clustering resolution"
    )
    parser.add_argument(
        "--n_comps",
        type=int,
        default=50,
        help="Number of PCA components"
    )
    return parser

def analyze_embeddings(adata, resolution=0.5, n_comps=50):
    """Perform comprehensive embedding analysis."""
    
    # Dimensionality reduction
    sc.pp.pca(adata, n_comps=n_comps)
    sc.pp.neighbors(adata)
    sc.tl.leiden(adata, resolution=resolution, flavor="igraph", n_iterations=2)
    sc.tl.umap(adata)
    
    return adata

def create_visualizations(adata, output_dir):
    """Create various visualizations."""
    
    # UMAP plot colored by clusters
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(adata.obsm['X_umap'][:, 0], 
                         adata.obsm['X_umap'][:, 1], 
                         c=adata.obs['leiden'].astype('category').cat.codes,
                         cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Domain Sequence Clusters (UMAP)', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'umap_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # PCA plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(adata.obsm['X_pca'][:, 0], 
                         adata.obsm['X_pca'][:, 1], 
                         c=adata.obs['leiden'].astype('category').cat.codes,
                         cmap='viridis', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Cluster')
    plt.title('Domain Sequence Clusters (PCA)', fontsize=16)
    plt.xlabel('PC 1', fontsize=12)
    plt.ylabel('PC 2', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pca_clusters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cluster size distribution
    plt.figure(figsize=(10, 6))
    cluster_counts = adata.obs['leiden'].value_counts()
    plt.bar(range(len(cluster_counts)), cluster_counts.values)
    plt.xlabel('Cluster')
    plt.ylabel('Number of sequences')
    plt.title('Cluster Size Distribution')
    plt.xticks(range(len(cluster_counts)), cluster_counts.index)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cluster_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save cluster statistics
    cluster_stats = pd.DataFrame({
        'cluster': cluster_counts.index,
        'size': cluster_counts.values,
        'percentage': (cluster_counts.values / len(adata)) * 100
    })
    cluster_stats.to_csv(os.path.join(output_dir, 'cluster_statistics.csv'), index=False)

def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outfile, exist_ok=True)
    
    # Load AnnData
    print("Loading AnnData...")
    adata = ad.read_h5ad(args.adata_file)
    print(f"Loaded {adata.n_obs} sequences with {adata.n_vars} features")
    
    # Analyze embeddings
    print("Analyzing embeddings...")
    adata = analyze_embeddings(adata, args.resolution, args.n_comps)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(adata, args.outfile)
    
    # Save analyzed data
    adata.write(os.path.join(args.outfile, 'analyzed_embeddings.h5ad'))
    
    # Print summary
    print(f"Found {len(adata.obs['leiden'].unique())} clusters")
    print("Cluster sizes:")
    print(adata.obs['leiden'].value_counts())
    
    print("Done! Analysis results saved in:", args.outfile)

if __name__ == "__main__":
    main()