#!/usr/bin/env python3
"""
Script to embed domain sequences and perform clustering analysis using anndata.
"""

import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import matplotlib.pyplot as plt
import pickle
import os
import sys
import argparse
import logging
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Add the main module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.data_loader import load_domain_sequences
from main.featurization import build_domain_vocab, tokenize_sequences
from main.models.transformer_custom import DomainTransformer

logging.basicConfig(level=logging.INFO)

def build_parser():
    """CLI parser"""
    parser = argparse.ArgumentParser(
        description="Embed domain sequences and perform clustering",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Input CSV file with domain sequences"
    )
    parser.add_argument(
        "outfile", 
        type=str, 
        help="Output directory for results"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/domain_model.pth",
        help="Path to pre-trained model"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=10,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=128,
        help="Embedding dimension"
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    return parser

def embed_sequences(model, tokenized_sequences, batch_size=32):
    """Extract embeddings for all sequences."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(tokenized_sequences), batch_size):
            batch = tokenized_sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch)
            # Use get_embeddings method to get 2D embeddings
            batch_embeddings = model.get_embeddings(batch_tensor)
            embeddings.append(batch_embeddings.numpy())
    
    return np.vstack(embeddings)

def cluster_with_anndata(embeddings, obs_df, resolution=0.5, n_comps=50):
    """Perform clustering using anndata and scanpy."""
    # Create AnnData object
    embed_adata = ad.AnnData(embeddings, obs=obs_df)
    
    # Dimensionality reduction
    sc.pp.pca(embed_adata, n_comps=n_comps)
    
    # Build neighborhood graph
    sc.pp.neighbors(embed_adata)
    
    # Perform clustering with explicit backend to avoid warning
    sc.tl.leiden(embed_adata, resolution=resolution, flavor="igraph", n_iterations=2)
    
    return embed_adata

def visualize_clusters(embed_adata, output_path):
    """Create UMAP visualization of clustered embeddings."""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Compute UMAP
    sc.tl.umap(embed_adata)
    
    # Create custom plot instead of using scanpy's save function
    plt.figure(figsize=(12, 10))
    
    # Get UMAP coordinates
    umap_coords = embed_adata.obsm['X_umap']
    
    # Create scatter plot
    scatter = plt.scatter(umap_coords[:, 0], umap_coords[:, 1], 
                         c=embed_adata.obs['leiden'].astype('category').cat.codes,
                         cmap='viridis', alpha=0.6, s=20)
    
    plt.colorbar(scatter, label='Cluster')
    plt.title('Domain Sequence Clusters (UMAP)', fontsize=16)
    plt.xlabel('UMAP 1', fontsize=12)
    plt.ylabel('UMAP 2', fontsize=12)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {output_path}")

def find_model_file(model_path):
    """Find the model file in various possible locations."""
    possible_paths = [
        model_path,
        os.path.join(os.path.dirname(__file__), '..', model_path),
        os.path.join(os.getcwd(), model_path),
        os.path.join(os.getcwd(), 'results', 'domain_model.pth'),
        os.path.join(os.path.dirname(__file__), '..', 'results', 'domain_model.pth')
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None

def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outfile, exist_ok=True)
    
    # Load and featurize data
    print("Loading domain sequences...")
    sequences = load_domain_sequences(args.infile)
    domain2idx = build_domain_vocab(sequences)
    tokenized = tokenize_sequences(sequences, domain2idx, max_length=args.max_length)
    
    # Create observation DataFrame
    protein_codes = [seq[0] for seq in sequences]
    obs_df = pd.DataFrame({
        'protein_code': protein_codes,
        'domain_sequence': [','.join(seq[1]) for seq in sequences]
    })
    
    print(f"Loaded {len(sequences)} sequences with {len(domain2idx)} unique domains")
    
    # Load model
    print("Loading model...")
    model = DomainTransformer(
        vocab_size=len(domain2idx), 
        embed_dim=args.embed_dim, 
        max_length=args.max_length
    )
    
    # Try to find and load the model file
    model_file = find_model_file(args.model_path)
    if model_file and os.path.exists(model_file):
        try:
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            print(f"Loaded pre-trained model from: {model_file}")
        except Exception as e:
            print(f"Warning: Could not load model from {model_file}: {e}")
            print("Using random weights.")
    else:
        print(f"Warning: No pre-trained model found at {args.model_path}")
        print("Using random weights.")
        print("Available model files:")
        for path in [args.model_path, "results/domain_model.pth", "results/"]:
            if os.path.exists(path):
                print(f"  - {path}")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = embed_sequences(model, tokenized, batch_size=args.batch_size)
    
    # Check embedding shape
    print(f"Embedding shape: {embeddings.shape}")
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embeddings, got shape {embeddings.shape}")
    
    # Perform clustering with anndata
    print("Performing clustering...")
    embed_adata = cluster_with_anndata(
        embeddings, obs_df, resolution=args.resolution, n_comps=args.n_comps
    )
    
    # Save results
    print("Saving results...")
    embed_adata.write(os.path.join(args.outfile, 'embeddings_adata.h5ad'))
    
    with open(os.path.join(args.outfile, 'domain2idx.pkl'), 'wb') as f:
        pickle.dump(domain2idx, f)
    
    # Save embeddings as numpy array for compatibility
    np.save(os.path.join(args.outfile, 'embeddings.npy'), embeddings)
    np.save(os.path.join(args.outfile, 'cluster_labels.npy'), 
            embed_adata.obs['leiden'].values)
    
    # Create visualization
    print("Creating visualization...")
    viz_path = os.path.join(args.outfile, 'cluster_visualization.png')
    visualize_clusters(embed_adata, viz_path)
    
    # Print clustering results
    print(f"Found {len(embed_adata.obs['leiden'].unique())} clusters")
    print("Cluster sizes:")
    print(embed_adata.obs['leiden'].value_counts())
    
    print("Done! Results saved in:", args.outfile)

if __name__ == "__main__":
    main()