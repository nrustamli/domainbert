#!/usr/bin/env python3
"""
Simple script to extract embeddings from your DomainBERT model.
"""

import torch
import numpy as np
import pandas as pd
import os
import sys
import argparse

# Add the main module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.data_loader import load_domain_sequences
from main.featurization import build_domain_vocab, tokenize_sequences
from main.models.transformer_custom import DomainTransformer

def build_parser():
    """CLI parser"""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from DomainBERT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/your_corpus.csv",
        help="Path to domain sequences CSV"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="results/domain_model.pth",
        help="Path to trained model"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="results/embeddings.npy",
        help="Output path for embeddings"
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
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding"
    )
    return parser

def extract_embeddings(model, tokenized_sequences, batch_size=32):
    """Extract embeddings from your model."""
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(tokenized_sequences), batch_size):
            batch = tokenized_sequences[i:i + batch_size]
            batch_tensor = torch.tensor(batch)
            # Get embeddings from your model
            batch_embeddings = model.get_embeddings(batch_tensor)
            embeddings.append(batch_embeddings.numpy())
    
    return np.vstack(embeddings)

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
    parser = build_parser()
    args = parser.parse_args()
    
    print("1. Loading domain sequences...")
    sequences = load_domain_sequences(args.data_path)
    
    print("2. Building vocabulary...")
    domain2idx = build_domain_vocab(sequences)
    
    print("3. Tokenizing sequences...")
    tokenized = tokenize_sequences(sequences, domain2idx, max_length=args.max_length)
    
    print("4. Loading your DomainBERT model...")
    model = DomainTransformer(
        vocab_size=len(domain2idx), 
        embed_dim=args.embed_dim, 
        max_length=args.max_length
    )
    
    # Load your trained model
    model_file = find_model_file(args.model_path)
    if model_file and os.path.exists(model_file):
        try:
            model.load_state_dict(torch.load(model_file, map_location='cpu'))
            print(f"   Loaded model from: {model_file}")
        except Exception as e:
            print(f"   Warning: Could not load model from {model_file}: {e}")
            print("   Using random weights (embeddings won't be meaningful)")
    else:
        print(f"   Warning: Model not found at {args.model_path}")
        print("   Using random weights (embeddings won't be meaningful)")
    
    print("5. Extracting embeddings...")
    embeddings = extract_embeddings(model, tokenized, batch_size=args.batch_size)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    print(f"6. Saving embeddings... Shape: {embeddings.shape}")
    np.save(args.output_path, embeddings)
    
    print(f"Done! Embeddings saved to: {args.output_path}")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Number of sequences: {len(sequences)}")

if __name__ == "__main__":
    main()