#!/usr/bin/env python3
"""
Script to train the domain embedding model (self-supervised learning).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import anndata as ad
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
        description="Train domain embedding model",
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
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.0001,
        help="Learning rate"
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.15,
        help="Probability of masking tokens"
    )
    return parser

def create_masked_sequences(tokenized_sequences, mask_prob=0.15):
    """Create masked sequences for MLM training."""
    masked_sequences = []
    labels = []
    
    for seq in tokenized_sequences:
        masked_seq = seq.copy()
        seq_labels = [-100] * len(seq)  # -100 is ignored in loss computation
        
        # Randomly mask some tokens
        for i in range(len(seq)):
            if np.random.random() < mask_prob and seq[i] != 0:  # Don't mask padding
                seq_labels[i] = seq[i]  # Original token becomes the label
                masked_seq[i] = 1  # Mask token (assuming 1 is [MASK])
        
        masked_sequences.append(masked_seq)
        labels.append(seq_labels)
    
    return masked_sequences, labels

def train_domain_model(model, train_loader, num_epochs=10, learning_rate=0.0001):
    """Train the domain model using masked language modeling."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through the model
            # Use the model's forward method directly for MLM
            logits = model(batch_x, return_embeddings=False)  # This returns MLM logits
            
            # Reshape for loss computation
            logits = logits.reshape(-1, logits.size(-1))
            batch_y = batch_y.reshape(-1)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model, train_losses

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
    
    print(f"Loaded {len(sequences)} sequences with {len(domain2idx)} unique domains")
    
    # Create masked sequences for MLM
    print("Creating masked sequences...")
    masked_sequences, labels = create_masked_sequences(tokenized, args.mask_prob)
    
    # Create data loader
    dataset = TensorDataset(
        torch.tensor(masked_sequences), 
        torch.tensor(labels)
    )
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # Initialize model
    print("Initializing model...")
    model = DomainTransformer(
        vocab_size=len(domain2idx), 
        embed_dim=args.embed_dim, 
        max_length=args.max_length
    )
    
    # Train model
    print("Training domain model...")
    trained_model, train_losses = train_domain_model(
        model, train_loader, args.num_epochs, args.learning_rate
    )
    
    # Save results
    print("Saving results...")
    torch.save(trained_model.state_dict(), 
              os.path.join(args.outfile, 'domain_model.pth'))
    
    np.save(os.path.join(args.outfile, 'train_losses.npy'), train_losses)
    
    with open(os.path.join(args.outfile, 'domain2idx.pkl'), 'wb') as f:
        import pickle
        pickle.dump(domain2idx, f)
    
    # Create anndata object with final embeddings
    print("Creating final embeddings...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()
    with torch.no_grad():
        embeddings = trained_model.get_embeddings(torch.tensor(tokenized).to(device)).cpu().numpy()
    
    obs_df = pd.DataFrame({
        'protein_code': [seq[0] for seq in sequences],
        'domain_sequence': [','.join(seq[1]) for seq in sequences]
    })
    
    adata = ad.AnnData(embeddings, obs=obs_df)
    adata.write(os.path.join(args.outfile, 'final_embeddings.h5ad'))
    
    print("Done! Model and results saved in:", args.outfile)

if __name__ == "__main__":
    main()