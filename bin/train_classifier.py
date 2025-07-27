#!/usr/bin/env python3
"""
Script to train a classifier on domain embeddings using anndata.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
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
        description="Train classifier on domain embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Input CSV file with domain sequences"
    )
    parser.add_argument(
        "labels_file",
        type=str,
        help="File containing labels for classification"
    )
    parser.add_argument(
        "outfile",
        type=str,
        help="Output directory for results"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../results/domain_model.pth",
        help="Path to pre-trained domain model"
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
        default=0.001,
        help="Learning rate"
    )
    return parser

class DomainClassifier(nn.Module):
    def __init__(self, embed_dim, num_classes, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, embeddings):
        return self.classifier(embeddings)

def load_labels(labels_file, protein_codes):
    """Load labels for the given protein codes."""
    # This is a placeholder - you'll need to implement based on your label format
    # For now, creating dummy labels
    labels = np.random.randint(0, 3, len(protein_codes))  # 3 classes
    return labels

def prepare_classification_data(sequences, labels, domain2idx, max_length):
    """Prepare data for classification training."""
    tokenized = tokenize_sequences(sequences, domain2idx, max_length)
    return torch.tensor(tokenized), torch.tensor(labels)

def train_classifier(model, classifier, train_loader, val_loader, 
                   num_epochs=10, learning_rate=0.001):
    """Train the classifier."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classifier = classifier.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training
        model.eval()
        classifier.train()
        train_loss = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            with torch.no_grad():
                embeddings = model(batch_x)
            
            optimizer.zero_grad()
            outputs = classifier(embeddings)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        classifier.eval()
        val_predictions = []
        val_true = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                embeddings = model(batch_x)
                outputs = classifier(embeddings)
                predictions = torch.argmax(outputs, dim=1)
                
                val_predictions.extend(predictions.cpu().numpy())
                val_true.extend(batch_y.cpu().numpy())
        
        val_accuracy = accuracy_score(val_true, val_predictions)
        train_losses.append(train_loss / len(train_loader))
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_losses[-1]:.4f}")
        print(f"  Val Accuracy: {val_accuracy:.4f}")
    
    return classifier, train_losses, val_accuracies

def main():
    """Run script"""
    parser = build_parser()
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.outfile, exist_ok=True)
    
    # Load data
    print("Loading domain sequences...")
    sequences = load_domain_sequences(args.infile)
    protein_codes = [seq[0] for seq in sequences]
    
    # Load labels
    print("Loading labels...")
    labels = load_labels(args.labels_file, protein_codes)
    
    # Featurize data
    domain2idx = build_domain_vocab(sequences)
    tokenized, labels_tensor = prepare_classification_data(
        sequences, labels, domain2idx, args.max_length
    )
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        tokenized, labels_tensor, test_size=0.2, random_state=42
    )
    
    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    
    # Load pre-trained domain model
    print("Loading pre-trained domain model...")
    model = DomainTransformer(
        vocab_size=len(domain2idx), 
        embed_dim=args.embed_dim, 
        max_length=args.max_length
    )
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
        print("Loaded pre-trained model")
    else:
        print("Warning: No pre-trained model found. Using random weights.")
    
    # Initialize classifier
    num_classes = len(np.unique(labels))
    classifier = DomainClassifier(args.embed_dim, num_classes)
    
    # Train classifier
    print("Training classifier...")
    trained_classifier, train_losses, val_accuracies = train_classifier(
        model, classifier, train_loader, val_loader, 
        args.num_epochs, args.learning_rate
    )
    
    # Save results
    print("Saving results...")
    torch.save(trained_classifier.state_dict(), 
              os.path.join(args.outfile, 'classifier.pth'))
    
    np.save(os.path.join(args.outfile, 'train_losses.npy'), train_losses)
    np.save(os.path.join(args.outfile, 'val_accuracies.npy'), val_accuracies)
    
    # Create anndata object with embeddings and results
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        embeddings = model(tokenized.to(device)).cpu().numpy()
    
    obs_df = pd.DataFrame({
        'protein_code': protein_codes,
        'domain_sequence': [','.join(seq[1]) for seq in sequences],
        'label': labels,
        'split': ['train' if i < len(X_train) else 'val' 
                 for i in range(len(sequences))]
    })
    
    adata = ad.AnnData(embeddings, obs=obs_df)
    adata.write(os.path.join(args.outfile, 'classification_results.h5ad'))
    
    print("Done! Results saved in:", args.outfile)

if __name__ == "__main__":
    main()