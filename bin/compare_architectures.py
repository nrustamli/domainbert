#!/usr/bin/env python3
"""
Comparative analysis script for different embedding dimensions.
Records training time, memory usage, and embedding quality metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import time
import psutil
import os
import sys
import argparse
from datetime import datetime
import json

# Add the main module to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from main.data_loader import load_domain_sequences
from main.featurization import build_domain_vocab, tokenize_sequences
from main.models.transformer_custom import DomainTransformer

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def get_gpu_memory_usage():
    """Get GPU memory usage in MB if available."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def calculate_embedding_quality(embeddings, labels=None):
    """Calculate embedding quality metrics."""
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    
    # Basic statistics
    quality_metrics = {
        'mean_norm': float(np.mean(np.linalg.norm(embeddings, axis=1))),
        'std_norm': float(np.std(np.linalg.norm(embeddings, axis=1))),
        'sparsity': float(np.mean(embeddings == 0)),
        'unique_embeddings': int(len(np.unique(embeddings, axis=0))),
        'total_embeddings': int(len(embeddings))
    }
    
    # Dimensionality reduction for clustering quality
    if embeddings.shape[0] > 1000:  # Sample for large datasets
        sample_indices = np.random.choice(embeddings.shape[0], 1000, replace=False)
        sample_embeddings = embeddings[sample_indices]
    else:
        sample_embeddings = embeddings
    
    try:
        # TSNE for 2D visualization
        tsne = TSNE(n_components=2, random_state=42, n_jobs=-1)
        tsne_embeddings = tsne.fit_transform(sample_embeddings)
        
        # Clustering quality metrics
        kmeans = KMeans(n_clusters=min(10, len(sample_embeddings)//10), random_state=42)
        cluster_labels = kmeans.fit_predict(sample_embeddings)
        
        quality_metrics.update({
            'silhouette_score': float(silhouette_score(sample_embeddings, cluster_labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(sample_embeddings, cluster_labels)),
            'tsne_variance_explained': float(np.var(tsne_embeddings).sum())
        })
    except Exception as e:
        print(f"Warning: Could not calculate clustering metrics: {e}")
        quality_metrics.update({
            'silhouette_score': -1,
            'calinski_harabasz_score': -1,
            'tsne_variance_explained': -1
        })
    
    return quality_metrics

def train_and_evaluate_model(embed_dim, sequences, domain2idx, max_length=10, 
                           num_epochs=4, batch_size=32, learning_rate=0.0001):
    """Train a model with given parameters and return comprehensive metrics."""
    
    print(f"\n{'='*60}")
    print(f"Testing Embedding Dimension: {embed_dim}")
    print(f"{'='*60}")
    
    # Record start time and memory
    start_time = time.time()
    initial_memory = get_memory_usage()
    initial_gpu_memory = get_gpu_memory_usage()
    
    # Prepare data
    tokenized = tokenize_sequences(sequences, domain2idx, max_length)
    
    # Create masked sequences for MLM
    from main.featurization import create_masked_sequences
    masked_sequences, labels = create_masked_sequences(tokenized, mask_prob=0.15)
    
    # Create data loader
    dataset = TensorDataset(
        torch.tensor(masked_sequences), 
        torch.tensor(labels)
    )
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = DomainTransformer(
        vocab_size=len(domain2idx), 
        embed_dim=embed_dim, 
        max_length=max_length
    )
    
    # Training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    train_accuracies = []
    epoch_times = []
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        model.train()
        total_loss = 0
        total_correct = 0
        total_masked = 0
        
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_x, return_embeddings=False)
            
            logits = logits.reshape(-1, logits.size(-1))
            batch_y = batch_y.reshape(-1)
            
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy for masked tokens
            predictions = torch.argmax(logits, dim=1)
            masked_indices = (batch_y != -100)
            if masked_indices.sum() > 0:
                correct = (predictions[masked_indices] == batch_y[masked_indices]).sum().item()
                total_correct += correct
                total_masked += masked_indices.sum().item()
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        avg_loss = total_loss / len(train_loader)
        train_accuracy = total_correct / total_masked if total_masked > 0 else 0.0
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.4f}, Time: {epoch_time:.2f}s")
    
    # Record end time and memory
    end_time = time.time()
    final_memory = get_memory_usage()
    final_gpu_memory = get_gpu_memory_usage()
    
    # Generate embeddings for quality analysis
    print("Generating embeddings for quality analysis...")
    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(torch.tensor(tokenized).to(device)).cpu().numpy()
    
    # Calculate embedding quality metrics
    quality_metrics = calculate_embedding_quality(embeddings)
    
    # Compile results
    results = {
        'embed_dim': embed_dim,
        'training_time_total': float(end_time - start_time),
        'training_time_per_epoch': float(np.mean(epoch_times)),
        'memory_usage_initial_mb': float(initial_memory),
        'memory_usage_final_mb': float(final_memory),
        'memory_usage_peak_mb': float(max(initial_memory, final_memory)),
        'gpu_memory_initial_mb': float(initial_gpu_memory),
        'gpu_memory_final_mb': float(final_gpu_memory),
        'gpu_memory_peak_mb': float(max(initial_gpu_memory, final_gpu_memory)),
        'final_loss': float(train_losses[-1]),
        'final_accuracy': float(train_accuracies[-1]),
        'loss_trajectory': [float(x) for x in train_losses],
        'accuracy_trajectory': [float(x) for x in train_accuracies],
        'epoch_times': [float(x) for x in epoch_times],
        'embedding_quality': quality_metrics,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    print(f"\nResults Summary for {embed_dim}D:")
    print(f"  Total Training Time: {results['training_time_total']:.2f}s")
    print(f"  Average Epoch Time: {results['training_time_per_epoch']:.2f}s")
    print(f"  Peak Memory Usage: {results['memory_usage_peak_mb']:.1f}MB")
    print(f"  Peak GPU Memory: {results['gpu_memory_peak_mb']:.1f}MB")
    print(f"  Final Loss: {results['final_loss']:.4f}")
    print(f"  Final Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Model Parameters: {results['model_parameters']:,}")
    print(f"  Embedding Quality - Silhouette: {results['embedding_quality']['silhouette_score']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Compare different embedding dimensions")
    parser.add_argument("--corpus_path", default="data/your_corpus.csv", help="Path to corpus CSV")
    parser.add_argument("--output_dir", default="results/architecture_comparison", help="Output directory")
    parser.add_argument("--embed_dims", nargs='+', type=int, default=[64, 128, 256], help="Embedding dimensions to test")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--max_length", type=int, default=10, help="Maximum sequence length")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("Loading domain sequences...")
    sequences = load_domain_sequences(args.corpus_path)
    domain2idx = build_domain_vocab(sequences)
    
    print(f"Loaded {len(sequences)} sequences with {len(domain2idx)} unique domains")
    
    # Test different architectures
    all_results = []
    
    for embed_dim in args.embed_dims:
        try:
            results = train_and_evaluate_model(
                embed_dim=embed_dim,
                sequences=sequences,
                domain2idx=domain2idx,
                max_length=args.max_length,
                num_epochs=args.num_epochs,
                batch_size=args.batch_size
            )
            all_results.append(results)
        except Exception as e:
            print(f"Error testing {embed_dim}D: {e}")
            continue
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save as JSON
    json_path = os.path.join(args.output_dir, f"comparison_results_{timestamp}.json")
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Save as text report
    txt_path = os.path.join(args.output_dir, f"comparison_report_{timestamp}.txt")
    with open(txt_path, 'w') as f:
        f.write("DOMAINBERT ARCHITECTURE COMPARISON REPORT\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Dataset: {len(sequences)} sequences, {len(domain2idx)} domains\n")
        f.write(f"Training: {args.num_epochs} epochs, batch size {args.batch_size}\n\n")
        
        for results in all_results:
            f.write(f"EMBEDDING DIMENSION: {results['embed_dim']}D\n")
            f.write("-" * 30 + "\n")
            f.write(f"Training Time: {results['training_time_total']:.2f}s total, {results['training_time_per_epoch']:.2f}s/epoch\n")
            f.write(f"Memory Usage: {results['memory_usage_peak_mb']:.1f}MB peak\n")
            f.write(f"GPU Memory: {results['gpu_memory_peak_mb']:.1f}MB peak\n")
            f.write(f"Model Parameters: {results['model_parameters']:,}\n")
            f.write(f"Final Loss: {results['final_loss']:.4f}\n")
            f.write(f"Final Accuracy: {results['final_accuracy']:.4f}\n")
            f.write(f"Embedding Quality:\n")
            f.write(f"  - Silhouette Score: {results['embedding_quality']['silhouette_score']:.4f}\n")
            f.write(f"  - Calinski-Harabasz: {results['embedding_quality']['calinski_harabasz_score']:.4f}\n")
            f.write(f"  - Mean Norm: {results['embedding_quality']['mean_norm']:.4f}\n")
            f.write(f"  - Sparsity: {results['embedding_quality']['sparsity']:.4f}\n")
            f.write("\n")
        
        # Summary comparison
        f.write("SUMMARY COMPARISON\n")
        f.write("=" * 20 + "\n")
        
        if all_results:
            fastest = min(all_results, key=lambda x: x['training_time_total'])
            best_accuracy = max(all_results, key=lambda x: x['final_accuracy'])
            best_quality = max(all_results, key=lambda x: x['embedding_quality']['silhouette_score'])
            
            f.write(f"Fastest Training: {fastest['embed_dim']}D ({fastest['training_time_total']:.2f}s)\n")
            f.write(f"Best Accuracy: {best_accuracy['embed_dim']}D ({best_accuracy['final_accuracy']:.4f})\n")
            f.write(f"Best Embedding Quality: {best_quality['embed_dim']}D (Silhouette: {best_quality['embedding_quality']['silhouette_score']:.4f})\n")
    
    # Create comparison table
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(args.output_dir, f"comparison_table_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    
    print(f"\n{'='*60}")
    print("COMPARISON COMPLETE!")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")
    print(f"JSON: {json_path}")
    print(f"Text Report: {txt_path}")
    print(f"CSV Table: {csv_path}")
    
    # Print summary
    if all_results:
        print(f"\nSummary:")
        for results in all_results:
            print(f"  {results['embed_dim']}D: {results['training_time_total']:.1f}s, "
                  f"Acc: {results['final_accuracy']:.3f}, "
                  f"Quality: {results['embedding_quality']['silhouette_score']:.3f}")

if __name__ == "__main__":
    main() 