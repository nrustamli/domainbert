"""
Utility functions for DomainBERT.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.metrics import pairwise_distances
from sklearn.manifold import TSNE
import umap

logger = logging.getLogger(__name__)

def set_random_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the best available device (GPU or CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    return device

def save_model(model: torch.nn.Module, filepath: str, **kwargs):
    """Save model with additional metadata."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'vocab_size': model.vocab_size,
            'embed_dim': model.embed_dim,
            'num_layers': model.num_layers,
            'num_heads': model.num_heads,
            'max_length': model.max_length
        },
        **kwargs
    }
    torch.save(save_dict, filepath)
    logger.info(f"Model saved to {filepath}")

def load_model(model_class, filepath: str, **kwargs) -> torch.nn.Module:
    """Load model from file."""
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Create model with saved config
    model_config = checkpoint['model_config']
    model = model_class(**model_config, **kwargs)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    logger.info(f"Model loaded from {filepath}")
    return model

def compute_similarity_matrix(embeddings: np.ndarray, 
                            metric: str = 'cosine') -> np.ndarray:
    """Compute pairwise similarity matrix between embeddings."""
    return 1 - pairwise_distances(embeddings, metric=metric)

def find_nearest_neighbors(embeddings: np.ndarray, 
                          query_idx: int, 
                          k: int = 5,
                          metric: str = 'cosine') -> Tuple[np.ndarray, np.ndarray]:
    """Find k nearest neighbors for a given embedding."""
    similarities = compute_similarity_matrix(embeddings, metric)
    query_similarities = similarities[query_idx]
    
    # Get top k indices (excluding self)
    nearest_indices = np.argsort(query_similarities)[::-1][1:k+1]
    nearest_similarities = query_similarities[nearest_indices]
    
    return nearest_indices, nearest_similarities

def visualize_embeddings(embeddings: np.ndarray, 
                        labels: Optional[np.ndarray] = None,
                        method: str = 'tsne',
                        n_components: int = 2,
                        **kwargs) -> np.ndarray:
    """Visualize embeddings using dimensionality reduction."""
    
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=42, **kwargs)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=n_components, random_state=42, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Plot if labels are provided
    if labels is not None:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], 
                             c=labels, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter)
        plt.title(f'Domain Embeddings ({method.upper()})')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.tight_layout()
        plt.show()
    
    return reduced_embeddings

def analyze_embedding_quality(embeddings: np.ndarray, 
                            labels: Optional[np.ndarray] = None) -> Dict:
    """Analyze the quality of embeddings."""
    
    # Basic statistics
    stats = {
        'num_embeddings': len(embeddings),
        'embedding_dim': embeddings.shape[1],
        'mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
        'std_norm': np.std(np.linalg.norm(embeddings, axis=1)),
        'min_norm': np.min(np.linalg.norm(embeddings, axis=1)),
        'max_norm': np.max(np.linalg.norm(embeddings, axis=1))
    }
    
    # Compute pairwise distances
    distances = pairwise_distances(embeddings, metric='euclidean')
    stats.update({
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances[distances > 0]),
        'max_distance': np.max(distances)
    })
    
    # If labels are provided, compute intra/inter-class distances
    if labels is not None:
        unique_labels = np.unique(labels)
        intra_distances = []
        inter_distances = []
        
        for label in unique_labels:
            label_mask = labels == label
            label_embeddings = embeddings[label_mask]
            
            # Intra-class distances
            if len(label_embeddings) > 1:
                label_distances = pairwise_distances(label_embeddings, metric='euclidean')
                intra_distances.extend(label_distances[label_distances > 0])
            
            # Inter-class distances
            other_embeddings = embeddings[~label_mask]
            if len(other_embeddings) > 0:
                inter_distances.extend(pairwise_distances(label_embeddings, other_embeddings, metric='euclidean').flatten())
        
        if intra_distances:
            stats['mean_intra_distance'] = np.mean(intra_distances)
            stats['std_intra_distance'] = np.std(intra_distances)
        
        if inter_distances:
            stats['mean_inter_distance'] = np.mean(inter_distances)
            stats['std_inter_distance'] = np.std(inter_distances)
        
        if intra_distances and inter_distances:
            stats['separation_ratio'] = np.mean(inter_distances) / np.mean(intra_distances)
    
    return stats

def create_embedding_report(embeddings: np.ndarray, 
                          metadata: Optional[pd.DataFrame] = None,
                          output_file: Optional[str] = None) -> str:
    """Create a comprehensive report about embeddings."""
    
    report_lines = []
    report_lines.append("=" * 50)
    report_lines.append("DOMAIN EMBEDDING ANALYSIS REPORT")
    report_lines.append("=" * 50)
    
    # Basic statistics
    stats = analyze_embedding_quality(embeddings)
    report_lines.append(f"\nBasic Statistics:")
    report_lines.append(f"  Number of embeddings: {stats['num_embeddings']}")
    report_lines.append(f"  Embedding dimension: {stats['embedding_dim']}")
    report_lines.append(f"  Mean norm: {stats['mean_norm']:.4f}")
    report_lines.append(f"  Std norm: {stats['std_norm']:.4f}")
    
    # Distance statistics
    report_lines.append(f"\nDistance Statistics:")
    report_lines.append(f"  Mean pairwise distance: {stats['mean_distance']:.4f}")
    report_lines.append(f"  Std pairwise distance: {stats['std_distance']:.4f}")
    report_lines.append(f"  Min distance: {stats['min_distance']:.4f}")
    report_lines.append(f"  Max distance: {stats['max_distance']:.4f}")
    
    # Metadata statistics if available
    if metadata is not None:
        report_lines.append(f"\nMetadata Statistics:")
        for col in metadata.columns:
            if metadata[col].dtype == 'object':
                unique_vals = metadata[col].nunique()
                report_lines.append(f"  {col}: {unique_vals} unique values")
            else:
                mean_val = metadata[col].mean()
                std_val = metadata[col].std()
                report_lines.append(f"  {col}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    report = "\n".join(report_lines)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        logger.info(f"Embedding report saved to {output_file}")
    
    return report