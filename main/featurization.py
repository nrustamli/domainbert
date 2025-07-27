"""
Featurization utilities for domain sequences.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
import logging
from collections import Counter

logger = logging.getLogger(__name__)

def build_domain_vocab(sequences: List[Tuple[str, List[str]]], 
                      min_freq: int = 1,
                      max_vocab_size: Optional[int] = None) -> Dict[str, int]:
    """
    Build vocabulary from domain sequences.
    
    Args:
        sequences: List of (protein_code, domains) tuples
        min_freq: Minimum frequency for a domain to be included in vocabulary
        max_vocab_size: Maximum vocabulary size (None for no limit)
        
    Returns:
        Dictionary mapping domain tokens to indices
    """
    # Count domain frequencies
    domain_counter = Counter()
    for _, domains in sequences:
        domain_counter.update(domains)
    
    # Filter by minimum frequency
    if min_freq > 1:
        domain_counter = {domain: count for domain, count in domain_counter.items() 
                         if count >= min_freq}
    
    # Sort by frequency (most frequent first)
    sorted_domains = sorted(domain_counter.items(), key=lambda x: x[1], reverse=True)
    
    # Limit vocabulary size if specified
    if max_vocab_size:
        sorted_domains = sorted_domains[:max_vocab_size]
    
    # Create vocabulary
    domain2idx = {domain: idx for idx, (domain, _) in enumerate(sorted_domains)}
    
    # Add special tokens
    special_tokens = {
        '[PAD]': len(domain2idx),
        '[MASK]': len(domain2idx) + 1,
        '[UNK]': len(domain2idx) + 2,
        '[CLS]': len(domain2idx) + 3,
        '[SEP]': len(domain2idx) + 4
    }
    domain2idx.update(special_tokens)
    
    logger.info(f"Built vocabulary with {len(domain2idx)} tokens")
    logger.info(f"Most frequent domains: {list(sorted_domains[:10])}")
    
    return domain2idx

def tokenize_sequences(sequences: List[Tuple[str, List[str]]], 
                      domain2idx: Dict[str, int],
                      max_length: Optional[int] = None,
                      pad_token: str = '[PAD]',
                      unk_token: str = '[UNK]',
                      cls_token: str = '[CLS]',
                      sep_token: str = '[SEP]') -> List[List[int]]:
    """
    Tokenize domain sequences to indices.
    
    Args:
        sequences: List of (protein_code, domains) tuples
        domain2idx: Domain to index mapping
        max_length: Maximum sequence length (None for no limit)
        pad_token: Token used for padding
        unk_token: Token used for unknown domains
        cls_token: Classification token
        sep_token: Separation token
        
    Returns:
        List of tokenized sequences (lists of indices)
    """
    tokenized = []
    
    for protein_code, domains in sequences:
        # Convert domains to indices
        indices = []
        for domain in domains:
            if domain in domain2idx:
                indices.append(domain2idx[domain])
            else:
                indices.append(domain2idx[unk_token])
        
        # Add special tokens if specified
        if cls_token in domain2idx:
            indices = [domain2idx[cls_token]] + indices
        
        if sep_token in domain2idx:
            indices = indices + [domain2idx[sep_token]]
        
        # Pad or truncate to max_length
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                padding_length = max_length - len(indices)
                indices = indices + [domain2idx[pad_token]] * padding_length
        
        tokenized.append(indices)
    
    logger.info(f"Tokenized {len(tokenized)} sequences")
    if max_length:
        logger.info(f"All sequences padded/truncated to length {max_length}")
    
    return tokenized

def create_attention_masks(tokenized_sequences: List[List[int]], 
                          pad_idx: int) -> List[List[int]]:
    """
    Create attention masks for tokenized sequences.
    
    Args:
        tokenized_sequences: List of tokenized sequences
        pad_idx: Index of padding token
        
    Returns:
        List of attention masks (1 for real tokens, 0 for padding)
    """
    attention_masks = []
    
    for sequence in tokenized_sequences:
        mask = [1 if token != pad_idx else 0 for token in sequence]
        attention_masks.append(mask)
    
    return attention_masks

def create_masked_sequences(tokenized_sequences: List[List[int]], 
                           mask_prob: float = 0.15,
                           mask_idx: int = 1,  # [MASK] token index
                           pad_idx: int = 0) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Create masked sequences for masked language modeling.
    
    Args:
        tokenized_sequences: List of tokenized sequences
        mask_prob: Probability of masking a token
        mask_idx: Index of mask token
        pad_idx: Index of padding token
        
    Returns:
        Tuple of (masked_sequences, labels)
    """
    masked_sequences = []
    labels = []
    
    for sequence in tokenized_sequences:
        masked_seq = sequence.copy()
        seq_labels = [-100] * len(sequence)  # -100 is ignored in loss computation
        
        for i, token in enumerate(sequence):
            # Don't mask padding tokens or special tokens
            if (token != pad_idx and 
                token not in [0, 1, 2, 3, 4] and  # Special token indices
                np.random.random() < mask_prob):
                
                seq_labels[i] = token  # Original token becomes the label
                masked_seq[i] = mask_idx  # Replace with mask token
        
        masked_sequences.append(masked_seq)
        labels.append(seq_labels)
    
    logger.info(f"Created masked sequences with mask probability {mask_prob}")
    return masked_sequences, labels

def get_vocabulary_statistics(domain2idx: Dict[str, int], 
                            sequences: List[Tuple[str, List[str]]]) -> Dict:
    """
    Calculate statistics about the vocabulary.
    
    Args:
        domain2idx: Domain to index mapping
        sequences: Original sequences
        
    Returns:
        Dictionary with vocabulary statistics
    """
    # Count domain frequencies
    domain_counter = Counter()
    for _, domains in sequences:
        domain_counter.update(domains)
    
    # Calculate coverage
    covered_domains = set(domain2idx.keys()) - {'[PAD]', '[MASK]', '[UNK]', '[CLS]', '[SEP]'}
    total_domains = set(domain_counter.keys())
    coverage = len(covered_domains.intersection(total_domains)) / len(total_domains)
    
    stats = {
        'vocab_size': len(domain2idx),
        'special_tokens': 5,  # [PAD], [MASK], [UNK], [CLS], [SEP]
        'domain_tokens': len(domain2idx) - 5,
        'total_unique_domains': len(total_domains),
        'vocabulary_coverage': coverage,
        'most_frequent_domains': domain_counter.most_common(10)
    }
    
    return stats

def save_vocabulary(domain2idx: Dict[str, int], filepath: str):
    """Save vocabulary to file."""
    import json
    with open(filepath, 'w') as f:
        json.dump(domain2idx, f, indent=2)
    logger.info(f"Vocabulary saved to {filepath}")

def load_vocabulary(filepath: str) -> Dict[str, int]:
    """Load vocabulary from file."""
    import json
    with open(filepath, 'r') as f:
        domain2idx = json.load(f)
    logger.info(f"Vocabulary loaded from {filepath}")
    return domain2idx