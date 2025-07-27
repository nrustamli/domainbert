"""
Data loading utilities for domain-annotated protein sequences.
"""

import csv
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_domain_sequences(csv_path: str) -> List[Tuple[str, List[str]]]:
    """
    Load domain sequences from CSV file.
    
    Args:
        csv_path: Path to CSV file containing domain sequences
        
    Returns:
        List of tuples: (protein_code, [domain1, domain2, ...])
        
    Expected CSV format:
        protein_code:domain1,domain2,domain3,...
        A0A001:3.40.50.300_0_0,UNK_0_1,mobidb-lite_2
        A0A005:3.40.50_0_0,UAS,3.40.50.2000_0_1
    """
    sequences = []
    
    try:
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            for row_num, row in enumerate(reader, 1):
                if not row or not row[0].strip():
                    continue
                    
                try:
                    # Split on first colon to separate protein code from domains
                    if ':' not in row[0]:
                        logger.warning(f"Row {row_num}: No colon found, skipping: {row[0]}")
                        continue
                        
                    protein, domains_part = row[0].split(':', 1)
                    protein = protein.strip()
                    
                    # Split domains and clean them
                    domain_tokens = []
                    for domain in domains_part.split(','):
                        domain = domain.strip()
                        if domain:
                            # Remove suffixes like _0_0, _0_1 if present
                            clean_domain = domain.split('_')[0]
                            domain_tokens.append(clean_domain)
                    
                    if domain_tokens:
                        sequences.append((protein, domain_tokens))
                    else:
                        logger.warning(f"Row {row_num}: No valid domains found: {row[0]}")
                        
                except Exception as e:
                    logger.error(f"Error processing row {row_num}: {row[0]}, Error: {e}")
                    continue
                    
    except FileNotFoundError:
        logger.error(f"File not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading file {csv_path}: {e}")
        raise
    
    logger.info(f"Successfully loaded {len(sequences)} sequences from {csv_path}")
    return sequences

def load_domain_sequences_with_metadata(csv_path: str, 
                                       metadata_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Load domain sequences with additional metadata.
    
    Args:
        csv_path: Path to CSV file
        metadata_columns: List of column names to include as metadata
        
    Returns:
        DataFrame with columns: protein_code, domain_sequence, domain_list, [metadata_cols...]
    """
    sequences = load_domain_sequences(csv_path)
    
    # Create DataFrame
    data = []
    for protein_code, domains in sequences:
        row = {
            'protein_code': protein_code,
            'domain_sequence': ','.join(domains),
            'domain_list': domains,
            'num_domains': len(domains)
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # If additional metadata columns are specified, try to load them
    if metadata_columns:
        try:
            metadata_df = pd.read_csv(csv_path)
            for col in metadata_columns:
                if col in metadata_df.columns:
                    df[col] = metadata_df[col].values[:len(df)]
        except Exception as e:
            logger.warning(f"Could not load metadata columns: {e}")
    
    return df

def validate_domain_sequences(sequences: List[Tuple[str, List[str]]]) -> List[Tuple[str, List[str]]]:
    """
    Validate and filter domain sequences.
    
    Args:
        sequences: List of (protein_code, domains) tuples
        
    Returns:
        Filtered list of valid sequences
    """
    valid_sequences = []
    
    for protein_code, domains in sequences:
        # Check if protein code is valid
        if not protein_code or len(protein_code.strip()) == 0:
            continue
            
        # Check if domains list is not empty
        if not domains:
            continue
            
        # Check if all domains are valid (non-empty strings)
        valid_domains = [d for d in domains if d and len(d.strip()) > 0]
        
        if valid_domains:
            valid_sequences.append((protein_code.strip(), valid_domains))
    
    logger.info(f"Validated {len(valid_sequences)} sequences from {len(sequences)} total")
    return valid_sequences

def get_sequence_statistics(sequences: List[Tuple[str, List[str]]]) -> Dict:
    """
    Calculate statistics about the domain sequences.
    
    Args:
        sequences: List of (protein_code, domains) tuples
        
    Returns:
        Dictionary with statistics
    """
    if not sequences:
        return {}
    
    domain_lengths = [len(domains) for _, domains in sequences]
    all_domains = [domain for _, domains in sequences for domain in domains]
    unique_domains = set(all_domains)
    
    stats = {
        'total_sequences': len(sequences),
        'total_domains': len(all_domains),
        'unique_domains': len(unique_domains),
        'avg_sequence_length': np.mean(domain_lengths),
        'min_sequence_length': np.min(domain_lengths),
        'max_sequence_length': np.max(domain_lengths),
        'std_sequence_length': np.std(domain_lengths)
    }
    
    return stats