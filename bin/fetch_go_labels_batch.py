#!/usr/bin/env python3
"""
Fetch GO labels using batch processing for speed.
"""

import csv
import requests
import time
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_proteins():
    """Read protein accessions from your corpus."""
    proteins = []
    with open("data/your_corpus.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row and ":" in row[0]:
                acc = row[0].split(":", 1)[0].strip()
                if acc:
                    proteins.append(acc)
    return list(dict.fromkeys(proteins))

def fetch_go_terms(protein):
    """Fetch GO terms for a single protein."""
    
    # Method 1: Try the old-style endpoint
    try:
        url = f"https://www.uniprot.org/uniprot/{protein}.txt"
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            lines = r.text.split('\n')
            go_terms = []
            for line in lines:
                if line.startswith('DR   GO'):
                    parts = line.split(';')
                    if len(parts) >= 2:
                        go_id = parts[1].strip()
                        if go_id.startswith('GO:'):
                            go_terms.append(go_id)
            if go_terms:
                return protein, go_terms
    except:
        pass
    
    # Method 2: Try the new REST API
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{protein}"
        params = {"format": "json"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            go_terms = []
            
            if 'uniProtKBCrossReferences' in data:
                for ref in data['uniProtKBCrossReferences']:
                    if ref.get('database') == 'GO':
                        go_id = ref.get('id')
                        if go_id and go_id.startswith('GO:'):
                            go_terms.append(go_id)
            
            if go_terms:
                return protein, go_terms
    except:
        pass
    
    return protein, []

def process_batch(proteins_batch):
    """Process a batch of proteins."""
    results = {}
    for protein in proteins_batch:
        protein, go_terms = fetch_go_terms(protein)
        results[protein] = go_terms
        time.sleep(0.05)  # Small delay between requests
    return results

def main():
    print("Starting BATCH GO label fetching...")
    start_time = datetime.now()
    
    proteins = read_proteins()
    total_proteins = len(proteins)
    print(f"Found {total_proteins} proteins")
    
    # Process in batches
    batch_size = 100
    total_batches = (total_proteins + batch_size - 1) // batch_size
    
    all_results = {}
    successful = 0
    failed = 0
    
    print(f"Processing in {total_batches} batches of {batch_size} proteins...")
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        end_idx = min(start_idx + batch_size, total_proteins)
        batch_proteins = proteins[start_idx:end_idx]
        
        print(f"\n�� Processing batch {batch_num + 1}/{total_batches} (proteins {start_idx + 1}-{end_idx})...")
        
        # Process batch
        batch_results = process_batch(batch_proteins)
        all_results.update(batch_results)
        
        # Count results
        batch_successful = sum(1 for terms in batch_results.values() if terms)
        batch_failed = len(batch_results) - batch_successful
        successful += batch_successful
        failed += batch_failed
        
        # Progress update
        elapsed = datetime.now() - start_time
        rate = (end_idx) / elapsed.total_seconds() if elapsed.total_seconds() > 0 else 0
        eta = (total_proteins - end_idx) / rate if rate > 0 else 0
        
        print(f"  Batch complete: {batch_successful} with GO terms, {batch_failed} without")
        print(f"  Overall: {successful}/{end_idx} ({successful/end_idx*100:.1f}%)")
        print(f"  Rate: {rate:.1f} proteins/sec, ETA: {eta/60:.1f} minutes")
        
        # Save checkpoint every 5 batches
        if (batch_num + 1) % 5 == 0:
            print(f"  💾 Saving checkpoint...")
            save_checkpoint(all_results, end_idx)
        
        # Be nice to the API between batches
        time.sleep(1)
    
    # Final save
    print("\n💾 Saving final results...")
    save_final_results(all_results, total_proteins)
    
    # Summary
    elapsed = datetime.now() - start_time
    print(f"\n{'='*60}")
    print("BATCH FETCHING COMPLETE!")
    print(f"{'='*60}")
    print(f"Total proteins processed: {total_proteins}")
    print(f"Proteins with GO terms: {successful}")
    print(f"Proteins without GO terms: {failed}")
    print(f"Coverage: {(successful/total_proteins)*100:.1f}%")
    print(f"Total time: {elapsed.total_seconds()/3600:.1f} hours")
    print(f"Average rate: {total_proteins/elapsed.total_seconds():.1f} proteins/sec")

def save_checkpoint(results, checkpoint_num):
    """Save intermediate results."""
    os.makedirs("data", exist_ok=True)
    checkpoint_file = f"data/go_labels_checkpoint_{checkpoint_num}.csv"
    
    with open(checkpoint_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protein_code", "go_terms"])
        
        for protein, terms in results.items():
            go_string = ";".join(terms) if terms else ""
            w.writerow([protein, go_string])

def save_final_results(results, total_proteins):
    """Save final results."""
    os.makedirs("data", exist_ok=True)
    
    # Read all proteins to maintain order
    proteins = read_proteins()
    
    with open("data/go_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protein_code", "go_terms"])
        
        for protein in proteins:
            terms = results.get(protein, [])
            go_string = ";".join(terms) if terms else ""
            w.writerow([protein, go_string])

if __name__ == "__main__":
    main() 