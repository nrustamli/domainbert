#!/usr/bin/env python3
"""
Fetch GO labels using UniProt's GO download.
"""

import csv
import requests
import time
import os

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
    """Try to get GO terms using different UniProt endpoints."""
    
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
                return go_terms
    except:
        pass
    
    # Method 2: Try the new REST API with different fields
    try:
        url = f"https://rest.uniprot.org/uniprotkb/{protein}"
        params = {"format": "json"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            go_terms = []
            
            # Look in crossReferences
            if 'uniProtKBCrossReferences' in data:
                for ref in data['uniProtKBCrossReferences']:
                    if ref.get('database') == 'GO':
                        go_id = ref.get('id')
                        if go_id and go_id.startswith('GO:'):
                            go_terms.append(go_id)
            
            if go_terms:
                return go_terms
    except:
        pass
    
    return []

def main():
    print("Starting GO label fetching...")
    
    proteins = read_proteins()
    print(f"Found {len(proteins)} proteins")
    
    # Test with first 10
    proteins = proteins[:10]
    print(f"Testing with first {len(proteins)} proteins")
    
    results = {}
    for i, protein in enumerate(proteins):
        print(f"\nProcessing {i+1}/{len(proteins)}: {protein}")
        
        go_terms = fetch_go_terms(protein)
        if go_terms:
            results[protein] = go_terms
            print(f"  ✓ Found {len(go_terms)} GO terms: {go_terms}")
        else:
            print(f"  ⚠ No GO terms found")
        
        time.sleep(0.5)
    
    # Save results
    print("\nSaving results...")
    os.makedirs("data", exist_ok=True)
    
    with open("data/go_labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["protein_code", "go_terms"])
        
        for protein in proteins:
            terms = results.get(protein, [])
            go_string = ";".join(terms) if terms else ""
            w.writerow([protein, go_string])
    
    covered = len([p for p in proteins if p in results])
    print(f"\nDone! {covered}/{len(proteins)} proteins have GO terms")
    print(f"Saved to: data/go_labels.csv")

if __name__ == "__main__":
    main() 