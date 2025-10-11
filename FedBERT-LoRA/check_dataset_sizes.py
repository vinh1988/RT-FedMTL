#!/usr/bin/env python3
"""
Check the sizes of GLUE datasets being used in the experiment.
"""
from datasets import load_dataset

def check_dataset_sizes():
    print("=== GLUE Dataset Sizes ===")
    
    # Check SST-2
    try:
        sst2 = load_dataset("glue", "sst2")
        print(f"SST-2: {len(sst2['train'])} training samples")
        print(f"      Validation: {len(sst2['validation'])} samples")
        print(f"      Test: {len(sst2['test'])} samples")
    except Exception as e:
        print(f"Error loading SST-2: {e}")
    
    print("\n---")
    
    # Check QQP
    try:
        qqp = load_dataset("glue", "qqp")
        print(f"QQP: {len(qqp['train'])} training samples")
        print(f"    Validation: {len(qqp['validation'])} samples")
        print(f"    Test: {len(qqp['test'])} samples")
    except Exception as e:
        print(f"Error loading QQP: {e}")
    
    print("\n---")
    
    # Check STSB
    try:
        stsb = load_dataset("glue", "stsb")
        print(f"STSB: {len(stsb['train'])} training samples")
        print(f"     Validation: {len(stsb['validation'])} samples")
        print(f"     Test: {len(stsb['test'])} samples")
    except Exception as e:
        print(f"Error loading STSB: {e}")

if __name__ == "__main__":
    check_dataset_sizes()
