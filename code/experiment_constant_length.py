"""
EXP-4: Constant-Length Capacity Control
Hold sequence length constant at 200 tokens, vary the proportion that are unique.
This controls for the sequence-length confound in the original capacity experiment.
"""
import json
import numpy as np

# Check if we already have this data
try:
    with open('../results/experiment7_capacity_confound_control.json') as f:
        data = json.load(f)
    print("=== EXISTING EXP-7 DATA (Capacity Confound Control) ===")
    print(f"Experiment: {data.get('experiment', 'unknown')}")
    
    if 'results' in data:
        for head, hdata in data.get('results', {}).items():
            print(f"\n{head}:")
            if isinstance(hdata, dict):
                for k, v in hdata.items():
                    print(f"  {k}: {v}")
    elif 'conditions' in data:
        print(f"Conditions: {data['conditions']}")
        for head in data.get('bloom_heads_results', data.get('heads', {})):
            print(f"\n{head}:")
            hdata = data.get('bloom_heads_results', data.get('heads', {}))[head] if isinstance(data.get('bloom_heads_results', data.get('heads', {})), dict) else {}
            for k, v in hdata.items():
                print(f"  {k}: {v}")
    else:
        # Just dump top-level keys
        for k in data:
            v = data[k]
            if isinstance(v, (list, dict)):
                print(f"{k}: {type(v).__name__}, len={len(v)}")
            else:
                print(f"{k}: {v}")
                
except FileNotFoundError:
    print("No existing data found. Need to run experiment.")
except Exception as e:
    print(f"Error reading: {e}")
