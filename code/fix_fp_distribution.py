"""
Fix FP Distribution: Re-collect per-probe head counts for Fig 6.

The original experiment4 only saved 3 categories (none_fp, mixed, all_agree_fp).
The figure generation code then FABRICATED the 1/2/3 head breakdown using 
arbitrary weights (55%/30%/15%). This script re-runs just the FP distribution
measurement to get the real per-probe head counts.

Run this, then re-run generate_figures.py (which will be updated to use the 
new data).
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
import json
from collections import Counter

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)

BLOOM_HEADS = [(0, 1), (0, 5), (1, 11), (3, 0)]
FP_THRESHOLD = 0.1  # Same threshold as experiment 4


def build_test_sequences(model, n_unique=50, n_probes=20, n_trials=30):
    """Same sequence construction as experiment 4."""
    sequences = []
    vocab_size = model.cfg.d_vocab
    
    for trial in range(n_trials):
        all_tokens = torch.randperm(vocab_size - 1000) + 1000
        prefix_tokens = all_tokens[:n_unique]
        probe_tokens = all_tokens[n_unique:n_unique + n_probes]
        
        repeat_indices = torch.randperm(n_unique)[:min(10, n_unique)]
        repeat_tokens = prefix_tokens[repeat_indices]
        
        seq = torch.cat([
            torch.tensor([model.tokenizer.bos_token_id]),
            prefix_tokens,
            probe_tokens,
            repeat_tokens
        ]).unsqueeze(0).to(device)
        
        sequences.append({
            "tokens": seq,
            "prefix_len": n_unique,
            "probe_start": 1 + n_unique,
            "probe_end": 1 + n_unique + n_probes,
        })
    
    return sequences


print("\nBuilding test sequences (50 unique tokens, 20 probes × 30 trials = 600 probes)...")
sequences = build_test_sequences(model, n_unique=50, n_probes=20, n_trials=30)

print("Collecting per-probe head fire counts...")

# For each probe token, count how many Bloom heads fire (FP)
head_fire_counts = []  # List of ints, one per probe token

for seq_idx, seq_info in enumerate(sequences):
    if seq_idx % 10 == 0:
        print(f"  Sequence {seq_idx}/{len(sequences)}...")
    
    tokens = seq_info["tokens"]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    for probe_pos in range(seq_info["probe_start"], min(seq_info["probe_end"], tokens.shape[1])):
        n_heads_firing = 0
        for layer, head in BLOOM_HEADS:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head].cpu()
            prefix_attn = pattern[probe_pos, 1:1 + seq_info["prefix_len"]].sum().item()
            if prefix_attn > FP_THRESHOLD:
                n_heads_firing += 1
        head_fire_counts.append(n_heads_firing)

# Compute distribution
count_distribution = Counter(head_fire_counts)
total = len(head_fire_counts)

print(f"\n{'='*60}")
print(f"FP DISTRIBUTION (n={total} probe tokens)")
print(f"{'='*60}")

for k in range(5):
    count = count_distribution.get(k, 0)
    pct = 100 * count / total
    bar = "█" * int(pct)
    print(f"  {k} heads fire: {count:>4} ({pct:5.1f}%) {bar}")

# Save results
results = {
    "experiment": "fp_distribution_fix",
    "model": "gpt2-small",
    "bloom_heads": [f"L{h[0]}H{h[1]}" for h in BLOOM_HEADS],
    "fp_threshold": FP_THRESHOLD,
    "n_probes": total,
    "n_trials": 30,
    "n_unique_tokens": 50,
    "n_probes_per_trial": 20,
    "distribution": {str(k): count_distribution.get(k, 0) for k in range(5)},
    "distribution_pct": {str(k): round(100 * count_distribution.get(k, 0) / total, 1) for k in range(5)},
    "raw_counts": head_fire_counts,
}

results_path = "/Users/pabalogh/clawd/projects/bloom-filter-heads/results/fp_distribution_fix.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {results_path}")
print("\nNext step: update generate_figures.py to use this data for fig6, then re-run.")

# Also update experiment4 results with the real phi matrix
print("\n\nAlso re-computing phi matrix for experiment 4 (it wasn't saved originally)...")

# Collect binary FP decisions per head per probe
head_decisions = {h: [] for h in BLOOM_HEADS}

for seq_idx, seq_info in enumerate(sequences):
    tokens = seq_info["tokens"]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens)
    
    for probe_pos in range(seq_info["probe_start"], min(seq_info["probe_end"], tokens.shape[1])):
        for layer, head in BLOOM_HEADS:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head].cpu()
            prefix_attn = pattern[probe_pos, 1:1 + seq_info["prefix_len"]].sum().item()
            head_decisions[(layer, head)].append(int(prefix_attn > FP_THRESHOLD))

# Compute phi matrix
from itertools import combinations

phi_matrix = {}
for (h1, h2) in combinations(BLOOM_HEADS, 2):
    d1 = np.array(head_decisions[h1])
    d2 = np.array(head_decisions[h2])
    
    n11 = np.sum((d1 == 1) & (d2 == 1))
    n10 = np.sum((d1 == 1) & (d2 == 0))
    n01 = np.sum((d1 == 0) & (d2 == 1))
    n00 = np.sum((d1 == 0) & (d2 == 0))
    
    denom = np.sqrt(float((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01)))
    phi = float(n11 * n00 - n10 * n01) / denom if denom > 0 else 0.0
    
    key = f"L{h1[0]}H{h1[1]} <-> L{h2[0]}H{h2[1]}"
    phi_matrix[key] = round(phi, 3)
    print(f"  {key}: φ = {phi:.3f}")

avg_phi = np.mean(list(phi_matrix.values()))
print(f"\n  Average pairwise φ: {avg_phi:.3f}")

# Save phi matrix
results["phi_matrix"] = phi_matrix
results["avg_phi"] = round(avg_phi, 4)

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nUpdated results saved to {results_path}")
print("\n✅ Done. Run this script, then regenerate figures.")
