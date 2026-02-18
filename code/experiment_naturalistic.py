"""
EXP-5: Naturalistic validation of Bloom filter heads on WikiText-103.

Shows that the Bloom filter signature (high selectivity for repeated tokens,
near-zero miss rate) holds on natural text, not just constructed stimuli.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from datasets import load_dataset
import json
import os
from collections import defaultdict

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
tokenizer = model.tokenizer

# Bloom filter heads identified in prior experiments
BLOOM_HEADS = {
    "L0H1": (0, 1),
    "L0H5": (0, 5),
    "L1H11": (1, 11),
    "L3H0": (3, 0),
}

# Also track some non-Bloom heads as controls
CONTROL_HEADS = {
    "L0H0": (0, 0),
    "L0H3": (0, 3),
    "L2H2": (2, 2),
}

ALL_HEADS = {**BLOOM_HEADS, **CONTROL_HEADS}

# ============================================================
# Load WikiText-103 validation
# ============================================================
print("Loading WikiText-103 validation set...")
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

# Filter to passages with enough text (at least 50 chars)
passages = [t for t in ds["text"] if len(t.strip()) > 50]
print(f"Got {len(passages)} non-empty passages, using up to 800")
passages = passages[:800]

# ============================================================
# Analysis
# ============================================================

MAX_SEQ_LEN = 256  # truncate long passages

# Per-head accumulators
head_stats = {name: {
    "repeated_attn": [],      # attention from 2nd occurrence -> 1st occurrence
    "non_repeated_attn": [],  # attention from 2nd occurrence -> non-matching positions
    "miss_count": 0,          # times repeated attn < median non-repeated attn
    "hit_count": 0,           # times we checked
} for name in ALL_HEADS}

n_processed = 0

for pi, passage in enumerate(passages):
    tokens = tokenizer.encode(passage, return_tensors="pt")[0]
    if len(tokens) < 10:
        continue
    tokens = tokens[:MAX_SEQ_LEN]
    
    # Find repeated tokens: for each position, find if it appeared earlier
    token_list = tokens.tolist()
    # Map: token_id -> list of positions
    token_positions = defaultdict(list)
    for i, t in enumerate(token_list):
        token_positions[t].append(i)
    
    # Build pairs: (second_pos, first_pos) for naturally repeated tokens
    repeat_pairs = []
    for tid, positions in token_positions.items():
        if len(positions) >= 2:
            # Each later occurrence paired with first occurrence
            for p in positions[1:]:
                repeat_pairs.append((p, positions[0]))
    
    if len(repeat_pairs) < 3:
        continue  # need enough repeats
    
    # Run model
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens.unsqueeze(0).to(device))
    
    for head_name, (layer, head) in ALL_HEADS.items():
        attn = cache["pattern", layer][0, head].cpu().numpy()  # [seq, seq]
        
        for second_pos, first_pos in repeat_pairs:
            # Attention from second occurrence to first occurrence
            repeated_a = float(attn[second_pos, first_pos])
            
            # Attention from second occurrence to all OTHER positions (not first, not self)
            # attn[second_pos] has shape [seq_len], but causal mask means only [:second_pos+1] matters
            # We want positions < second_pos, excluding first_pos
            row = attn[second_pos, :second_pos]  # all causal positions before second_pos
            mask = np.ones(second_pos, dtype=bool)
            mask[first_pos] = False
            if mask.sum() == 0:
                continue
            non_repeated_a = float(row[mask].mean())
            
            head_stats[head_name]["repeated_attn"].append(repeated_a)
            head_stats[head_name]["non_repeated_attn"].append(non_repeated_a)
            
            # Miss: repeated attention is less than median of non-repeated
            median_nr = float(np.median(row[mask]))
            if repeated_a < median_nr:
                head_stats[head_name]["miss_count"] += 1
            head_stats[head_name]["hit_count"] += 1
    
    n_processed += 1
    if (pi + 1) % 100 == 0:
        print(f"  Processed {pi+1}/{len(passages)} passages ({n_processed} usable)")

print(f"\nProcessed {n_processed} passages total")

# ============================================================
# Compute summary statistics
# ============================================================

results = {
    "experiment": "EXP-5: Naturalistic validation on WikiText-103",
    "n_passages": n_processed,
    "max_seq_len": MAX_SEQ_LEN,
    "device": device,
    "heads": {},
}

print("\n" + "=" * 70)
print(f"{'Head':<10} {'Selectivity':>12} {'Miss Rate':>12} {'Mean Rep':>10} {'Mean NR':>10} {'N pairs':>10}")
print("=" * 70)

for head_name in ALL_HEADS:
    s = head_stats[head_name]
    rep = np.array(s["repeated_attn"])
    nrep = np.array(s["non_repeated_attn"])
    
    mean_rep = float(rep.mean())
    mean_nrep = float(nrep.mean())
    selectivity = mean_rep / max(mean_nrep, 1e-10)
    miss_rate = s["miss_count"] / max(s["hit_count"], 1) * 100
    
    is_bloom = head_name in BLOOM_HEADS
    marker = " ***" if is_bloom else ""
    print(f"{head_name:<10} {selectivity:>12.1f}x {miss_rate:>11.1f}% {mean_rep:>10.4f} {mean_nrep:>10.4f} {len(rep):>10}{marker}")
    
    results["heads"][head_name] = {
        "is_bloom_head": is_bloom,
        "selectivity": round(selectivity, 2),
        "miss_rate_pct": round(miss_rate, 2),
        "mean_repeated_attn": round(mean_rep, 6),
        "mean_non_repeated_attn": round(mean_nrep, 6),
        "n_pairs": len(rep),
    }

print("=" * 70)
print("*** = Bloom filter head")

# Compare to constructed stimuli benchmarks
bloom_selectivities = [results["heads"][h]["selectivity"] for h in BLOOM_HEADS]
control_selectivities = [results["heads"][h]["selectivity"] for h in CONTROL_HEADS]

results["summary"] = {
    "bloom_heads_mean_selectivity": round(float(np.mean(bloom_selectivities)), 2),
    "control_heads_mean_selectivity": round(float(np.mean(control_selectivities)), 2),
    "bloom_heads_mean_miss_rate": round(float(np.mean([results["heads"][h]["miss_rate_pct"] for h in BLOOM_HEADS])), 2),
    "constructed_stimuli_selectivity_range": "51x-146x",
    "note": "Naturalistic selectivity expected to be lower due to function words, but should still be significantly above controls.",
}

print(f"\nBloom heads mean selectivity: {results['summary']['bloom_heads_mean_selectivity']}x")
print(f"Control heads mean selectivity: {results['summary']['control_heads_mean_selectivity']}x")
print(f"Bloom heads mean miss rate: {results['summary']['bloom_heads_mean_miss_rate']}%")

# Save
os.makedirs("results", exist_ok=True)
with open("results/naturalistic_validation.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nResults saved to results/naturalistic_validation.json")
