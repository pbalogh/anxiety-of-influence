"""
Experiment 3: Capacity Analysis — Do Bloom filter heads follow theoretical predictions?

A real Bloom filter with:
  - m bits
  - k hash functions  
  - n inserted elements
has false positive rate:  p ≈ (1 - e^(-kn/m))^k

For attention heads:
  - m ≈ d_head (head dimension, 64 for GPT-2)
  - k ≈ 1 (single QK dot product = single hash function)
  - n = number of unique tokens in context

If heads are Bloom filters:
  1. False positive rate should INCREASE with more unique tokens (capacity fills up)
  2. The rate should follow the theoretical curve
  3. Miss rate should remain ~0 regardless of load

We test by varying sequence length and unique token density.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from collections import defaultdict
import json
import math

device = "cpu"  # FORCED CPU
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
print(f"Model: {n_layers} layers, {n_heads} heads, d_head={d_head}")

# Our Bloom filter heads from Experiment 1
BLOOM_HEADS = [(0, 1), (1, 11), (3, 0), (0, 5)]
# Control heads (high induction, no Bloom behavior)
CONTROL_HEADS = [(5, 5), (7, 10), (6, 9)]

# ============================================================
# Experiment 3a: Capacity vs. false positive rate
# ============================================================

def build_capacity_test_sequences(model, n_unique_tokens, n_trials=20):
    """
    Build sequences with controlled unique token count.
    Each sequence has:
      - A prefix of n_unique_tokens unique tokens
      - A probe section where we test if NEW tokens get false-positive attention
        to positions in the prefix
      - A repeat section where we test if REPEATED tokens still get detected
    """
    sequences = []
    vocab_size = model.cfg.d_vocab
    
    for trial in range(n_trials):
        # Pick random unique tokens for the prefix (avoid special tokens)
        all_tokens = torch.randperm(vocab_size - 1000) + 1000
        prefix_tokens = all_tokens[:n_unique_tokens]
        
        # Pick tokens NOT in the prefix for probes (false positive test)
        probe_tokens = all_tokens[n_unique_tokens:n_unique_tokens + 5]
        
        # Pick tokens FROM the prefix for repeats (miss rate test)
        repeat_indices = torch.randperm(n_unique_tokens)[:min(5, n_unique_tokens)]
        repeat_tokens = prefix_tokens[repeat_indices]
        
        # Build sequence: [BOS] + prefix + probes + repeats
        seq = torch.cat([
            torch.tensor([model.tokenizer.bos_token_id]),
            prefix_tokens,
            probe_tokens,
            repeat_tokens
        ]).unsqueeze(0).to(device)
        
        sequences.append({
            "tokens": seq,
            "prefix_len": n_unique_tokens,
            "probe_start": 1 + n_unique_tokens,
            "probe_end": 1 + n_unique_tokens + 5,
            "repeat_start": 1 + n_unique_tokens + 5,
            "repeat_end": 1 + n_unique_tokens + 5 + len(repeat_tokens),
            "repeat_source_positions": (1 + repeat_indices).tolist(),  # positions in prefix
        })
    
    return sequences


def measure_fp_and_miss_rates(model, sequences, heads):
    """
    For each head, measure:
      - False positive rate: attention from probe tokens to prefix positions
        (these tokens are NOT in the prefix, so any attention is a false positive)
      - Miss rate: attention from repeat tokens to their original position
        (these tokens ARE in the prefix, so failure to attend is a miss)
    """
    results = defaultdict(lambda: {"fp_attentions": [], "hit_attentions": [], "baseline_attentions": []})
    
    for seq_info in sequences:
        tokens = seq_info["tokens"]
        _, cache = model.run_with_cache(tokens)
        
        for layer, head in heads:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head].cpu()  # [dest, src]
            
            # False positive: probe tokens attending to prefix
            for probe_pos in range(seq_info["probe_start"], min(seq_info["probe_end"], pattern.shape[0])):
                # Sum attention to all prefix positions
                prefix_attn = pattern[probe_pos, 1:1 + seq_info["prefix_len"]].sum().item()
                results[(layer, head)]["fp_attentions"].append(prefix_attn)
            
            # Hit rate: repeat tokens attending to their source position
            repeat_positions = list(range(seq_info["repeat_start"], min(seq_info["repeat_end"], pattern.shape[0])))
            source_positions = seq_info["repeat_source_positions"]
            
            for i, (rpt_pos, src_pos) in enumerate(zip(repeat_positions, source_positions)):
                if rpt_pos < pattern.shape[0] and src_pos < pattern.shape[1]:
                    hit_attn = pattern[rpt_pos, src_pos].item()
                    results[(layer, head)]["hit_attentions"].append(hit_attn)
            
            # Baseline: attention from probe tokens to a random single position
            for probe_pos in range(seq_info["probe_start"], min(seq_info["probe_end"], pattern.shape[0])):
                rand_src = np.random.randint(1, seq_info["prefix_len"] + 1)
                baseline_attn = pattern[probe_pos, rand_src].item()
                results[(layer, head)]["baseline_attentions"].append(baseline_attn)
    
    return results


# Test at different capacity levels
capacity_levels = [5, 10, 20, 30, 50, 75, 100, 150, 200]
all_heads = BLOOM_HEADS + CONTROL_HEADS

print("\n" + "="*60)
print("EXPERIMENT 3a: CAPACITY vs FALSE POSITIVE RATE")
print("="*60)

capacity_results = {}

for n_unique in capacity_levels:
    print(f"\nTesting with {n_unique} unique tokens in context...")
    sequences = build_capacity_test_sequences(model, n_unique, n_trials=20)
    measurements = measure_fp_and_miss_rates(model, sequences, all_heads)
    
    for head_key in all_heads:
        if head_key not in capacity_results:
            capacity_results[head_key] = {"n_unique": [], "fp_rate": [], "miss_rate": [], "mean_hit": []}
        
        m = measurements[head_key]
        fp_vals = m["fp_attentions"]
        hit_vals = m["hit_attentions"]
        
        # FP rate: fraction of probe tokens that attend >threshold to prefix
        fp_threshold = 0.1  # >10% attention to prefix = "false positive"
        fp_rate = sum(1 for v in fp_vals if v > fp_threshold) / len(fp_vals) if fp_vals else 0
        
        # Miss rate: fraction of repeat tokens that DON'T attend to source
        miss_threshold = 0.01
        miss_rate = sum(1 for v in hit_vals if v < miss_threshold) / len(hit_vals) if hit_vals else 0
        
        mean_hit = np.mean(hit_vals) if hit_vals else 0
        mean_fp = np.mean(fp_vals) if fp_vals else 0
        
        capacity_results[head_key]["n_unique"].append(n_unique)
        capacity_results[head_key]["fp_rate"].append(round(fp_rate, 4))
        capacity_results[head_key]["miss_rate"].append(round(miss_rate, 4))
        capacity_results[head_key]["mean_hit"].append(round(mean_hit, 4))

# ============================================================
# Theoretical Bloom filter prediction
# ============================================================

def theoretical_bloom_fp(n, m, k=1):
    """Theoretical false positive rate for Bloom filter."""
    return (1 - math.exp(-k * n / m)) ** k

# Compute theoretical curve using d_head as m
theoretical_fp = [theoretical_bloom_fp(n, d_head) for n in capacity_levels]

# ============================================================
# Print results
# ============================================================

print("\n" + "="*60)
print("RESULTS: FALSE POSITIVE RATE vs CONTEXT SIZE")
print("="*60)

print(f"\n{'Unique tokens':>14}", end="")
for n in capacity_levels:
    print(f"  {n:>6}", end="")
print()
print("-" * (14 + 8 * len(capacity_levels)))

# Theoretical
print(f"{'Theory(d=64)':>14}", end="")
for fp in theoretical_fp:
    print(f"  {fp:>5.1%}", end="")
print()

print("-" * (14 + 8 * len(capacity_levels)))

# Bloom filter heads
for head_key in BLOOM_HEADS:
    label = f"L{head_key[0]}H{head_key[1]} (BF)"
    print(f"{label:>14}", end="")
    for fp in capacity_results[head_key]["fp_rate"]:
        print(f"  {fp:>5.1%}", end="")
    print()

print("-" * (14 + 8 * len(capacity_levels)))

# Control heads
for head_key in CONTROL_HEADS:
    label = f"L{head_key[0]}H{head_key[1]} (ctl)"
    print(f"{label:>14}", end="")
    for fp in capacity_results[head_key]["fp_rate"]:
        print(f"  {fp:>5.1%}", end="")
    print()

# Miss rates
print("\n" + "="*60)
print("MISS RATES (should stay ~0 for Bloom filter heads)")
print("="*60)

print(f"\n{'Unique tokens':>14}", end="")
for n in capacity_levels:
    print(f"  {n:>6}", end="")
print()
print("-" * (14 + 8 * len(capacity_levels)))

for head_key in BLOOM_HEADS:
    label = f"L{head_key[0]}H{head_key[1]} (BF)"
    print(f"{label:>14}", end="")
    for mr in capacity_results[head_key]["miss_rate"]:
        print(f"  {mr:>5.1%}", end="")
    print()

for head_key in CONTROL_HEADS:
    label = f"L{head_key[0]}H{head_key[1]} (ctl)"
    print(f"{label:>14}", end="")
    for mr in capacity_results[head_key]["miss_rate"]:
        print(f"  {mr:>5.1%}", end="")
    print()

# Mean hit attention
print("\n" + "="*60)
print("MEAN HIT ATTENTION (how strongly repeats are detected)")
print("="*60)

print(f"\n{'Unique tokens':>14}", end="")
for n in capacity_levels:
    print(f"  {n:>6}", end="")
print()
print("-" * (14 + 8 * len(capacity_levels)))

for head_key in BLOOM_HEADS:
    label = f"L{head_key[0]}H{head_key[1]} (BF)"
    print(f"{label:>14}", end="")
    for mh in capacity_results[head_key]["mean_hit"]:
        print(f"  {mh:>5.3f}", end="")
    print()

# ============================================================
# Correlation with theoretical Bloom filter curve
# ============================================================
print("\n" + "="*60)
print("CORRELATION WITH THEORETICAL BLOOM FILTER CURVE")
print("="*60)

for head_key in BLOOM_HEADS:
    observed = capacity_results[head_key]["fp_rate"]
    # Compute Pearson correlation with theoretical curve
    if len(observed) > 2 and np.std(observed) > 0:
        corr = np.corrcoef(observed, theoretical_fp)[0, 1]
        print(f"  L{head_key[0]}H{head_key[1]}: r = {corr:.4f} {'✅ STRONG' if corr > 0.7 else '🟡 MODERATE' if corr > 0.4 else '❌ WEAK'}")
    else:
        print(f"  L{head_key[0]}H{head_key[1]}: insufficient variance in FP rate")

for head_key in CONTROL_HEADS:
    observed = capacity_results[head_key]["fp_rate"]
    if len(observed) > 2 and np.std(observed) > 0:
        corr = np.corrcoef(observed, theoretical_fp)[0, 1]
        print(f"  L{head_key[0]}H{head_key[1]} (ctrl): r = {corr:.4f}")
    else:
        print(f"  L{head_key[0]}H{head_key[1]} (ctrl): insufficient variance")

# ============================================================
# Fit effective Bloom filter parameters
# ============================================================
print("\n" + "="*60)
print("EFFECTIVE BLOOM FILTER PARAMETERS")
print("="*60)

from scipy.optimize import curve_fit

def bloom_model(n, m, k):
    """Bloom filter FP rate as function of n, with m and k as parameters."""
    return (1 - np.exp(-k * n / m)) ** k

for head_key in BLOOM_HEADS:
    observed = np.array(capacity_results[head_key]["fp_rate"])
    n_values = np.array(capacity_levels, dtype=float)
    
    if np.std(observed) > 0.01:
        try:
            popt, pcov = curve_fit(bloom_model, n_values, observed, p0=[64, 1], bounds=([1, 0.1], [1000, 10]))
            fitted_m, fitted_k = popt
            print(f"  L{head_key[0]}H{head_key[1]}: effective m={fitted_m:.1f} bits, k={fitted_k:.2f} hash functions")
            print(f"    (GPT-2 d_head={d_head} → head uses {fitted_m/d_head:.1%} of its capacity for membership)")
            
            # Compute R² of the fit
            predicted = bloom_model(n_values, fitted_m, fitted_k)
            ss_res = np.sum((observed - predicted) ** 2)
            ss_tot = np.sum((observed - np.mean(observed)) ** 2)
            r_squared = 1 - ss_res / ss_tot
            print(f"    Fit quality: R² = {r_squared:.4f}")
        except Exception as e:
            print(f"  L{head_key[0]}H{head_key[1]}: curve fit failed — {e}")
    else:
        print(f"  L{head_key[0]}H{head_key[1]}: FP rate too flat to fit (may be perfect filter)")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3 VERDICT")
print("="*60)

# Check key predictions
bloom_fp_increases = []
bloom_miss_stays_low = []

for head_key in BLOOM_HEADS:
    fp_rates = capacity_results[head_key]["fp_rate"]
    miss_rates = capacity_results[head_key]["miss_rate"]
    
    # Does FP rate increase with capacity?
    if len(fp_rates) > 2:
        early_fp = np.mean(fp_rates[:3])
        late_fp = np.mean(fp_rates[-3:])
        bloom_fp_increases.append(late_fp > early_fp)
    
    # Does miss rate stay low?
    bloom_miss_stays_low.append(np.mean(miss_rates) < 0.15)

print(f"\nPrediction 1: FP rate increases with context size")
print(f"  Bloom heads showing increase: {sum(bloom_fp_increases)}/{len(bloom_fp_increases)}")

print(f"\nPrediction 2: Miss rate stays near zero regardless of load")
print(f"  Bloom heads maintaining low miss rate: {sum(bloom_miss_stays_low)}/{len(bloom_miss_stays_low)}")

if sum(bloom_fp_increases) >= 2 and sum(bloom_miss_stays_low) >= 3:
    print("\n🟢 CAPACITY BEHAVIOR MATCHES BLOOM FILTER THEORY")
    print("   FP rate increases with load, miss rate stays low — exactly as predicted.")
elif sum(bloom_miss_stays_low) >= 3:
    print("\n🟡 PARTIAL MATCH — miss rate is Bloom-like, FP scaling is ambiguous")
    print("   Heads maintain zero miss rate but FP doesn't scale cleanly with capacity.")
else:
    print("\n🔴 WEAK MATCH — behavior doesn't follow Bloom filter capacity predictions")

# Save full results
save_data = {
    "experiment": "capacity_analysis",
    "model": "gpt2-small",
    "d_head": d_head,
    "capacity_levels": capacity_levels,
    "theoretical_fp": [round(x, 4) for x in theoretical_fp],
    "bloom_heads": {},
    "control_heads": {},
}

for head_key in BLOOM_HEADS:
    save_data["bloom_heads"][f"L{head_key[0]}H{head_key[1]}"] = capacity_results[head_key]
for head_key in CONTROL_HEADS:
    save_data["control_heads"][f"L{head_key[0]}H{head_key[1]}"] = capacity_results[head_key]

results_path = "/Users/peter/clawd/projects/bloom-filter-heads/results/experiment3_capacity.json"
with open(results_path, "w") as f:
    json.dump(save_data, f, indent=2)
print(f"\nResults saved to {results_path}")
