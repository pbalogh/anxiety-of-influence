"""
Experiment 4: Hash Function Analysis — Do multiple Bloom filter heads act as
multiple hash functions for a single distributed Bloom filter?

Real Bloom filters use k independent hash functions. Using k > 1 reduces false
positive rate because an item must hash to the same position under ALL functions.

If attention heads are Bloom filter hash functions:
1. Multiple Bloom heads should make INDEPENDENT membership decisions
   (low correlation between their false positive patterns)
2. Combining their decisions ("AND" logic) should reduce the false positive rate
   more than any single head alone
3. The combined FP rate should approximate: p_combined ≈ p1 * p2 * ... * pk
   (the product rule for independent Bloom filter hash functions)

We test this by:
1. Measuring per-head false positive patterns on the SAME inputs
2. Computing independence (correlation between false positive events across heads)
3. Testing whether combining heads reduces FP rate as predicted
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from collections import defaultdict
import json
from itertools import combinations

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
print(f"Model: {n_layers} layers, {n_heads} heads, d_head={d_head}")

BLOOM_HEADS = [(0, 1), (0, 5), (1, 11), (3, 0)]
# Non-Bloom heads for comparison
OTHER_HEADS = [(5, 5), (7, 10), (6, 9), (2, 2)]

ALL_TEST_HEADS = BLOOM_HEADS + OTHER_HEADS


def build_test_sequences(model, n_unique=50, n_probes=20, n_trials=30):
    """
    Build sequences with a known prefix and probe tokens that are NOT in the prefix.
    For each probe token, record which Bloom filter heads produce a "false positive"
    (high attention to the prefix).
    """
    sequences = []
    vocab_size = model.cfg.d_vocab
    
    for trial in range(n_trials):
        all_tokens = torch.randperm(vocab_size - 1000) + 1000
        prefix_tokens = all_tokens[:n_unique]
        probe_tokens = all_tokens[n_unique:n_unique + n_probes]
        
        # Also include some TRUE positives (tokens that ARE in the prefix)
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
            "repeat_start": 1 + n_unique + n_probes,
            "repeat_end": 1 + n_unique + n_probes + len(repeat_tokens),
            "n_probes": n_probes,
            "repeat_source_positions": (1 + repeat_indices).tolist(),
        })
    
    return sequences


def collect_per_token_decisions(model, sequences, heads, fp_threshold=0.1):
    """
    For each probe token in each sequence, record whether each head produced
    a "false positive" (attended to prefix despite probe not being in it).
    
    Returns a dict: {head: [bool, bool, ...]} where each bool is one probe token decision.
    Also returns per-token vectors for correlation analysis.
    """
    # Per-head: list of (trial, probe_idx, fp_attention, is_fp) tuples
    head_decisions = {h: [] for h in heads}
    # Aligned decision vectors for correlation
    decision_matrix = {h: [] for h in heads}
    # Continuous attention values for finer analysis
    attention_matrix = {h: [] for h in heads}
    # True positive tracking
    hit_matrix = {h: [] for h in heads}
    
    for seq_info in sequences:
        tokens = seq_info["tokens"]
        _, cache = model.run_with_cache(tokens)
        
        for layer, head in heads:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head].cpu()
            
            # False positive decisions for probe tokens
            for probe_pos in range(seq_info["probe_start"], min(seq_info["probe_end"], pattern.shape[0])):
                prefix_attn = pattern[probe_pos, 1:1 + seq_info["prefix_len"]].sum().item()
                is_fp = prefix_attn > fp_threshold
                decision_matrix[(layer, head)].append(int(is_fp))
                attention_matrix[(layer, head)].append(prefix_attn)
            
            # True positive decisions for repeat tokens
            repeat_positions = list(range(seq_info["repeat_start"], min(seq_info["repeat_end"], pattern.shape[0])))
            source_positions = seq_info["repeat_source_positions"]
            
            for rpt_pos, src_pos in zip(repeat_positions, source_positions):
                if rpt_pos < pattern.shape[0] and src_pos < pattern.shape[1]:
                    hit_attn = pattern[rpt_pos, src_pos].item()
                    hit_matrix[(layer, head)].append(hit_attn > 0.01)
    
    return decision_matrix, attention_matrix, hit_matrix


print("\n" + "="*70)
print("EXPERIMENT 4: MULTI-HEAD HASH FUNCTION ANALYSIS")
print("="*70)

# Use moderate context to get meaningful FP rates
print("\nBuilding test sequences (50 unique tokens, 20 probes × 30 trials)...")
sequences = build_test_sequences(model, n_unique=50, n_probes=20, n_trials=30)

print("Collecting per-token decisions for all heads...")
decisions, attentions, hits = collect_per_token_decisions(model, sequences, ALL_TEST_HEADS)

# ============================================================
# Analysis 1: Per-head false positive rates
# ============================================================
print("\n" + "="*70)
print("1. PER-HEAD FALSE POSITIVE RATES")
print("="*70)

fp_rates = {}
for head in ALL_TEST_HEADS:
    d = decisions[head]
    fp_rate = np.mean(d) if d else 0
    fp_rates[head] = fp_rate
    hit_rate = np.mean(hits[head]) if hits[head] else 0
    tag = "BLOOM" if head in BLOOM_HEADS else "other"
    print(f"  L{head[0]}H{head[1]} ({tag:>5}): FP rate = {fp_rate:.1%}, Hit rate = {hit_rate:.1%}, n={len(d)}")

# ============================================================
# Analysis 2: Independence of false positive decisions
# ============================================================
print("\n" + "="*70)
print("2. INDEPENDENCE OF FALSE POSITIVE DECISIONS (Bloom heads)")
print("="*70)

print("\nPhi correlation between Bloom head FP decisions:")
print("(Low correlation = independent hash functions = good)")
print(f"\n{'':>12}", end="")
for h in BLOOM_HEADS:
    print(f"  L{h[0]}H{h[1]:>2}", end="")
print()

phi_matrix = np.zeros((len(BLOOM_HEADS), len(BLOOM_HEADS)))

for i, h1 in enumerate(BLOOM_HEADS):
    print(f"  L{h1[0]}H{h1[1]:>2}     ", end="")
    for j, h2 in enumerate(BLOOM_HEADS):
        d1 = np.array(decisions[h1])
        d2 = np.array(decisions[h2])
        min_len = min(len(d1), len(d2))
        d1, d2 = d1[:min_len], d2[:min_len]
        
        if i == j:
            phi = 1.0
        else:
            # Phi coefficient (Matthews correlation for binary variables)
            n11 = np.sum((d1 == 1) & (d2 == 1))
            n10 = np.sum((d1 == 1) & (d2 == 0))
            n01 = np.sum((d1 == 0) & (d2 == 1))
            n00 = np.sum((d1 == 0) & (d2 == 0))
            
            denom = np.sqrt((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01))
            phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else 0
        
        phi_matrix[i, j] = phi
        if j >= i:
            print(f"  {phi:>6.3f}", end="")
        else:
            print(f"  {'':>6}", end="")
    print()

# Average pairwise phi
bloom_pairs = list(combinations(range(len(BLOOM_HEADS)), 2))
avg_phi = np.mean([phi_matrix[i, j] for i, j in bloom_pairs])
print(f"\nAverage pairwise phi (Bloom heads): {avg_phi:.4f}")
if abs(avg_phi) < 0.2:
    print("✅ LOW correlation — heads make INDEPENDENT decisions (like separate hash functions)")
elif abs(avg_phi) < 0.5:
    print("🟡 MODERATE correlation — partially independent")
else:
    print("❌ HIGH correlation — heads are redundant (same hash function)")

# Compare with other heads
print("\nComparison: Phi between Bloom heads vs non-Bloom heads:")
for h_other in OTHER_HEADS:
    for h_bloom in BLOOM_HEADS:
        d1 = np.array(decisions[h_bloom])
        d2 = np.array(decisions[h_other])
        min_len = min(len(d1), len(d2))
        d1, d2 = d1[:min_len], d2[:min_len]
        
        n11 = np.sum((d1 == 1) & (d2 == 1))
        n10 = np.sum((d1 == 1) & (d2 == 0))
        n01 = np.sum((d1 == 0) & (d2 == 1))
        n00 = np.sum((d1 == 0) & (d2 == 0))
        denom = np.sqrt((n11 + n10) * (n11 + n01) * (n00 + n10) * (n00 + n01))
        phi = (n11 * n00 - n10 * n01) / denom if denom > 0 else 0
        
    # Just show one summary per other head
    avg_phi_other = np.mean([
        (lambda d1, d2: (np.sum((d1==1)&(d2==1))*np.sum((d1==0)&(d2==0)) - np.sum((d1==1)&(d2==0))*np.sum((d1==0)&(d2==1))) / 
         max(np.sqrt(np.sum(d1==1)*np.sum(d2==1)*np.sum(d1==0)*np.sum(d2==0)), 1e-10))(
            np.array(decisions[h_bloom])[:min(len(decisions[h_bloom]), len(decisions[h_other]))],
            np.array(decisions[h_other])[:min(len(decisions[h_bloom]), len(decisions[h_other]))]
        )
        for h_bloom in BLOOM_HEADS
    ])
    print(f"  L{h_other[0]}H{h_other[1]} vs Bloom heads: avg phi = {avg_phi_other:.4f}")

# ============================================================
# Analysis 3: Combined Bloom filter — AND logic
# ============================================================
print("\n" + "="*70)
print("3. COMBINED BLOOM FILTER (AND logic across heads)")
print("="*70)

# For each probe token, check if ALL Bloom heads flagged it as FP
n_probes_total = min(len(decisions[h]) for h in BLOOM_HEADS)
bloom_decisions = np.array([decisions[h][:n_probes_total] for h in BLOOM_HEADS])

# Individual FP rates
individual_fps = bloom_decisions.mean(axis=1)

# Combined FP rates using AND of different subsets
print(f"\n{'Combination':>35}  {'Observed FP':>12}  {'Predicted FP':>12}  {'Match?':>8}")
print("-" * 75)

for k in range(1, len(BLOOM_HEADS) + 1):
    for combo in combinations(range(len(BLOOM_HEADS)), k):
        combo_decisions = bloom_decisions[list(combo)]
        # AND logic: all heads must agree it's a FP
        combined_fp = np.all(combo_decisions, axis=0).mean()
        # Predicted by independence: product of individual rates
        predicted_fp = np.prod(individual_fps[list(combo)])
        
        head_names = " ∧ ".join([f"L{BLOOM_HEADS[i][0]}H{BLOOM_HEADS[i][1]}" for i in combo])
        ratio = combined_fp / predicted_fp if predicted_fp > 0 else float('inf')
        match = "✅" if 0.3 < ratio < 3.0 else "❌"
        
        print(f"  {head_names:>33}  {combined_fp:>10.4f}  {predicted_fp:>10.4f}  {match:>6} ({ratio:.2f}x)")

# The big combined number
all_combined = np.all(bloom_decisions, axis=0).mean()
all_predicted = np.prod(individual_fps)
print(f"\n  ALL 4 Bloom heads combined:")
print(f"    Observed FP rate:  {all_combined:.4f}")
print(f"    Predicted (indep): {all_predicted:.4f}")
print(f"    Ratio:             {all_combined / all_predicted if all_predicted > 0 else float('inf'):.2f}x")

# ============================================================
# Analysis 4: Attention pattern correlation (continuous values)
# ============================================================
print("\n" + "="*70)
print("4. ATTENTION PATTERN CORRELATION (continuous)")
print("="*70)

print("\nPearson r between continuous attention values of Bloom heads:")
print(f"{'':>12}", end="")
for h in BLOOM_HEADS:
    print(f"  L{h[0]}H{h[1]:>2}", end="")
print()

for i, h1 in enumerate(BLOOM_HEADS):
    print(f"  L{h1[0]}H{h1[1]:>2}     ", end="")
    for j, h2 in enumerate(BLOOM_HEADS):
        a1 = np.array(attentions[h1])
        a2 = np.array(attentions[h2])
        min_len = min(len(a1), len(a2))
        a1, a2 = a1[:min_len], a2[:min_len]
        
        if i == j:
            r = 1.0
        else:
            r = np.corrcoef(a1, a2)[0, 1] if len(a1) > 2 else 0
        
        if j >= i:
            print(f"  {r:>6.3f}", end="")
        else:
            print(f"  {'':>6}", end="")
    print()

# ============================================================
# Analysis 5: What do false positives have in common?
# ============================================================
print("\n" + "="*70)
print("5. FALSE POSITIVE PATTERN ANALYSIS")
print("="*70)

# For each pair of Bloom heads, check: when both false-positive on the same token,
# is there something special about that token?
n_both_fp = 0
n_only_one_fp = 0
n_neither_fp = 0

for i in range(n_probes_total):
    fp_count = sum(bloom_decisions[j, i] for j in range(len(BLOOM_HEADS)))
    if fp_count == len(BLOOM_HEADS):
        n_both_fp += 1
    elif fp_count > 0:
        n_only_one_fp += 1
    else:
        n_neither_fp += 1

total = n_both_fp + n_only_one_fp + n_neither_fp
print(f"\n  All {len(BLOOM_HEADS)} heads agree FP:  {n_both_fp:>5} ({n_both_fp/total:.1%})")
print(f"  Mixed (some FP, some not): {n_only_one_fp:>5} ({n_only_one_fp/total:.1%})")
print(f"  No head flags FP:          {n_neither_fp:>5} ({n_neither_fp/total:.1%})")

# Distribution of how many heads fire per probe
print("\n  Distribution of FP head count per probe token:")
for k in range(len(BLOOM_HEADS) + 1):
    count = sum(1 for i in range(n_probes_total) if sum(bloom_decisions[j, i] for j in range(len(BLOOM_HEADS))) == k)
    bar = "█" * (count // 3)
    print(f"    {k} heads fire: {count:>4} ({count/n_probes_total:.1%}) {bar}")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*70)
print("EXPERIMENT 4 VERDICT")
print("="*70)

independence_pass = abs(avg_phi) < 0.3
combination_works = all_combined < max(individual_fps) * 0.8  # combining should reduce FP
diverse_patterns = n_only_one_fp > n_both_fp  # more diversity than agreement

print(f"\n  Independence test (phi < 0.3):      {'✅ PASS' if independence_pass else '❌ FAIL'} (phi={avg_phi:.3f})")
print(f"  Combination reduces FP:             {'✅ PASS' if combination_works else '❌ FAIL'} ({all_combined:.3f} < {max(individual_fps):.3f})")
print(f"  Diverse FP patterns:                {'✅ PASS' if diverse_patterns else '❌ FAIL'} (mixed={n_only_one_fp} > all={n_both_fp})")

if independence_pass and combination_works:
    print("\n  🟢 BLOOM HEADS ACT AS INDEPENDENT HASH FUNCTIONS")
    print("     Multiple heads checking membership with different projections = ")
    print("     a distributed Bloom filter with reduced false positive rate.")
    print(f"     Combined FP ({all_combined:.3f}) vs best single head ({min(individual_fps):.3f})")
elif combination_works:
    print("\n  🟡 PARTIALLY INDEPENDENT — combining helps, but not fully independent")
    print("     Heads share some information but provide complementary coverage.")
else:
    print("\n  🔴 HEADS ARE NOT INDEPENDENT HASH FUNCTIONS")
    print("     They may use correlated features for membership testing.")

# Save results
results = {
    "experiment": "hash_function_analysis",
    "model": "gpt2-small",
    "bloom_heads": [f"L{h[0]}H{h[1]}" for h in BLOOM_HEADS],
    "individual_fp_rates": {f"L{h[0]}H{h[1]}": round(fp_rates[h], 4) for h in BLOOM_HEADS},
    "phi_matrix": phi_matrix.tolist(),
    "avg_phi": round(avg_phi, 4),
    "combined_fp_all": round(float(all_combined), 4),
    "predicted_fp_all": round(float(all_predicted), 6),
    "independence_pass": bool(independence_pass),
    "combination_works": bool(combination_works),
    "diverse_patterns": bool(diverse_patterns),
    "fp_distribution": {
        "all_agree_fp": n_both_fp,
        "mixed": n_only_one_fp,
        "none_fp": n_neither_fp,
    }
}

results_path = "/Users/pabalogh/clawd/projects/bloom-filter-heads/results/experiment4_hash_functions.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")
