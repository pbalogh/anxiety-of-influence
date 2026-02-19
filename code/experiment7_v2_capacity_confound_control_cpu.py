"""
Experiment 7 v2: Capacity Confound Control (Fixed)

The v1 used cyclic repeats to fill sequences, which created a measurement artifact:
every prefix position contained a repeated token, so "attending to prefix" was always
high even for novel probes. 

Fix: Use DISTINCT prefix tokens + novel probe tokens + PADDING to hold total sequence
length constant. This isolates the effect of n_unique on FP rate without length confounds.

Structure: [BOS] + [n_unique DISTINCT tokens] + [5 probe tokens] + [PAD to total_length]
"""

import torch
import numpy as np
import json
import os
import math
from collections import defaultdict
from scipy.optimize import curve_fit
from transformer_lens import HookedTransformer

device = "cpu"  # FORCED CPU
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
vocab_size = model.cfg.d_vocab
print(f"Model loaded: {n_layers} layers, {n_heads} heads, d_head={d_head}")

# Use EOS token as padding
PAD_TOKEN = model.tokenizer.eos_token_id
print(f"PAD token: {PAD_TOKEN}")

BLOOM_HEADS = [(0, 1), (0, 5), (1, 11), (3, 0)]
CONTROL_HEADS = [(5, 5), (7, 10), (6, 9)]

N_TRIALS = 30
N_PROBES = 5
FP_THRESHOLD = 0.1


def build_sequence_v2(n_unique, total_length, n_probes=5):
    """
    Build: [BOS] + [n_unique DISTINCT tokens] + [n_probes NOVEL tokens] + [PAD...]
    
    Total token count = 1 + total_length (BOS + total_length positions).
    The prefix occupies positions 1..n_unique.
    Probes occupy positions n_unique+1..n_unique+n_probes.
    Padding fills the rest to total_length.
    
    FP measurement: attention from probe tokens to PREFIX positions only (1..n_unique).
    """
    # Pick distinct tokens (avoid special tokens 0-255 and PAD)
    candidates = [t for t in range(1000, vocab_size) if t != PAD_TOKEN]
    selected = np.random.choice(candidates, size=n_unique + n_probes, replace=False)
    
    prefix_tokens = selected[:n_unique]
    probe_tokens = selected[n_unique:n_unique + n_probes]
    
    # Padding to fill total_length positions after BOS
    n_pad = total_length - n_unique - n_probes
    if n_pad < 0:
        # If n_unique + n_probes > total_length, truncate padding (no padding needed)
        n_pad = 0
        # Adjust: we need at least n_probes after prefix
        # Reduce prefix if needed
        actual_unique = total_length - n_probes
        if actual_unique < n_unique:
            prefix_tokens = prefix_tokens[:actual_unique]
            n_unique = actual_unique
            n_pad = 0
    
    seq_list = [model.tokenizer.bos_token_id] + list(prefix_tokens) + list(probe_tokens) + [PAD_TOKEN] * n_pad
    seq = torch.tensor(seq_list, dtype=torch.long).unsqueeze(0).to(device)
    
    return {
        "tokens": seq,
        "prefix_start": 1,
        "prefix_end": 1 + n_unique,  # exclusive
        "probe_start": 1 + n_unique,
        "probe_end": 1 + n_unique + n_probes,
        "n_unique": n_unique,
        "total_length": total_length,
    }


def measure_fp_rate(sequences, heads):
    """
    FP rate: fraction of probe tokens attending > FP_THRESHOLD to PREFIX positions.
    Only prefix positions (containing the distinct tokens) count — NOT padding.
    """
    per_probe_binary = defaultdict(list)
    per_probe_attn = defaultdict(list)

    for seq_info in sequences:
        tokens = seq_info["tokens"]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        for layer, head in heads:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]

            for probe_pos in range(seq_info["probe_start"], seq_info["probe_end"]):
                if probe_pos >= pattern.shape[0]:
                    continue
                # Attention to PREFIX positions only (distinct tokens)
                prefix_attn = pattern[probe_pos, seq_info["prefix_start"]:seq_info["prefix_end"]].sum().item()
                per_probe_binary[(layer, head)].append(1 if prefix_attn > FP_THRESHOLD else 0)
                per_probe_attn[(layer, head)].append(prefix_attn)

    results = {}
    for key in heads:
        kt = tuple(key)
        binary = per_probe_binary[kt]
        attns = per_probe_attn[kt]
        ci = bootstrap_ci(binary)
        results[kt] = {
            "fp_rate": float(np.mean(binary)) if binary else 0.0,
            "mean_fp_attention": float(np.mean(attns)) if attns else 0.0,
            "ci_95": [round(ci[0], 4), round(ci[1], 4)],
            "n_probes_tested": len(binary),
        }
    return results


def bootstrap_ci(values, n_bootstrap=5000, ci=0.95):
    if len(values) < 2:
        return (0.0, 0.0)
    values = np.array(values)
    boot_means = np.array([np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)])
    boot_means.sort()
    lo = boot_means[int((1 - ci) / 2 * n_bootstrap)]
    hi = boot_means[int((1 + ci) / 2 * n_bootstrap)]
    return (float(lo), float(hi))


def bloom_model(n, m, k):
    return (1 - np.exp(-k * n / m)) ** k


all_heads = BLOOM_HEADS + CONTROL_HEADS

# ============================================================
# Control 1: FIXED LENGTH = 200, vary unique tokens
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 1: FIXED LENGTH = 200, VARY UNIQUE TOKENS")
print("=" * 70)
print("Sequence: [BOS] + [n_unique distinct] + [5 probes] + [PAD to 200]")
print("FP = attention from probes to prefix positions only\n")

FIXED_LENGTH = 200
unique_counts = [5, 20, 50, 100, 180]

fixed_length_results = {}

for n_unique in unique_counts:
    print(f"  n_unique = {n_unique:>3d} (prefix={n_unique}, probes={N_PROBES}, "
          f"pad={FIXED_LENGTH - n_unique - N_PROBES}, {N_TRIALS} trials)...")

    sequences = [build_sequence_v2(n_unique, FIXED_LENGTH, N_PROBES) for _ in range(N_TRIALS)]
    fp_results = measure_fp_rate(sequences, all_heads)

    condition_key = f"unique_{n_unique}"
    fixed_length_results[condition_key] = {"n_unique": n_unique, "total_length": FIXED_LENGTH}

    for head in all_heads:
        head_key = f"L{head[0]}H{head[1]}"
        r = fp_results[tuple(head)]
        fixed_length_results[condition_key][head_key] = {
            "fp_rate": r["fp_rate"],
            "mean_fp_attention": r["mean_fp_attention"],
            "ci_95": r["ci_95"],
            "n_probes": r["n_probes_tested"],
        }
        tag = "BF" if head in BLOOM_HEADS else "ctrl"
        print(f"    {head_key} ({tag}): FP={r['fp_rate']:.4f} [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]  "
              f"mean_attn={r['mean_fp_attention']:.4f}")

# ============================================================
# Control 2: FIXED UNIQUE = 50, vary sequence length
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 2: FIXED UNIQUE = 50, VARY SEQUENCE LENGTH")
print("=" * 70)
print("Sequence: [BOS] + [50 distinct] + [5 probes] + [PAD to total_length]")
print("FP should be ~constant (same unique tokens in filter)\n")

FIXED_UNIQUE = 50
seq_lengths = [50 + N_PROBES, 100, 150, 200]  # min length = unique + probes

fixed_unique_results = {}

for total_len in seq_lengths:
    n_pad = total_len - FIXED_UNIQUE - N_PROBES
    print(f"  total_length = {total_len} (prefix=50, probes={N_PROBES}, pad={n_pad}, {N_TRIALS} trials)...")

    sequences = [build_sequence_v2(FIXED_UNIQUE, total_len, N_PROBES) for _ in range(N_TRIALS)]
    fp_results = measure_fp_rate(sequences, all_heads)

    condition_key = f"length_{total_len}"
    fixed_unique_results[condition_key] = {"n_unique": FIXED_UNIQUE, "total_length": total_len}

    for head in all_heads:
        head_key = f"L{head[0]}H{head[1]}"
        r = fp_results[tuple(head)]
        fixed_unique_results[condition_key][head_key] = {
            "fp_rate": r["fp_rate"],
            "mean_fp_attention": r["mean_fp_attention"],
            "ci_95": r["ci_95"],
            "n_probes": r["n_probes_tested"],
        }
        tag = "BF" if head in BLOOM_HEADS else "ctrl"
        print(f"    {head_key} ({tag}): FP={r['fp_rate']:.4f} [{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]  "
              f"mean_attn={r['mean_fp_attention']:.4f}")

# ============================================================
# Section 3: Bloom Filter Curve Fitting (fixed-length data)
# ============================================================

print("\n" + "=" * 70)
print("BLOOM FILTER CURVE FITTING TO FIXED-LENGTH DATA")
print("=" * 70)
print("Formula: p = (1 - e^(-kn/m))^k\n")

fixed_length_fit = {}

for head in BLOOM_HEADS:
    hk = f"L{head[0]}H{head[1]}"
    n_vals = np.array([float(uc) for uc in unique_counts])
    fp_vals = np.array([fixed_length_results[f"unique_{uc}"][hk]["fp_rate"] for uc in unique_counts])

    print(f"  {hk}: observed FP = {[round(float(x),4) for x in fp_vals]}")

    if np.std(fp_vals) > 0.005:
        try:
            popt, _ = curve_fit(bloom_model, n_vals, fp_vals, p0=[64, 1],
                                bounds=([1, 0.1], [1000, 10]), maxfev=10000)
            fitted_m, fitted_k = popt
            predicted = bloom_model(n_vals, fitted_m, fitted_k)
            ss_res = np.sum((fp_vals - predicted) ** 2)
            ss_tot = np.sum((fp_vals - np.mean(fp_vals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fixed_length_fit[hk] = {
                "fitted_m": round(float(fitted_m), 2),
                "fitted_k": round(float(fitted_k), 2),
                "r_squared": round(float(r2), 4),
                "observed_fp": [round(float(x), 4) for x in fp_vals],
                "predicted_fp": [round(float(x), 4) for x in predicted],
            }
            print(f"    → m={fitted_m:.1f}, k={fitted_k:.2f}, R²={r2:.4f}")
        except Exception as e:
            print(f"    → fit failed: {e}")
            fixed_length_fit[hk] = {"error": str(e)}
    else:
        print(f"    → too flat (std={np.std(fp_vals):.4f})")
        fixed_length_fit[hk] = {"error": "variance too low", "std": round(float(np.std(fp_vals)), 4)}

# Also fit variable-length (exp3-style) for comparison
print("\n  Variable-length comparison (seq_length = n_unique)...")
variable_length_fit = {}
var_fp_data = defaultdict(list)

for n_unique in [5, 20, 50, 100, 200]:
    print(f"    n_unique = seq_length = {n_unique}...", end="", flush=True)
    sequences = []
    for _ in range(N_TRIALS):
        candidates = [t for t in range(1000, vocab_size) if t != PAD_TOKEN]
        selected = np.random.choice(candidates, size=n_unique + N_PROBES, replace=False)
        prefix = selected[:n_unique]
        probes = selected[n_unique:]
        seq = torch.tensor(
            [model.tokenizer.bos_token_id] + list(prefix) + list(probes),
            dtype=torch.long
        ).unsqueeze(0).to(device)
        sequences.append({
            "tokens": seq,
            "prefix_start": 1,
            "prefix_end": 1 + n_unique,
            "probe_start": 1 + n_unique,
            "probe_end": 1 + n_unique + N_PROBES,
            "n_unique": n_unique,
            "total_length": n_unique + N_PROBES,
        })
    fp_results = measure_fp_rate(sequences, BLOOM_HEADS)
    for head in BLOOM_HEADS:
        hk = f"L{head[0]}H{head[1]}"
        var_fp_data[hk].append((n_unique, fp_results[tuple(head)]["fp_rate"]))
    print(" done")

for head in BLOOM_HEADS:
    hk = f"L{head[0]}H{head[1]}"
    data = var_fp_data[hk]
    n_vals = np.array([d[0] for d in data], dtype=float)
    fp_vals = np.array([d[1] for d in data])
    if np.std(fp_vals) > 0.005:
        try:
            popt, _ = curve_fit(bloom_model, n_vals, fp_vals, p0=[64, 1],
                                bounds=([1, 0.1], [1000, 10]), maxfev=10000)
            predicted = bloom_model(n_vals, *popt)
            ss_res = np.sum((fp_vals - predicted) ** 2)
            ss_tot = np.sum((fp_vals - np.mean(fp_vals)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            variable_length_fit[hk] = {
                "fitted_m": round(float(popt[0]), 2),
                "fitted_k": round(float(popt[1]), 2),
                "r_squared": round(float(r2), 4),
            }
            print(f"  {hk} (var-len): m={popt[0]:.1f}, k={popt[1]:.2f}, R²={r2:.4f}")
        except Exception as e:
            variable_length_fit[hk] = {"error": str(e)}
    else:
        variable_length_fit[hk] = {"error": "variance too low"}

# ============================================================
# R² Comparison
# ============================================================

print("\n" + "=" * 70)
print("R² COMPARISON: FIXED-LENGTH vs VARIABLE-LENGTH")
print("=" * 70)

print(f"  {'Head':>10s}  {'Var-Len R²':>12s}  {'Fixed-Len R²':>14s}")
print(f"  {'-'*10}  {'-'*12}  {'-'*14}")
for head in BLOOM_HEADS:
    hk = f"L{head[0]}H{head[1]}"
    vr = variable_length_fit.get(hk, {}).get("r_squared", "N/A")
    fr = fixed_length_fit.get(hk, {}).get("r_squared", "N/A")
    print(f"  {hk:>10s}  {str(vr):>12s}  {str(fr):>14s}")

# ============================================================
# Stability test
# ============================================================

print("\n" + "=" * 70)
print("FP STABILITY (fixed unique=50, varying length)")
print("=" * 70)

for head in BLOOM_HEADS:
    hk = f"L{head[0]}H{head[1]}"
    fp_rates = [fixed_unique_results[f"length_{sl}"][hk]["fp_rate"] for sl in seq_lengths]
    mean_fp = np.mean(fp_rates)
    std_fp = np.std(fp_rates)
    cv = std_fp / mean_fp if mean_fp > 0 else float('inf')
    print(f"  {hk}: ", end="")
    for sl, fp in zip(seq_lengths, fp_rates):
        print(f"len={sl}→{fp:.4f}  ", end="")
    print(f"| mean={mean_fp:.4f}, std={std_fp:.4f}, CV={cv:.4f}")

# ============================================================
# Theoretical comparison
# ============================================================

print("\n" + "=" * 70)
print("THEORETICAL BLOOM FILTER COMPARISON (m=64, k=1)")
print("=" * 70)

print(f"  {'n_unique':>8s}  {'Theory':>8s}", end="")
for head in BLOOM_HEADS:
    print(f"  {f'L{head[0]}H{head[1]}':>8s}", end="")
print()

for n_unique in unique_counts:
    theory = (1 - math.exp(-1.0 * n_unique / 64)) ** 1
    print(f"  {n_unique:>8d}  {theory:>8.4f}", end="")
    for head in BLOOM_HEADS:
        hk = f"L{head[0]}H{head[1]}"
        fp = fixed_length_results[f"unique_{n_unique}"][hk]["fp_rate"]
        print(f"  {fp:>8.4f}", end="")
    print()

# ============================================================
# Save results
# ============================================================

results = {
    "experiment": "capacity_confound_control_v2",
    "fix_description": "Use distinct prefix tokens + probes + PAD (not cyclic repeats)",
    "model": "gpt2-small",
    "device": device,
    "d_head": d_head,
    "pad_token": PAD_TOKEN,
    "bloom_heads": [list(h) for h in BLOOM_HEADS],
    "control_heads": [list(h) for h in CONTROL_HEADS],
    "parameters": {"n_trials": N_TRIALS, "n_probes": N_PROBES, "fp_threshold": FP_THRESHOLD},
    "control_1_fixed_length": {
        "total_length": FIXED_LENGTH,
        "unique_counts": unique_counts,
        "results": fixed_length_results,
    },
    "control_2_fixed_unique": {
        "n_unique": FIXED_UNIQUE,
        "seq_lengths": seq_lengths,
        "results": fixed_unique_results,
    },
    "curve_fitting": {
        "fixed_length_fit": fixed_length_fit,
        "variable_length_fit": variable_length_fit,
    },
}

results_dir = "/Users/peter/clawd/projects/bloom-filter-heads/results"
os.makedirs(results_dir, exist_ok=True)

json_path = os.path.join(results_dir, "experiment7_v2_results.json")
with open(json_path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to {json_path}")
print("Done!")
