"""
Experiment 7: Capacity Confound Control

The current capacity experiment (experiment 3) varies both the number of unique
tokens AND the sequence length simultaneously. This confounds "filter filling"
with positional/length effects.

We disentangle with two controls:
  1. FIXED-LENGTH: Hold sequence length at 200, vary proportion of unique tokens.
     If Bloom filter theory holds, FP rate depends on unique tokens, not length.
  2. FIXED-UNIQUE: Hold unique tokens at 50, vary sequence length (repeats).
     FP rate should be roughly CONSTANT (same n in the filter).

We also refit the Bloom filter formula to the fixed-length data and compare R²
with the original variable-length design.
"""

import torch
import numpy as np
import json
import os
import math
from collections import defaultdict
from scipy.optimize import curve_fit
from transformer_lens import HookedTransformer

# ============================================================
# Setup
# ============================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
d_head = model.cfg.d_head
vocab_size = model.cfg.d_vocab
print(f"Model loaded: {n_layers} layers, {n_heads} heads, d_head={d_head}")

BLOOM_HEADS = [(0, 1), (0, 5), (1, 11), (3, 0)]
CONTROL_HEADS = [(5, 5), (7, 10), (6, 9)]  # Induction heads as controls

N_TRIALS = 30
N_PROBES = 5
FP_THRESHOLD = 0.1  # attention > 10% to prefix = false positive

# ============================================================
# Helper: Build sequences and measure FP rate
# ============================================================

def build_fixed_length_sequence(model, n_unique, total_length, n_probes=5):
    """
    Build a sequence of exactly total_length prefix tokens using n_unique unique tokens
    (repeated cyclically to fill), then n_probes probe tokens not in the prefix vocabulary.

    Returns dict with token tensor and metadata.
    """
    # Pick n_unique random tokens (avoid special tokens 0-999)
    all_tokens = torch.randperm(vocab_size - 1000) + 1000
    unique_tokens = all_tokens[:n_unique]

    # Fill prefix by cycling through unique tokens
    if n_unique >= total_length:
        prefix = unique_tokens[:total_length]
    else:
        repeats = total_length // n_unique
        remainder = total_length % n_unique
        prefix = torch.cat([unique_tokens] * repeats + [unique_tokens[:remainder]])

    assert len(prefix) == total_length, f"Expected {total_length}, got {len(prefix)}"

    # Probe tokens: guaranteed not in prefix vocabulary
    prefix_set = set(unique_tokens.tolist())
    probe_candidates = [t.item() for t in all_tokens[n_unique:] if t.item() not in prefix_set]
    probe_tokens = torch.tensor(probe_candidates[:n_probes])

    # Build: [BOS] + prefix + probes
    seq = torch.cat([
        torch.tensor([model.tokenizer.bos_token_id]),
        prefix,
        probe_tokens,
    ]).unsqueeze(0).to(device)

    return {
        "tokens": seq,
        "prefix_len": total_length,
        "probe_start": 1 + total_length,
        "probe_end": 1 + total_length + n_probes,
        "n_unique": n_unique,
    }


def measure_fp_rate_with_probes(model, sequences, heads):
    """
    Measure false positive rate: fraction of probe tokens that attend > FP_THRESHOLD
    to prefix positions, for each head.

    Returns per-head dict with fp_rate, mean_fp_attention, ci_95, and raw per-probe values.
    Single inference pass per sequence (no redundant forward passes).
    """
    per_probe_binary = defaultdict(list)  # head -> list of 0/1 per probe
    per_probe_attn = defaultdict(list)    # head -> list of continuous attention values

    for seq_info in sequences:
        tokens = seq_info["tokens"]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)

        for layer, head in heads:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # [dest, src]

            for probe_pos in range(seq_info["probe_start"], seq_info["probe_end"]):
                if probe_pos >= pattern.shape[0]:
                    continue
                # Total attention from probe to prefix positions
                prefix_attn = pattern[probe_pos, 1:1 + seq_info["prefix_len"]].sum().item()
                per_probe_binary[(layer, head)].append(1 if prefix_attn > FP_THRESHOLD else 0)
                per_probe_attn[(layer, head)].append(prefix_attn)

    results = {}
    for key in heads:
        key_tuple = tuple(key)
        binary = per_probe_binary[key_tuple]
        attns = per_probe_attn[key_tuple]
        ci_lo, ci_hi = bootstrap_ci(binary)
        results[key_tuple] = {
            "fp_rate": float(np.mean(binary)) if binary else 0.0,
            "mean_fp_attention": float(np.mean(attns)) if attns else 0.0,
            "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            "n_probes_tested": len(binary),
        }

    return results


def bootstrap_ci(values, n_bootstrap=5000, ci=0.95):
    """Compute bootstrap confidence interval."""
    if len(values) < 2:
        return (0.0, 0.0)
    values = np.array(values)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=len(values), replace=True)
        boot_means.append(np.mean(sample))
    boot_means = sorted(boot_means)
    lo = boot_means[int((1 - ci) / 2 * n_bootstrap)]
    hi = boot_means[int((1 + ci) / 2 * n_bootstrap)]
    return (float(lo), float(hi))


# ============================================================
# Control 1: FIXED LENGTH (200 tokens), vary unique count
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 1: FIXED LENGTH = 200, VARY UNIQUE TOKENS")
print("=" * 70)
print("Purpose: If FP rate depends on unique tokens (not length), this is the")
print("true Bloom filter capacity curve — unconfounded by sequence length.\n")

FIXED_LENGTH = 200
unique_counts = [5, 20, 50, 100, 200]
all_heads = BLOOM_HEADS + CONTROL_HEADS

fixed_length_results = {}

for n_unique in unique_counts:
    reps = FIXED_LENGTH // n_unique if n_unique < FIXED_LENGTH else 1
    print(f"  n_unique = {n_unique:>3d} ({reps}x repeats to fill {FIXED_LENGTH} positions, "
          f"{N_TRIALS} trials)...")

    sequences = [
        build_fixed_length_sequence(model, n_unique, FIXED_LENGTH, N_PROBES)
        for _ in range(N_TRIALS)
    ]

    fp_results = measure_fp_rate_with_probes(model, sequences, all_heads)

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

        is_bloom = "BF" if head in BLOOM_HEADS else "ctrl"
        print(f"    {head_key} ({is_bloom}): FP rate = {r['fp_rate']:.4f} "
              f"[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]  "
              f"mean_attn = {r['mean_fp_attention']:.4f}")

# ============================================================
# Control 2: FIXED UNIQUE TOKENS (50), vary sequence length
# ============================================================

print("\n" + "=" * 70)
print("CONTROL 2: FIXED UNIQUE = 50, VARY SEQUENCE LENGTH")
print("=" * 70)
print("Purpose: Per Bloom filter theory, FP rate depends on unique tokens (n),")
print("not total length. FP rate should be roughly CONSTANT across lengths.\n")

FIXED_UNIQUE = 50
seq_lengths = [50, 100, 150, 200]

fixed_unique_results = {}

for total_len in seq_lengths:
    repeats_per_token = total_len / FIXED_UNIQUE
    print(f"  total_length = {total_len} ({repeats_per_token:.1f}x per token, {N_TRIALS} trials)...")

    sequences = [
        build_fixed_length_sequence(model, FIXED_UNIQUE, total_len, N_PROBES)
        for _ in range(N_TRIALS)
    ]

    fp_results = measure_fp_rate_with_probes(model, sequences, all_heads)

    condition_key = f"length_{total_len}"
    fixed_unique_results[condition_key] = {
        "n_unique": FIXED_UNIQUE,
        "total_length": total_len,
        "repeats_per_token": repeats_per_token,
    }

    for head in all_heads:
        head_key = f"L{head[0]}H{head[1]}"
        r = fp_results[tuple(head)]

        fixed_unique_results[condition_key][head_key] = {
            "fp_rate": r["fp_rate"],
            "mean_fp_attention": r["mean_fp_attention"],
            "ci_95": r["ci_95"],
            "n_probes": r["n_probes_tested"],
        }

        is_bloom = "BF" if head in BLOOM_HEADS else "ctrl"
        print(f"    {head_key} ({is_bloom}): FP rate = {r['fp_rate']:.4f} "
              f"[{r['ci_95'][0]:.4f}, {r['ci_95'][1]:.4f}]  "
              f"mean_attn = {r['mean_fp_attention']:.4f}")

# ============================================================
# Section 3: Bloom Filter Curve Fitting
# ============================================================

print("\n" + "=" * 70)
print("SECTION 3: BLOOM FILTER CURVE FITTING")
print("=" * 70)
print("Formula: p = (1 - e^(-kn/m))^k")
print("  n = unique tokens, m = effective capacity, k = hash functions\n")


def bloom_model(n, m, k):
    """Bloom filter FP rate as function of n (items), with m (bits) and k (hashes)."""
    return (1 - np.exp(-k * n / m)) ** k


def theoretical_bloom_fp(n, m=64, k=1):
    """Theoretical FP rate with d_head=64, k=1."""
    return (1 - math.exp(-k * n / m)) ** k


# --- Fit to fixed-length data (Control 1) ---
print("  A) Fitting Bloom filter model to FIXED-LENGTH data (Control 1)...")
fixed_length_fit = {}

for head in BLOOM_HEADS:
    head_key = f"L{head[0]}H{head[1]}"
    n_vals = np.array([float(uc) for uc in unique_counts])
    fp_vals = np.array([
        fixed_length_results[f"unique_{uc}"][head_key]["fp_rate"]
        for uc in unique_counts
    ])

    if np.std(fp_vals) > 0.005:
        try:
            popt, pcov = curve_fit(bloom_model, n_vals, fp_vals,
                                   p0=[64, 1], bounds=([1, 0.1], [1000, 10]),
                                   maxfev=10000)
            fitted_m, fitted_k = popt

            predicted = bloom_model(n_vals, fitted_m, fitted_k)
            ss_res = np.sum((fp_vals - predicted) ** 2)
            ss_tot = np.sum((fp_vals - np.mean(fp_vals)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            fixed_length_fit[head_key] = {
                "fitted_m": round(float(fitted_m), 2),
                "fitted_k": round(float(fitted_k), 2),
                "r_squared": round(float(r_squared), 4),
                "observed_fp": [round(float(x), 4) for x in fp_vals],
                "predicted_fp": [round(float(x), 4) for x in predicted],
            }

            print(f"    {head_key}: m={fitted_m:.1f}, k={fitted_k:.2f}, "
                  f"R²={r_squared:.4f}")
        except Exception as e:
            print(f"    {head_key}: curve fit failed — {e}")
            fixed_length_fit[head_key] = {"error": str(e)}
    else:
        print(f"    {head_key}: FP rate too flat to fit (std={np.std(fp_vals):.4f})")
        fixed_length_fit[head_key] = {
            "error": "variance too low",
            "std": round(float(np.std(fp_vals)), 4),
        }

# --- Fit to original variable-length data for comparison ---
# Replicate experiment 3's approach: sequence length = unique token count
print("\n  B) Fitting Bloom filter model to VARIABLE-LENGTH data (exp3 replication)...")
print("     (Here seq_length = n_unique, so length and capacity are confounded)")

variable_length_levels = [5, 10, 20, 50, 100, 150, 200]
variable_length_fit = {}

var_fp_data = defaultdict(list)  # head -> list of (n_unique, fp_rate)

for n_unique in variable_length_levels:
    print(f"    n_unique = seq_length = {n_unique}...", end="", flush=True)
    sequences = []
    for _ in range(N_TRIALS):
        all_tokens = torch.randperm(vocab_size - 1000) + 1000
        prefix_tokens = all_tokens[:n_unique]
        probe_tokens = all_tokens[n_unique:n_unique + N_PROBES]

        seq = torch.cat([
            torch.tensor([model.tokenizer.bos_token_id]),
            prefix_tokens,
            probe_tokens,
        ]).unsqueeze(0).to(device)

        sequences.append({
            "tokens": seq,
            "prefix_len": n_unique,
            "probe_start": 1 + n_unique,
            "probe_end": 1 + n_unique + N_PROBES,
        })

    fp_results = measure_fp_rate_with_probes(model, sequences, BLOOM_HEADS)
    for head in BLOOM_HEADS:
        head_key = f"L{head[0]}H{head[1]}"
        var_fp_data[head_key].append((n_unique, fp_results[tuple(head)]["fp_rate"]))

    print(" done")

for head in BLOOM_HEADS:
    head_key = f"L{head[0]}H{head[1]}"
    data = var_fp_data[head_key]
    n_vals = np.array([d[0] for d in data], dtype=float)
    fp_vals = np.array([d[1] for d in data])

    if np.std(fp_vals) > 0.005:
        try:
            popt, pcov = curve_fit(bloom_model, n_vals, fp_vals,
                                   p0=[64, 1], bounds=([1, 0.1], [1000, 10]),
                                   maxfev=10000)
            fitted_m, fitted_k = popt

            predicted = bloom_model(n_vals, fitted_m, fitted_k)
            ss_res = np.sum((fp_vals - predicted) ** 2)
            ss_tot = np.sum((fp_vals - np.mean(fp_vals)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            variable_length_fit[head_key] = {
                "fitted_m": round(float(fitted_m), 2),
                "fitted_k": round(float(fitted_k), 2),
                "r_squared": round(float(r_squared), 4),
                "observed_fp": [round(float(x), 4) for x in fp_vals],
                "predicted_fp": [round(float(x), 4) for x in predicted],
            }

            print(f"    {head_key}: m={fitted_m:.1f}, k={fitted_k:.2f}, "
                  f"R²={r_squared:.4f}")
        except Exception as e:
            print(f"    {head_key}: curve fit failed — {e}")
            variable_length_fit[head_key] = {"error": str(e)}
    else:
        print(f"    {head_key}: FP rate too flat (std={np.std(fp_vals):.4f})")
        variable_length_fit[head_key] = {"error": "variance too low"}

# ============================================================
# Section 4: R² Comparison Table
# ============================================================

print("\n" + "=" * 70)
print("SECTION 4: R² COMPARISON — FIXED-LENGTH vs VARIABLE-LENGTH")
print("=" * 70)
print("\nIf R² is SIMILAR or BETTER for fixed-length, the capacity scaling is")
print("genuine Bloom filter behavior, not a length artifact.\n")

print(f"  {'Head':>10s}  {'Var-Length R²':>14s}  {'Fixed-Length R²':>16s}  {'Δ':>8s}")
print(f"  {'-'*10}  {'-'*14}  {'-'*16}  {'-'*8}")

comparison_table = {}
for head in BLOOM_HEADS:
    hk = f"L{head[0]}H{head[1]}"
    vr = variable_length_fit.get(hk, {}).get("r_squared", None)
    fr = fixed_length_fit.get(hk, {}).get("r_squared", None)

    vr_str = f"{vr:.4f}" if vr is not None else "N/A"
    fr_str = f"{fr:.4f}" if fr is not None else "N/A"

    if vr is not None and fr is not None:
        delta = fr - vr
        delta_str = f"{delta:+.4f}"
    else:
        delta = None
        delta_str = "N/A"

    comparison_table[hk] = {
        "variable_length_r2": vr,
        "fixed_length_r2": fr,
        "delta": delta,
    }

    print(f"  {hk:>10s}  {vr_str:>14s}  {fr_str:>16s}  {delta_str:>8s}")

# ============================================================
# Section 5: Fixed-Unique Stability Test
# ============================================================

print("\n" + "=" * 70)
print("SECTION 5: FP RATE STABILITY (FIXED UNIQUE, VARYING LENGTH)")
print("=" * 70)
print("\nIf Bloom filter theory holds, FP rate should be CONSTANT across lengths")
print("(same unique tokens = same filter load, regardless of repetitions)\n")

stability_results = {}
for head in BLOOM_HEADS:
    hk = f"L{head[0]}H{head[1]}"
    fp_rates = [
        fixed_unique_results[f"length_{sl}"][hk]["fp_rate"]
        for sl in seq_lengths
    ]
    mean_fp = float(np.mean(fp_rates))
    std_fp = float(np.std(fp_rates))
    cv = std_fp / mean_fp if mean_fp > 0 else float('inf')

    stability_results[hk] = {
        "fp_rates_by_length": {str(sl): round(fp, 4) for sl, fp in zip(seq_lengths, fp_rates)},
        "mean": round(mean_fp, 4),
        "std": round(std_fp, 4),
        "cv": round(cv, 4),
    }

    print(f"  {hk}: ", end="")
    for sl, fp in zip(seq_lengths, fp_rates):
        print(f"len={sl}→{fp:.4f}  ", end="")
    print(f"  | mean={mean_fp:.4f}, std={std_fp:.4f}, CV={cv:.4f}")

# ============================================================
# VERDICT
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 7 VERDICT")
print("=" * 70)

# Check if fixed-length R² is still good
fixed_r2_vals = [
    fixed_length_fit[f"L{h[0]}H{h[1]}"].get("r_squared", 0)
    for h in BLOOM_HEADS
    if isinstance(fixed_length_fit.get(f"L{h[0]}H{h[1]}"), dict)
    and "r_squared" in fixed_length_fit[f"L{h[0]}H{h[1]}"]
]
mean_fixed_r2 = float(np.mean(fixed_r2_vals)) if fixed_r2_vals else 0.0

# Check stability
stability_cvs = [
    stability_results[f"L{h[0]}H{h[1]}"]["cv"]
    for h in BLOOM_HEADS
    if f"L{h[0]}H{h[1]}" in stability_results
]
mean_cv = float(np.mean(stability_cvs)) if stability_cvs else float('inf')

print(f"\n1. FIXED-LENGTH BLOOM FIT:")
print(f"   Mean R² across Bloom heads = {mean_fixed_r2:.4f}")
if mean_fixed_r2 > 0.8:
    fit_verdict = ("STRONG: Bloom filter curve fits well even at fixed length. "
                   "Length confound RULED OUT.")
elif mean_fixed_r2 > 0.5:
    fit_verdict = ("MODERATE: Reasonable fit. Confound partially addressed.")
else:
    fit_verdict = ("WEAK: Poor fit at fixed length. Length effects may be a real confound.")
print(f"   → {fit_verdict}")

print(f"\n2. FP STABILITY (fixed unique tokens):")
print(f"   Mean CV across Bloom heads = {mean_cv:.4f}")
if mean_cv < 0.2:
    stab_verdict = ("STABLE: FP rate is roughly constant when unique token count is fixed. "
                    "Confirms dependence on unique tokens, not sequence length.")
elif mean_cv < 0.5:
    stab_verdict = ("SOMEWHAT STABLE: FP rate varies modestly with length. "
                    "Mostly consistent with Bloom filter theory.")
else:
    stab_verdict = ("UNSTABLE: FP rate varies substantially with length. "
                    "Possible length/positional confound remains.")
print(f"   → {stab_verdict}")

overall_verdict = (f"Fixed-length R²={mean_fixed_r2:.4f}, stability CV={mean_cv:.4f}. "
                   f"{fit_verdict} {stab_verdict}")
print(f"\n3. OVERALL: {overall_verdict}")

# ============================================================
# Save results
# ============================================================

results = {
    "experiment": "capacity_confound_control",
    "model": "gpt2-small",
    "device": device,
    "d_head": d_head,
    "bloom_heads": [list(h) for h in BLOOM_HEADS],
    "control_heads": [list(h) for h in CONTROL_HEADS],
    "parameters": {
        "n_trials": N_TRIALS,
        "n_probes": N_PROBES,
        "fp_threshold": FP_THRESHOLD,
    },
    "control_1_fixed_length": {
        "description": "Fixed sequence length, vary unique token count",
        "total_length": FIXED_LENGTH,
        "unique_counts": unique_counts,
        "results": fixed_length_results,
    },
    "control_2_fixed_unique": {
        "description": "Fixed unique token count, vary sequence length",
        "n_unique": FIXED_UNIQUE,
        "seq_lengths": seq_lengths,
        "results": fixed_unique_results,
    },
    "curve_fitting": {
        "fixed_length_fit": fixed_length_fit,
        "variable_length_fit": variable_length_fit,
        "comparison_table": comparison_table,
    },
    "stability_analysis": stability_results,
    "verdict": {
        "mean_fixed_r2": round(mean_fixed_r2, 4),
        "mean_stability_cv": round(mean_cv, 4),
        "fit_verdict": fit_verdict,
        "stability_verdict": stab_verdict,
        "overall": overall_verdict,
    },
}

results_dir = "/Users/pabalogh/clawd/projects/bloom-filter-heads/results"
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, "experiment7_capacity_confound_control.json")

with open(output_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("Done!")
