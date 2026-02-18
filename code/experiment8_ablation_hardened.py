"""
Experiment 8: Hardened Ablation Study

Improvements over experiment 5:
  1. TWO ablation methods: zero ablation AND mean ablation
  2. Layer-matched controls with 10 random selections (robustness)
  3. Bootstrap 95% CIs on all deltas
  4. Head-count-matched induction comparison (4 vs 4, not 4 vs 16)
  5. Per-head individual ablation
  6. Raw per-sentence perplexity saved for distribution plots

Addresses review checklist items #10 (ablation CIs/controls) and #12 (method specification).
"""

import torch
import numpy as np
import json
import os
import sys
import random
from collections import defaultdict
from transformer_lens import HookedTransformer

sys.path.insert(0, '/Users/pabalogh/clawd/projects/bloom-filter-heads/code')
from expanded_stimuli import EXACT_REPEAT, NO_REPEAT, SEMANTIC_NEAR

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
print(f"Model: {n_layers} layers, {n_heads} heads")

# ============================================================
# Head definitions
# ============================================================

BLOOM_HEADS = [(0, 1), (0, 5), (1, 11), (3, 0)]
INDUCTION_HEADS_ALL = [(5, 5), (7, 10), (6, 9), (5, 1)]
INDUCTION_HEADS_4 = INDUCTION_HEADS_ALL[:4]  # head-count-matched

SPECIAL_HEADS = set(BLOOM_HEADS + INDUCTION_HEADS_ALL)

N_BOOTSTRAP = 10000

# ============================================================
# Mean activation calibration
# ============================================================

CALIBRATION_SENTENCES = [
    "The quick brown fox jumps over the lazy dog near the old barn.",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean.",
    "She ordered a cappuccino and a croissant at the small French bakery.",
    "The stock market experienced significant volatility during the earnings season.",
    "Children laughed and played in the park while their parents watched nearby.",
    "The ancient ruins of Pompeii attract millions of tourists from around the world.",
    "He carefully assembled the furniture using only the instructions and a screwdriver.",
    "Lightning illuminated the dark sky as thunder rolled across the empty plains.",
    "The committee voted unanimously to approve the new environmental protection regulations.",
    "A gentle breeze carried the scent of jasmine through the open window.",
    "The algorithm processes millions of data points to generate accurate weather forecasts.",
    "She played a haunting melody on the violin that moved everyone to tears.",
    "Fresh snow covered the mountain peaks creating a breathtaking winter landscape.",
    "The detective examined the crime scene thoroughly looking for overlooked evidence.",
    "Astronomers detected gravitational waves from two merging black holes far away.",
    "The chef combined unexpected ingredients to create an innovative fusion dish.",
    "Morning fog rolled in from the coast blanketing the harbor in gray mist.",
    "Students debated the philosophical implications of artificial intelligence in ethics seminar.",
    "The marathon runner crossed the finish line exhausted but triumphant after months.",
    "An orchestra performed Beethoven's ninth symphony to a captivated audience in the hall.",
    "New research suggests that sleep plays a critical role in memory consolidation.",
    "The library contains thousands of rare manuscripts dating back to the medieval period.",
    "Engineers designed a bridge capable of withstanding earthquakes measuring up to magnitude eight.",
    "The documentary explored the impact of climate change on Arctic wildlife populations.",
    "A talented young artist displayed her paintings at the downtown gallery opening night.",
    "The pharmaceutical company announced promising results from its latest clinical trial.",
    "Hikers followed the narrow trail through dense forest to reach the waterfall.",
    "The professor explained quantum entanglement using an analogy involving pairs of gloves.",
    "Satellites orbiting Earth collect vast amounts of data used for weather prediction.",
    "The small fishing village has remained largely unchanged for over two centuries.",
    "Researchers developed a new algorithm that significantly improves natural language understanding.",
    "The vintage car collection featured models from every decade of the twentieth century.",
    "A sudden power outage plunged the entire neighborhood into complete darkness.",
    "The museum curator carefully restored the damaged painting to its original condition.",
    "Volunteers planted hundreds of trees along the riverbank to prevent soil erosion.",
    "The software update introduced several new features that users had been requesting.",
    "A colony of penguins huddled together on the ice to stay warm.",
    "The architect designed a building that maximizes natural light while minimizing energy use.",
    "Heavy traffic delayed commuters by over an hour during the morning rush.",
    "The novel explores themes of identity and belonging in a multicultural society.",
    "Bees play an essential role in pollinating crops that humans depend on.",
    "The spacecraft successfully landed on Mars after a seven month journey.",
    "A group of musicians performed jazz standards at the outdoor summer festival.",
    "The election results surprised many political analysts who had predicted a different outcome.",
    "Ocean currents play a significant role in regulating global climate patterns.",
    "The startup secured funding from several prominent venture capital firms this quarter.",
    "Archaeologists unearthed a collection of bronze age tools at the excavation site.",
    "The teacher used interactive games to help students learn multiplication tables.",
    "Storm clouds gathered on the horizon signaling the approach of severe weather.",
    "The bakery's sourdough bread became famous throughout the region for its taste.",
]


def compute_mean_activations(model, sentences):
    """Compute mean hook_z activation per layer/head/position for mean ablation."""
    print("  Computing mean activations for mean ablation calibration...")
    
    # Accumulate sum and count per (layer, head)
    # hook_z shape: [batch, pos, n_heads, d_head]
    # We average over batch, position → per-head mean vector of shape [d_head]
    sums = {}
    counts = {}
    
    for layer in range(n_layers):
        for head in range(n_heads):
            sums[(layer, head)] = torch.zeros(model.cfg.d_head, device=device)
            counts[(layer, head)] = 0
    
    for i, sent in enumerate(sentences):
        if i % 10 == 0:
            print(f"    Calibrating: {i}/{len(sentences)}...")
        tokens = model.to_tokens(sent)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        for layer in range(n_layers):
            z = cache[f"blocks.{layer}.attn.hook_z"][0]  # [pos, n_heads, d_head]
            for head in range(n_heads):
                sums[(layer, head)] += z[:, head, :].sum(dim=0)
                counts[(layer, head)] += z.shape[0]
    
    means = {}
    for key in sums:
        means[key] = sums[key] / counts[key]
    
    print(f"    Calibrated on {len(sentences)} sentences")
    return means


# ============================================================
# Perplexity computation (per-sentence)
# ============================================================

def compute_per_sentence_perplexity(model, sentences, ablation_heads=None, 
                                      method="zero", mean_activations=None):
    """
    Compute perplexity for each sentence individually.
    
    method: "zero" (set hook_z to 0) or "mean" (replace with calibration mean)
    Returns list of per-sentence perplexities.
    """
    per_sentence_ppl = []
    
    fwd_hooks = []
    if ablation_heads:
        layer_heads = defaultdict(list)
        for (layer, head) in ablation_heads:
            layer_heads[layer].append(head)
        
        for layer, heads in layer_heads.items():
            hook_name = f"blocks.{layer}.attn.hook_z"
            
            if method == "zero":
                def make_hook(heads_to_zero):
                    def hook_fn(value, hook):
                        modified = value.clone()
                        for h in heads_to_zero:
                            modified[:, :, h, :] = 0.0
                        return modified
                    return hook_fn
                fwd_hooks.append((hook_name, make_hook(heads)))
            
            elif method == "mean":
                def make_mean_hook(layer_idx, heads_to_replace, means):
                    def hook_fn(value, hook):
                        modified = value.clone()
                        for h in heads_to_replace:
                            modified[:, :, h, :] = means[(layer_idx, h)]
                        return modified
                    return hook_fn
                fwd_hooks.append((hook_name, make_mean_hook(layer, heads, mean_activations)))
    
    for sentence in sentences:
        tokens = model.to_tokens(sentence)
        seq_len = tokens.shape[1]
        if seq_len < 2:
            per_sentence_ppl.append(float('nan'))
            continue
        
        with torch.no_grad():
            if fwd_hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            else:
                logits = model(tokens)
        
        log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
        target_tokens = tokens[0, 1:]
        token_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
        
        avg_nll = -token_log_probs.mean().item()
        ppl = np.exp(avg_nll)
        per_sentence_ppl.append(float(ppl))
    
    return per_sentence_ppl


def bootstrap_delta(baseline_ppls, ablated_ppls, n_bootstrap=N_BOOTSTRAP):
    """Bootstrap CI for percent change in perplexity."""
    base = np.array(baseline_ppls)
    abl = np.array(ablated_ppls)
    
    # Remove any NaN pairs
    mask = ~(np.isnan(base) | np.isnan(abl))
    base, abl = base[mask], abl[mask]
    
    observed_delta_pct = 100 * (np.mean(abl) - np.mean(base)) / np.mean(base)
    
    boot_deltas = []
    n = len(base)
    for _ in range(n_bootstrap):
        idx = np.random.randint(0, n, size=n)
        b_mean = np.mean(base[idx])
        a_mean = np.mean(abl[idx])
        boot_deltas.append(100 * (a_mean - b_mean) / b_mean)
    
    boot_deltas = np.array(boot_deltas)
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))
    
    return {
        "mean_delta_pct": round(float(observed_delta_pct), 2),
        "ci_95": [round(ci_lo, 2), round(ci_hi, 2)],
        "baseline_mean_ppl": round(float(np.mean(base)), 2),
        "ablated_mean_ppl": round(float(np.mean(abl)), 2),
    }


# ============================================================
# Generate layer-matched control selections
# ============================================================

def generate_layer_matched_controls(bloom_heads, n_selections=10):
    """For each Bloom head, pick a random non-special head from the same layer."""
    all_selections = []
    
    for sel in range(n_selections):
        controls = []
        for (layer, head) in bloom_heads:
            candidates = [(layer, h) for h in range(n_heads) 
                         if (layer, h) not in SPECIAL_HEADS and (layer, h) not in controls]
            chosen = random.choice(candidates)
            controls.append(chosen)
        all_selections.append(controls)
    
    return all_selections


# ============================================================
# MAIN EXPERIMENT
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 8: HARDENED ABLATION STUDY")
print("=" * 70)

# Step 1: Calibrate mean activations
print("\n--- Step 1: Mean Activation Calibration ---")
mean_activations = compute_mean_activations(model, CALIBRATION_SENTENCES)

# Step 2: Compute baselines
print("\n--- Step 2: Baseline Perplexities ---")
print(f"  Computing on {len(EXACT_REPEAT)} repeat + {len(NO_REPEAT)} no-repeat sentences...")

baseline_repeat = compute_per_sentence_perplexity(model, EXACT_REPEAT)
baseline_norepeat = compute_per_sentence_perplexity(model, NO_REPEAT)

print(f"  Repeat baseline: mean PPL = {np.mean(baseline_repeat):.2f}")
print(f"  No-repeat baseline: mean PPL = {np.mean(baseline_norepeat):.2f}")

# Step 3: Ablation conditions
results = {
    "experiment": "ablation_hardened",
    "model": "gpt2-small",
    "bloom_heads": [list(h) for h in BLOOM_HEADS],
    "induction_heads_4": [list(h) for h in INDUCTION_HEADS_4],
    "n_bootstrap": N_BOOTSTRAP,
    "n_repeat_sentences": len(EXACT_REPEAT),
    "n_norepeat_sentences": len(NO_REPEAT),
    "n_calibration_sentences": len(CALIBRATION_SENTENCES),
    "methods": {},
}

for method in ["zero", "mean"]:
    print(f"\n{'=' * 70}")
    print(f"METHOD: {method.upper()} ABLATION")
    print(f"{'=' * 70}")
    
    method_results = {}
    
    # 3a: Bloom head ablation
    print(f"\n--- Ablating Bloom heads ({method}) ---")
    bloom_repeat = compute_per_sentence_perplexity(
        model, EXACT_REPEAT, BLOOM_HEADS, method=method, mean_activations=mean_activations)
    bloom_norepeat = compute_per_sentence_perplexity(
        model, NO_REPEAT, BLOOM_HEADS, method=method, mean_activations=mean_activations)
    
    bloom_repeat_delta = bootstrap_delta(baseline_repeat, bloom_repeat)
    bloom_norepeat_delta = bootstrap_delta(baseline_norepeat, bloom_norepeat)
    bloom_interaction = round(bloom_repeat_delta["mean_delta_pct"] - bloom_norepeat_delta["mean_delta_pct"], 2)
    
    print(f"  Repeat:    {bloom_repeat_delta['mean_delta_pct']:+.2f}% [{bloom_repeat_delta['ci_95'][0]:+.2f}, {bloom_repeat_delta['ci_95'][1]:+.2f}]")
    print(f"  No-repeat: {bloom_norepeat_delta['mean_delta_pct']:+.2f}% [{bloom_norepeat_delta['ci_95'][0]:+.2f}, {bloom_norepeat_delta['ci_95'][1]:+.2f}]")
    print(f"  Interaction: {bloom_interaction:+.2f}%")
    
    method_results["bloom_heads"] = {
        "repeat_delta": bloom_repeat_delta,
        "norepeat_delta": bloom_norepeat_delta,
        "interaction": bloom_interaction,
        "raw_repeat_ppl": bloom_repeat,
        "raw_norepeat_ppl": bloom_norepeat,
    }
    
    # 3b: Head-count-matched induction heads (4 only)
    print(f"\n--- Ablating 4 Induction heads ({method}) ---")
    ind_repeat = compute_per_sentence_perplexity(
        model, EXACT_REPEAT, INDUCTION_HEADS_4, method=method, mean_activations=mean_activations)
    ind_norepeat = compute_per_sentence_perplexity(
        model, NO_REPEAT, INDUCTION_HEADS_4, method=method, mean_activations=mean_activations)
    
    ind_repeat_delta = bootstrap_delta(baseline_repeat, ind_repeat)
    ind_norepeat_delta = bootstrap_delta(baseline_norepeat, ind_norepeat)
    ind_interaction = round(ind_repeat_delta["mean_delta_pct"] - ind_norepeat_delta["mean_delta_pct"], 2)
    
    print(f"  Repeat:    {ind_repeat_delta['mean_delta_pct']:+.2f}% [{ind_repeat_delta['ci_95'][0]:+.2f}, {ind_repeat_delta['ci_95'][1]:+.2f}]")
    print(f"  No-repeat: {ind_norepeat_delta['mean_delta_pct']:+.2f}% [{ind_norepeat_delta['ci_95'][0]:+.2f}, {ind_norepeat_delta['ci_95'][1]:+.2f}]")
    print(f"  Interaction: {ind_interaction:+.2f}%")
    
    method_results["induction_heads_4"] = {
        "repeat_delta": ind_repeat_delta,
        "norepeat_delta": ind_norepeat_delta,
        "interaction": ind_interaction,
        "raw_repeat_ppl": ind_repeat,
        "raw_norepeat_ppl": ind_norepeat,
    }
    
    # 3c: Layer-matched controls (10 random selections)
    print(f"\n--- Layer-matched controls: 10 random selections ({method}) ---")
    control_selections = generate_layer_matched_controls(BLOOM_HEADS, n_selections=10)
    
    control_interactions = []
    control_repeat_deltas = []
    control_norepeat_deltas = []
    control_details = []
    
    for sel_idx, control_heads in enumerate(control_selections):
        print(f"  Selection {sel_idx+1}/10: {control_heads}")
        
        ctrl_repeat = compute_per_sentence_perplexity(
            model, EXACT_REPEAT, control_heads, method=method, mean_activations=mean_activations)
        ctrl_norepeat = compute_per_sentence_perplexity(
            model, NO_REPEAT, control_heads, method=method, mean_activations=mean_activations)
        
        ctrl_r_delta = bootstrap_delta(baseline_repeat, ctrl_repeat)
        ctrl_nr_delta = bootstrap_delta(baseline_norepeat, ctrl_norepeat)
        ctrl_interaction = round(ctrl_r_delta["mean_delta_pct"] - ctrl_nr_delta["mean_delta_pct"], 2)
        
        control_interactions.append(ctrl_interaction)
        control_repeat_deltas.append(ctrl_r_delta["mean_delta_pct"])
        control_norepeat_deltas.append(ctrl_nr_delta["mean_delta_pct"])
        control_details.append({
            "heads": [list(h) for h in control_heads],
            "repeat_delta_pct": ctrl_r_delta["mean_delta_pct"],
            "norepeat_delta_pct": ctrl_nr_delta["mean_delta_pct"],
            "interaction": ctrl_interaction,
        })
    
    ctrl_int_mean = round(float(np.mean(control_interactions)), 2)
    ctrl_int_std = round(float(np.std(control_interactions)), 2)
    
    print(f"\n  Control interaction: {ctrl_int_mean:+.2f}% ± {ctrl_int_std:.2f}%")
    print(f"  Bloom interaction:   {bloom_interaction:+.2f}%")
    print(f"  Bloom is {abs(bloom_interaction - ctrl_int_mean) / max(ctrl_int_std, 0.01):.1f}σ from control mean")
    
    method_results["layer_matched_controls"] = {
        "n_selections": 10,
        "interaction_mean": ctrl_int_mean,
        "interaction_std": ctrl_int_std,
        "repeat_delta_mean": round(float(np.mean(control_repeat_deltas)), 2),
        "norepeat_delta_mean": round(float(np.mean(control_norepeat_deltas)), 2),
        "details": control_details,
        "bloom_sigma_from_control": round(abs(bloom_interaction - ctrl_int_mean) / max(ctrl_int_std, 0.01), 1),
    }
    
    # 3d: Per-head individual ablation
    print(f"\n--- Per-head individual ablation ({method}) ---")
    per_head_results = {}
    
    for (layer, head) in BLOOM_HEADS:
        head_key = f"L{layer}H{head}"
        single_head = [(layer, head)]
        
        h_repeat = compute_per_sentence_perplexity(
            model, EXACT_REPEAT, single_head, method=method, mean_activations=mean_activations)
        h_norepeat = compute_per_sentence_perplexity(
            model, NO_REPEAT, single_head, method=method, mean_activations=mean_activations)
        
        h_r_delta = bootstrap_delta(baseline_repeat, h_repeat)
        h_nr_delta = bootstrap_delta(baseline_norepeat, h_norepeat)
        h_interaction = round(h_r_delta["mean_delta_pct"] - h_nr_delta["mean_delta_pct"], 2)
        
        per_head_results[head_key] = {
            "repeat_delta": h_r_delta,
            "norepeat_delta": h_nr_delta,
            "interaction": h_interaction,
        }
        
        print(f"  {head_key}: repeat {h_r_delta['mean_delta_pct']:+.2f}%, "
              f"no-repeat {h_nr_delta['mean_delta_pct']:+.2f}%, "
              f"interaction {h_interaction:+.2f}%")
    
    method_results["per_head"] = per_head_results
    results["methods"][method] = method_results

# Save baselines
results["baselines"] = {
    "repeat_ppl": baseline_repeat,
    "norepeat_ppl": baseline_norepeat,
    "repeat_mean": round(float(np.mean(baseline_repeat)), 2),
    "norepeat_mean": round(float(np.mean(baseline_norepeat)), 2),
}

# ============================================================
# VERDICT
# ============================================================

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

for method in ["zero", "mean"]:
    m = results["methods"][method]
    bloom_int = m["bloom_heads"]["interaction"]
    ctrl_int_mean = m["layer_matched_controls"]["interaction_mean"]
    ctrl_int_std = m["layer_matched_controls"]["interaction_std"]
    ind_int = m["induction_heads_4"]["interaction"]
    sigma = m["layer_matched_controls"]["bloom_sigma_from_control"]
    
    print(f"\n{method.upper()} ABLATION:")
    print(f"  Bloom interaction:     {bloom_int:+.2f}%")
    print(f"  Control interaction:   {ctrl_int_mean:+.2f}% ± {ctrl_int_std:.2f}%")
    print(f"  Induction interaction: {ind_int:+.2f}%")
    print(f"  Bloom is {sigma:.1f}σ from control mean")
    
    if bloom_int > ctrl_int_mean + 2 * ctrl_int_std:
        print(f"  ✅ Bloom heads show SIGNIFICANTLY stronger repeat-specific effect")
    elif bloom_int > ctrl_int_mean:
        print(f"  🟡 Bloom heads show stronger repeat-specific effect (but < 2σ)")
    else:
        print(f"  ❌ Bloom heads do NOT show stronger repeat-specific effect")

# ============================================================
# Save
# ============================================================

results_path = "/Users/pabalogh/clawd/projects/bloom-filter-heads/results/experiment8_ablation_hardened.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {results_path}")
print("Done!")
