"""
Experiment 5: Ablation Study — Causal Evidence for Bloom Filter Heads

Key question: Does ablating Bloom filter heads hurt the model MORE on 
sentences with repeated tokens than on novel-token sentences?

If Bloom heads function as membership testers ("has this token appeared before?"),
then ablating them should:
  1. Increase perplexity overall (they contribute to prediction)
  2. Increase perplexity MORE on repeated-token sentences (where membership testing matters)
  3. Increase perplexity LESS than ablating induction heads (which do heavier lifting)

We compare ablation conditions:
  A. Baseline (no ablation)
  B. Ablate 4 Bloom filter heads: (0,1), (0,5), (1,11), (3,0)
  C. Ablate 4 random control heads from similar layers
  D. Ablate 4 induction heads: (5,5), (7,10), (6,9), (5,1)
  E. Ablate all 8: Bloom + induction heads together
"""

import torch
import numpy as np
import json
import os
from transformer_lens import HookedTransformer
from functools import partial

# ============================================================
# Setup
# ============================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")

# ============================================================
# Head groups to ablate
# ============================================================

# Bloom filter heads (identified in experiments 1-4)
BLOOM_HEADS = [(0, 1), (0, 5), (1, 11), (3, 0)]

# Induction heads (well-known in GPT-2 small)
INDUCTION_HEADS = [(5, 5), (7, 10), (6, 9), (5, 1)]

# Control heads: non-Bloom, non-induction, from similar early layers
# Avoiding known special heads — picking "boring" heads
CONTROL_HEADS = [(0, 3), (1, 4), (2, 7), (3, 6)]

# Combined: Bloom + Induction
COMBINED_HEADS = BLOOM_HEADS + INDUCTION_HEADS

# ============================================================
# Test sentences
# ============================================================

# Sentences WITH repeated tokens — Bloom heads should matter most here
repeat_sentences = [
    "The cat sat on the rug and the cat slept peacefully all day long",
    "A bright star appeared above the mountain and the star shone all night",
    "The doctor examined the patient carefully and the doctor prescribed medicine immediately",
    "My old friend called yesterday and my old friend invited me to dinner",
    "The river flows through the valley and the river feeds the lake below",
    "A tall tree stands in the garden and the tree provides cool shade",
    "The musician played the piano beautifully and the musician took a gracious bow",
    "Heavy rain fell on the city streets and the rain caused flooding everywhere",
    "The student read the book quickly and the student wrote a detailed summary",
    "A large ship sailed across the ocean and the ship reached port safely",
]

# Sentences WITHOUT repeated content words — Bloom heads less relevant
no_repeat_sentences = [
    "The cat sat on a comfortable rug while dogs played outside joyfully",
    "A bright star appeared above the distant mountain as clouds drifted slowly past",
    "Several doctors examined their patients carefully before prescribing any new medication today",
    "An elderly woman called from across the street inviting neighbors to brunch",
    "Water flows through narrow valleys feeding vast lakes with fresh mountain runoff",
    "Tall pine trees stand proudly in gardens providing ample shade for visitors",
    "Skilled musicians played grand pianos beautifully before taking humble bows on stage",
    "Heavy rainfall struck coastal cities causing widespread flooding throughout low terrain areas",
    "Diligent students read complex textbooks quickly then wrote comprehensive summaries of chapters",
    "Massive cargo vessels sailed across deep oceans reaching distant ports on schedule",
]

# Mixed diverse sentences for overall perplexity
diverse_sentences = [
    "The quick brown fox jumps over the lazy dog near the old barn",
    "Scientists discovered a new species of deep-sea fish in the Pacific Ocean",
    "She ordered a cappuccino and a croissant at the small French bakery",
    "The stock market experienced significant volatility during the third quarter earnings season",
    "Children laughed and played in the park while their parents watched nearby",
    "The ancient ruins of Pompeii attract millions of tourists from around the world",
    "He carefully assembled the furniture using only the instructions and a screwdriver",
    "Lightning illuminated the dark sky as thunder rolled across the empty plains",
    "The committee voted unanimously to approve the new environmental protection regulations",
    "A gentle breeze carried the scent of jasmine through the open window",
    "The algorithm processes millions of data points to generate accurate weather forecasts",
    "She played a haunting melody on the violin that moved everyone to tears",
    "Fresh snow covered the mountain peaks creating a breathtaking winter landscape scene",
    "The detective examined the crime scene thoroughly looking for any overlooked evidence",
    "Astronomers detected gravitational waves from two merging black holes billions of light years away",
    "The chef combined unexpected ingredients to create an innovative and delicious fusion dish",
    "Morning fog rolled in from the coast blanketing the harbor in gray mist",
    "Students debated the philosophical implications of artificial intelligence in their ethics seminar",
    "The marathon runner crossed the finish line exhausted but triumphant after months of training",
    "An orchestra performed Beethoven's ninth symphony to a captivated audience in the grand hall",
]

# ============================================================
# Perplexity computation
# ============================================================

def compute_perplexity(model, sentences, ablation_heads=None):
    """
    Compute perplexity over a list of sentences.
    If ablation_heads is provided, zero out those heads' contributions
    via hooks on blocks.{layer}.attn.hook_result.
    
    hook_result shape: [batch, pos, n_heads, d_model]
    We zero out the head dimension for specified heads.
    """
    
    total_log_prob = 0.0
    total_tokens = 0
    
    # Build hook functions for ablation
    # hook_z shape: [batch, pos, n_heads, d_head] — per-head output before W_O projection
    # Zeroing hook_z for a head removes that head's contribution entirely
    fwd_hooks = []
    if ablation_heads:
        # Group heads by layer for efficiency
        layer_heads = {}
        for (layer, head) in ablation_heads:
            if layer not in layer_heads:
                layer_heads[layer] = []
            layer_heads[layer].append(head)
        
        for layer, heads in layer_heads.items():
            hook_name = f"blocks.{layer}.attn.hook_z"
            
            def make_hook(heads_to_zero):
                def hook_fn(value, hook):
                    # value shape: [batch, pos, n_heads, d_head]
                    modified = value.clone()
                    for h in heads_to_zero:
                        modified[:, :, h, :] = 0.0
                    return modified
                return hook_fn
            
            fwd_hooks.append((hook_name, make_hook(heads)))
    
    for sentence in sentences:
        tokens = model.to_tokens(sentence)  # [1, seq_len]
        seq_len = tokens.shape[1]
        
        if seq_len < 2:
            continue
        
        with torch.no_grad():
            if fwd_hooks:
                logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)
            else:
                logits = model(tokens)
        
        # logits shape: [1, seq_len, vocab_size]
        # We want P(token_t | token_0..t-1) for t = 1..seq_len-1
        log_probs = torch.nn.functional.log_softmax(logits[0, :-1, :], dim=-1)
        target_tokens = tokens[0, 1:]
        
        # Gather log probs for actual next tokens
        token_log_probs = log_probs.gather(1, target_tokens.unsqueeze(1)).squeeze(1)
        
        total_log_prob += token_log_probs.sum().item()
        total_tokens += (seq_len - 1)
    
    avg_neg_log_prob = -total_log_prob / total_tokens
    perplexity = np.exp(avg_neg_log_prob)
    
    return {
        "perplexity": float(perplexity),
        "avg_neg_log_prob": float(avg_neg_log_prob),
        "total_tokens": total_tokens
    }


# ============================================================
# Run ablation conditions
# ============================================================

conditions = {
    "baseline": None,
    "ablate_bloom_heads": BLOOM_HEADS,
    "ablate_control_heads": CONTROL_HEADS,
    "ablate_induction_heads": INDUCTION_HEADS,
    "ablate_bloom_and_induction": COMBINED_HEADS,
}

sentence_sets = {
    "diverse": diverse_sentences,
    "with_repeats": repeat_sentences,
    "without_repeats": no_repeat_sentences,
}

results = {
    "experiment": "ablation_study",
    "model": "gpt2-small",
    "device": device,
    "bloom_heads": [list(h) for h in BLOOM_HEADS],
    "induction_heads": [list(h) for h in INDUCTION_HEADS],
    "control_heads": [list(h) for h in CONTROL_HEADS],
    "conditions": {},
}

print("\n" + "=" * 70)
print("EXPERIMENT 5: ABLATION STUDY")
print("=" * 70)

for cond_name, ablation_heads in conditions.items():
    print(f"\n--- Condition: {cond_name} ---")
    if ablation_heads:
        print(f"    Ablating heads: {ablation_heads}")
    else:
        print(f"    No ablation (baseline)")
    
    cond_results = {}
    
    for set_name, sentences in sentence_sets.items():
        result = compute_perplexity(model, sentences, ablation_heads)
        cond_results[set_name] = result
        print(f"    {set_name:20s}: perplexity = {result['perplexity']:8.2f}  "
              f"(avg NLL = {result['avg_neg_log_prob']:.4f}, {result['total_tokens']} tokens)")
    
    results["conditions"][cond_name] = cond_results

# ============================================================
# Compute deltas from baseline
# ============================================================

print("\n" + "=" * 70)
print("PERPLEXITY CHANGES FROM BASELINE")
print("=" * 70)

baseline = results["conditions"]["baseline"]
deltas = {}

for cond_name, cond_results in results["conditions"].items():
    if cond_name == "baseline":
        continue
    
    deltas[cond_name] = {}
    print(f"\n--- {cond_name} ---")
    
    for set_name in sentence_sets:
        base_ppl = baseline[set_name]["perplexity"]
        abl_ppl = cond_results[set_name]["perplexity"]
        abs_change = abl_ppl - base_ppl
        pct_change = (abs_change / base_ppl) * 100
        
        deltas[cond_name][set_name] = {
            "baseline_perplexity": base_ppl,
            "ablated_perplexity": abl_ppl,
            "absolute_change": abs_change,
            "percent_change": pct_change,
        }
        
        print(f"    {set_name:20s}: {base_ppl:8.2f} → {abl_ppl:8.2f}  "
              f"(Δ = {abs_change:+8.2f}, {pct_change:+6.2f}%)")

results["deltas"] = deltas

# ============================================================
# Key analysis: Bloom heads × repeated tokens interaction
# ============================================================

print("\n" + "=" * 70)
print("KEY ANALYSIS: Do Bloom heads matter MORE for repeated tokens?")
print("=" * 70)

bloom_delta = deltas["ablate_bloom_heads"]
control_delta = deltas["ablate_control_heads"]
induction_delta = deltas["ablate_induction_heads"]

# Bloom heads: repeat vs no-repeat difference
bloom_repeat_pct = bloom_delta["with_repeats"]["percent_change"]
bloom_norepeat_pct = bloom_delta["without_repeats"]["percent_change"]
bloom_interaction = bloom_repeat_pct - bloom_norepeat_pct

# Control heads: repeat vs no-repeat difference
control_repeat_pct = control_delta["with_repeats"]["percent_change"]
control_norepeat_pct = control_delta["without_repeats"]["percent_change"]
control_interaction = control_repeat_pct - control_norepeat_pct

# Induction heads: repeat vs no-repeat difference
induction_repeat_pct = induction_delta["with_repeats"]["percent_change"]
induction_norepeat_pct = induction_delta["without_repeats"]["percent_change"]
induction_interaction = induction_repeat_pct - induction_norepeat_pct

print(f"\nBloom heads ablation:")
print(f"  Repeated tokens:    {bloom_repeat_pct:+.2f}% perplexity change")
print(f"  No-repeat tokens:   {bloom_norepeat_pct:+.2f}% perplexity change")
print(f"  Interaction effect:  {bloom_interaction:+.2f}% (positive = hurts repeats MORE)")

print(f"\nControl heads ablation:")
print(f"  Repeated tokens:    {control_repeat_pct:+.2f}% perplexity change")
print(f"  No-repeat tokens:   {control_norepeat_pct:+.2f}% perplexity change")
print(f"  Interaction effect:  {control_interaction:+.2f}% (positive = hurts repeats MORE)")

print(f"\nInduction heads ablation:")
print(f"  Repeated tokens:    {induction_repeat_pct:+.2f}% perplexity change")
print(f"  No-repeat tokens:   {induction_norepeat_pct:+.2f}% perplexity change")
print(f"  Interaction effect:  {induction_interaction:+.2f}% (positive = hurts repeats MORE)")

interaction_analysis = {
    "bloom_heads": {
        "repeat_pct_change": bloom_repeat_pct,
        "no_repeat_pct_change": bloom_norepeat_pct,
        "interaction_effect": bloom_interaction,
    },
    "control_heads": {
        "repeat_pct_change": control_repeat_pct,
        "no_repeat_pct_change": control_norepeat_pct,
        "interaction_effect": control_interaction,
    },
    "induction_heads": {
        "repeat_pct_change": induction_repeat_pct,
        "no_repeat_pct_change": induction_norepeat_pct,
        "interaction_effect": induction_interaction,
    },
}

# Summary verdict
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)

if bloom_interaction > control_interaction and bloom_interaction > 0:
    verdict = ("POSITIVE: Bloom heads show a stronger repeat-token interaction "
               f"({bloom_interaction:+.2f}%) than control heads ({control_interaction:+.2f}%). "
               "This supports their role as membership testers.")
elif bloom_interaction > 0:
    verdict = ("WEAK: Bloom heads show some repeat-token interaction "
               f"({bloom_interaction:+.2f}%) but not clearly stronger than controls "
               f"({control_interaction:+.2f}%).")
else:
    verdict = ("NEGATIVE: Bloom heads do NOT hurt repeated-token sentences more "
               f"({bloom_interaction:+.2f}%). They may serve a different function "
               "than pure membership testing.")

print(f"\n{verdict}")

results["interaction_analysis"] = interaction_analysis
results["verdict"] = verdict

# ============================================================
# Save results
# ============================================================

results_dir = "/Users/pabalogh/clawd/projects/bloom-filter-heads/results"
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, "experiment5_ablation.json")

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print("Done!")
