"""
Experiment 1: Do any attention heads behave like Bloom filters?

A Bloom filter has the signature:
  - No false negatives (if X is in the set, always returns True)
  - False positives possible (sometimes returns True for X not in set)

We test this by:
  1. Constructing sequences with repeated tokens at known positions
  2. Measuring per-head attention from the SECOND occurrence back to the FIRST
  3. Checking if any heads show the Bloom filter signature:
     - High attention to first occurrence when token IS repeated (hit rate)
     - Near-zero failure to attend when token IS repeated (miss rate ≈ 0)
     - Some attention to "similar" tokens that aren't actually repeated (false positive)
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from collections import defaultdict
import json
import os

# Use MPS if available, else CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Load GPT-2 small
print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer")

# ============================================================
# Test sequences
# ============================================================

# Category 1: EXACT REPEATS — token appears twice
# The second occurrence should trigger Bloom-filter heads to attend to the first
exact_repeat_sentences = [
    "The cat sat on the rug and the cat slept peacefully",
    "A bright star appeared above the mountain and the star shone all night",
    "The doctor examined the patient carefully and the doctor prescribed medicine",
    "My old friend called yesterday and my old friend invited me to dinner",
    "The river flows through the valley and the river feeds the lake below",
    "A tall tree stands in the garden and the tree provides shade",
    "The musician played the piano beautifully and the musician took a bow",
    "Heavy rain fell on the city and the rain caused flooding everywhere",
    "The student read the book quickly and the student wrote a summary",
    "A large ship sailed across the ocean and the ship reached port safely",
    "The painter mixed the colors carefully and the painter created a masterpiece",
    "An old clock hung on the wall and the clock chimed every hour",
    "The scientist conducted the experiment twice and the scientist published results",
    "A small bird landed on the fence and the bird sang a melody",
    "The chef prepared the meal slowly and the chef served it hot",
]

# Category 2: NO REPEATS — all content words are unique
no_repeat_sentences = [
    "The cat sat on the rug while a dog slept peacefully nearby",
    "A bright star appeared above the mountain as clouds drifted slowly past",
    "The doctor examined the patient carefully before writing a new prescription",
    "My old friend called yesterday with exciting news about a promotion",
    "The river flows through the valley carrying leaves and small branches downstream",
    "A tall tree stands in the garden providing shelter for nesting birds",
    "The musician played the piano beautifully receiving applause from every listener",
    "Heavy rain fell on the city flooding several streets and parking lots",
    "The student read the book quickly finishing before anyone else had started",
    "A large ship sailed across the ocean encountering dolphins along its path",
    "The painter mixed the colors carefully producing shades nobody had seen before",
    "An old clock hung on the wall ticking steadily through quiet afternoons",
    "The scientist conducted the experiment twice achieving consistent and reliable measurements",
    "A small bird landed on the fence chirping loudly at passing strangers",
    "The chef prepared the meal slowly adding herbs and spices with care",
]

# Category 3: SEMANTIC NEAR-MISSES — similar but not identical tokens
# These test for "false positives" — does the head fire for synonyms?
semantic_near_sentences = [
    "The cat sat on the rug and the kitten slept peacefully nearby",
    "A bright star appeared above the mountain and the sun shone all day",
    "The doctor examined the patient carefully and the physician prescribed medicine",
    "My old friend called yesterday and my close companion invited me over",
    "The river flows through the valley and the stream feeds the lake",
    "A tall tree stands in the garden and the bush provides shade",
    "The musician played the piano beautifully and the pianist took a bow",
    "Heavy rain fell on the city and the downpour caused flooding everywhere",
    "The student read the book quickly and the pupil wrote a summary",
    "A large ship sailed across the ocean and the vessel reached port",
    "The painter mixed the colors carefully and the artist created a masterpiece",
    "An old clock hung on the wall and the watch chimed every hour",
    "The scientist conducted the experiment twice and the researcher published results",
    "A small bird landed on the fence and the sparrow sang a melody",
    "The chef prepared the meal slowly and the cook served it hot",
]


def get_attention_patterns(model, text):
    """Run model and extract per-head attention patterns."""
    tokens = model.to_tokens(text)
    str_tokens = model.to_str_tokens(text)

    # Run with cache
    _, cache = model.run_with_cache(tokens)

    # Extract attention patterns: [layer, head, dest, src]
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = tokens.shape[1]

    attention = torch.zeros(n_layers, n_heads, seq_len, seq_len)
    for layer in range(n_layers):
        # pattern shape: [batch, head, dest, src]
        pattern = cache[f"blocks.{layer}.attn.hook_pattern"]
        attention[layer] = pattern[0].cpu()

    return attention, str_tokens, tokens[0].cpu()


def find_repeated_token_positions(tokens):
    """Find positions where tokens repeat (second+ occurrence → first occurrence)."""
    token_list = tokens.tolist()
    first_occurrence = {}
    repeat_pairs = []  # (second_pos, first_pos, token_id)

    for pos, tok in enumerate(token_list):
        if tok in first_occurrence:
            repeat_pairs.append((pos, first_occurrence[tok], tok))
        else:
            first_occurrence[tok] = pos

    return repeat_pairs


def find_nonrepeat_positions(tokens):
    """Find positions of tokens that appear exactly once."""
    token_list = tokens.tolist()
    from collections import Counter
    counts = Counter(token_list)
    unique_positions = [pos for pos, tok in enumerate(token_list) if counts[tok] == 1]
    return unique_positions


def analyze_bloom_signature(model, sentences, category_name):
    """
    For each sentence, measure per-head attention from repeated tokens
    back to their first occurrence.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # Per-head metrics
    hit_attention = defaultdict(list)       # attention from repeat → first occurrence
    miss_attention = defaultdict(list)      # 1 - hit (how often it FAILS to attend)
    baseline_attention = defaultdict(list)  # attention from non-repeat → random earlier position

    for sent in sentences:
        attention, str_tokens, tokens = get_attention_patterns(model, sent)
        repeat_pairs = find_repeated_token_positions(tokens)
        unique_positions = find_nonrepeat_positions(tokens)

        for second_pos, first_pos, tok_id in repeat_pairs:
            for layer in range(n_layers):
                for head in range(n_heads):
                    # How much does the second occurrence attend to the first?
                    attn_to_first = attention[layer, head, second_pos, first_pos].item()
                    hit_attention[(layer, head)].append(attn_to_first)

        # Baseline: for unique tokens, how much do they attend to a random earlier position?
        for pos in unique_positions:
            if pos <= 1:
                continue
            # Pick a random earlier position
            rand_src = np.random.randint(1, pos)  # skip BOS
            for layer in range(n_layers):
                for head in range(n_heads):
                    attn_baseline = attention[layer, head, pos, rand_src].item()
                    baseline_attention[(layer, head)].append(attn_baseline)

    return hit_attention, baseline_attention


def analyze_false_positives(model, semantic_sentences, exact_sentences):
    """
    For semantic near-miss sentences, measure attention from the synonym position
    to where the original word appeared.
    Compare to exact repeat attention.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    # We need to find synonym pairs by comparing token positions
    # between exact repeat and semantic near-miss sentences
    synonym_attention = defaultdict(list)

    for exact_sent, sem_sent in zip(exact_sentences, semantic_sentences):
        # Get tokens for both
        exact_tokens = model.to_tokens(exact_sent)[0].cpu()
        sem_tokens = model.to_tokens(sem_sent)[0].cpu()
        exact_str = model.to_str_tokens(exact_sent)
        sem_str = model.to_str_tokens(sem_sent)

        # Find exact repeat positions in the exact sentence
        exact_repeats = find_repeated_token_positions(exact_tokens)

        # For the semantic version, look at the same position ranges
        # The synonym should be at approximately the same position
        sem_attention, _, _ = get_attention_patterns(model, sem_sent)

        for second_pos, first_pos, tok_id in exact_repeats:
            # In the semantic sentence, the token at second_pos is the SYNONYM
            # and first_pos has the ORIGINAL — but they're different tokens
            if second_pos < len(sem_tokens) and first_pos < len(sem_tokens):
                # Check if they're actually different tokens (synonym, not repeat)
                if sem_tokens[second_pos] != sem_tokens[first_pos]:
                    for layer in range(n_layers):
                        for head in range(n_heads):
                            attn = sem_attention[layer, head, second_pos, first_pos].item()
                            synonym_attention[(layer, head)].append(attn)

    return synonym_attention


print("\n" + "="*60)
print("EXPERIMENT 1: BLOOM FILTER SIGNATURE IN ATTENTION HEADS")
print("="*60)

# Phase 1: Exact repeats vs baseline
print("\nPhase 1: Analyzing exact repeats...")
hit_attn, baseline_attn = analyze_bloom_signature(model, exact_repeat_sentences, "exact_repeat")

print("Phase 1b: Analyzing no-repeat baseline...")
_, no_repeat_baseline = analyze_bloom_signature(model, no_repeat_sentences, "no_repeat")

# Phase 2: Semantic near-misses (false positives)
print("\nPhase 2: Analyzing semantic near-misses (false positive test)...")
synonym_attn = analyze_false_positives(model, semantic_near_sentences, exact_repeat_sentences)

# ============================================================
# Compute per-head Bloom filter scores
# ============================================================
print("\n" + "="*60)
print("RESULTS: Per-head Bloom Filter Analysis")
print("="*60)

n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads

results = []

for layer in range(n_layers):
    for head in range(n_heads):
        key = (layer, head)

        hits = hit_attn.get(key, [])
        baselines = baseline_attn.get(key, [])
        synonyms = synonym_attn.get(key, [])

        if not hits or not baselines:
            continue

        mean_hit = np.mean(hits)
        mean_baseline = np.mean(baselines)
        mean_synonym = np.mean(synonyms) if synonyms else 0.0

        # Bloom filter signature metrics:
        # 1. Hit rate: mean attention to repeated tokens (should be HIGH)
        # 2. Selectivity: hit_rate / baseline (should be HIGH — much more attention to repeats)
        # 3. False positive indicator: synonym_attention / hit_rate (should be > 0 but < 1)
        # 4. Miss rate: fraction of hits below some threshold (should be ≈ 0)

        miss_rate = sum(1 for h in hits if h < 0.01) / len(hits) if hits else 1.0
        selectivity = mean_hit / mean_baseline if mean_baseline > 0 else float('inf')
        fp_ratio = mean_synonym / mean_hit if mean_hit > 0 else 0.0

        # Bloom filter score: high hit, high selectivity, low miss rate
        # Bonus points for nonzero but sub-1.0 false positive ratio
        bloom_score = mean_hit * selectivity * (1 - miss_rate)
        if 0.05 < fp_ratio < 0.8:
            bloom_score *= 1.5  # Bonus for Bloom-filter-like FP behavior

        results.append({
            "layer": layer,
            "head": head,
            "mean_hit_attention": round(mean_hit, 4),
            "mean_baseline_attention": round(mean_baseline, 4),
            "mean_synonym_attention": round(mean_synonym, 4),
            "selectivity": round(selectivity, 2),
            "miss_rate": round(miss_rate, 4),
            "fp_ratio": round(fp_ratio, 4),
            "bloom_score": round(bloom_score, 4),
            "n_hits": len(hits),
            "n_synonyms": len(synonyms),
        })

# Sort by bloom_score
results.sort(key=lambda x: x["bloom_score"], reverse=True)

# Print top 20
print(f"\nTop 20 heads by Bloom Filter Score:")
print(f"{'Layer':>5} {'Head':>4} {'Hit Attn':>9} {'Baseline':>9} {'Synonym':>9} {'Select.':>8} {'Miss%':>6} {'FP Ratio':>9} {'Bloom':>8}")
print("-" * 80)

for r in results[:20]:
    print(f"  L{r['layer']:>2}   H{r['head']:>2}   {r['mean_hit_attention']:>7.4f}   {r['mean_baseline_attention']:>7.4f}   {r['mean_synonym_attention']:>7.4f}   {r['selectivity']:>6.1f}x   {r['miss_rate']:>5.1%}   {r['fp_ratio']:>7.4f}   {r['bloom_score']:>7.2f}")

# Print summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

# How many heads show strong Bloom filter signature?
strong_bloom = [r for r in results if r["selectivity"] > 3 and r["miss_rate"] < 0.1 and r["mean_hit_attention"] > 0.05]
moderate_bloom = [r for r in results if r["selectivity"] > 2 and r["miss_rate"] < 0.2 and r["mean_hit_attention"] > 0.03]

print(f"\nTotal heads analyzed: {len(results)}")
print(f"Strong Bloom signature (>3x selectivity, <10% miss, >5% hit): {len(strong_bloom)}")
print(f"Moderate Bloom signature (>2x selectivity, <20% miss, >3% hit): {len(moderate_bloom)}")

if strong_bloom:
    print("\n✅ STRONG BLOOM FILTER HEADS FOUND:")
    for r in strong_bloom[:10]:
        print(f"  Layer {r['layer']}, Head {r['head']}: "
              f"{r['selectivity']:.1f}x selectivity, "
              f"{r['miss_rate']:.1%} miss rate, "
              f"{r['fp_ratio']:.2f} FP ratio")
        if r['fp_ratio'] > 0.1:
            print(f"    → Shows false positive behavior (classic Bloom filter!)")
else:
    print("\n❌ No strong Bloom filter signatures found.")
    print("   This suggests attention heads blend membership testing with other functions.")

# Layer-wise analysis
print("\n" + "="*60)
print("LAYER-WISE ANALYSIS")
print("="*60)

for layer in range(n_layers):
    layer_results = [r for r in results if r["layer"] == layer]
    if layer_results:
        avg_hit = np.mean([r["mean_hit_attention"] for r in layer_results])
        avg_select = np.mean([r["selectivity"] for r in layer_results])
        avg_miss = np.mean([r["miss_rate"] for r in layer_results])
        valid_blooms = [r for r in layer_results if not np.isnan(r["bloom_score"])]
        if not valid_blooms:
            print(f"  Layer {layer:>2}: avg_hit={avg_hit:.4f}, avg_selectivity={avg_select:.1f}x, "
                  f"avg_miss={avg_miss:.1%}, best=N/A")
            continue
        max_bloom = max(r["bloom_score"] for r in valid_blooms)
        best_head = [r for r in valid_blooms if r["bloom_score"] == max_bloom][0]["head"]
        print(f"  Layer {layer:>2}: avg_hit={avg_hit:.4f}, avg_selectivity={avg_select:.1f}x, "
              f"avg_miss={avg_miss:.1%}, best=H{best_head} (bloom={max_bloom:.2f})")

# Save full results
results_path = "/Users/pabalogh/clawd/projects/bloom-filter-heads/results/experiment1_results.json"
with open(results_path, "w") as f:
    json.dump({
        "experiment": "bloom_filter_signature",
        "model": "gpt2-small",
        "n_exact_sentences": len(exact_repeat_sentences),
        "n_no_repeat_sentences": len(no_repeat_sentences),
        "n_semantic_sentences": len(semantic_near_sentences),
        "results": results,
        "summary": {
            "total_heads": len(results),
            "strong_bloom_count": len(strong_bloom),
            "moderate_bloom_count": len(moderate_bloom),
        }
    }, f, indent=2)
print(f"\nFull results saved to {results_path}")

print("\n" + "="*60)
print("VERDICT")
print("="*60)
if len(strong_bloom) >= 3:
    print("🟢 THERE IS A THERE THERE. Multiple heads show clear Bloom filter behavior.")
    print("   Proceed to Experiments 2-4 for deeper analysis.")
elif len(moderate_bloom) >= 5:
    print("🟡 PROMISING but not definitive. Several heads show partial Bloom filter behavior.")
    print("   Worth investigating further with more controlled stimuli.")
else:
    print("🔴 Weak signal. Attention heads don't cleanly separate into Bloom-filter-like behavior.")
    print("   The metaphor may be more poetic than mechanistic.")
