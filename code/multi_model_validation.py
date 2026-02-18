"""
Multi-Model Validation: Do Bloom filter heads exist across model sizes?

Test the Bloom filter hypothesis on:
1. GPT-2 small (12L, 12H) — already done, re-run for consistency
2. GPT-2 medium (24L, 16H) — larger model
3. GPT-2 large (36L, 20H) — even larger
4. Pythia-160M, Pythia-410M — different model family

This is the make-or-break: if Bloom heads generalize, the paper is strong.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from collections import defaultdict
import json
import sys
import time

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Models to test — start with what we can fit in memory
MODELS = [
    ("gpt2-small", "GPT-2 Small (12L×12H, 85M)"),
    ("gpt2-medium", "GPT-2 Medium (24L×16H, 302M)"),
    ("gpt2-large", "GPT-2 Large (36L×20H, 708M)"),
    ("pythia-160m", "Pythia-160M (12L×12H)"),
]

# Sentences — same as Experiment 1 but condensed for speed
exact_repeat_sentences = [
    "The cat sat on the rug and the cat slept peacefully",
    "A bright star appeared above the mountain and the star shone all night",
    "The doctor examined the patient carefully and the doctor prescribed medicine",
    "The river flows through the valley and the river feeds the lake below",
    "A tall tree stands in the garden and the tree provides shade",
    "The musician played the piano beautifully and the musician took a bow",
    "Heavy rain fell on the city and the rain caused flooding everywhere",
    "The student read the book quickly and the student wrote a summary",
    "A large ship sailed across the ocean and the ship reached port safely",
    "The painter mixed the colors carefully and the painter created a masterpiece",
]

no_repeat_sentences = [
    "The cat sat on the rug while a dog slept peacefully nearby",
    "A bright star appeared above the mountain as clouds drifted slowly past",
    "The doctor examined the patient carefully before writing a new prescription",
    "The river flows through the valley carrying leaves and small branches downstream",
    "A tall tree stands in the garden providing shelter for nesting birds",
    "The musician played the piano beautifully receiving applause from every listener",
    "Heavy rain fell on the city flooding several streets and parking lots",
    "The student read the book quickly finishing before anyone else had started",
    "A large ship sailed across the ocean encountering dolphins along its path",
    "The painter mixed the colors carefully producing shades nobody had seen before",
]

semantic_near_sentences = [
    "The cat sat on the rug and the kitten slept peacefully nearby",
    "A bright star appeared above the mountain and the sun shone all day",
    "The doctor examined the patient carefully and the physician prescribed medicine",
    "The river flows through the valley and the stream feeds the lake",
    "A tall tree stands in the garden and the bush provides shade",
    "The musician played the piano beautifully and the pianist took a bow",
    "Heavy rain fell on the city and the downpour caused flooding everywhere",
    "The student read the book quickly and the pupil wrote a summary",
    "A large ship sailed across the ocean and the vessel reached port",
    "The painter mixed the colors carefully and the artist created a masterpiece",
]


def find_repeated_token_positions(tokens):
    token_list = tokens.tolist()
    first_occurrence = {}
    repeat_pairs = []
    for pos, tok in enumerate(token_list):
        if tok in first_occurrence:
            repeat_pairs.append((pos, first_occurrence[tok], tok))
        else:
            first_occurrence[tok] = pos
    return repeat_pairs


def find_nonrepeat_positions(tokens):
    token_list = tokens.tolist()
    from collections import Counter
    counts = Counter(token_list)
    return [pos for pos, tok in enumerate(token_list) if counts[tok] == 1]


def analyze_model(model_name, model_label):
    """Run Bloom filter analysis on a single model."""
    print(f"\n{'='*60}")
    print(f"ANALYZING: {model_label}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        model = HookedTransformer.from_pretrained(model_name, device=device)
    except Exception as e:
        print(f"  ❌ Failed to load {model_name}: {e}")
        return None
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    print(f"  Architecture: {n_layers} layers, {n_heads} heads, d_head={d_head}")
    
    # Phase 1: Measure Bloom filter signature for all heads
    hit_attention = defaultdict(list)
    baseline_attention = defaultdict(list)
    synonym_attention = defaultdict(list)
    
    # Exact repeats
    for sent in exact_repeat_sentences:
        tokens = model.to_tokens(sent)
        _, cache = model.run_with_cache(tokens)
        tok_ids = tokens[0].cpu()
        repeat_pairs = find_repeated_token_positions(tok_ids)
        unique_positions = find_nonrepeat_positions(tok_ids)
        
        for second_pos, first_pos, tok_id in repeat_pairs:
            for layer in range(n_layers):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0].cpu()
                for head in range(n_heads):
                    attn = pattern[head, second_pos, first_pos].item()
                    hit_attention[(layer, head)].append(attn)
        
        for pos in unique_positions:
            if pos <= 1:
                continue
            rand_src = np.random.randint(1, pos)
            for layer in range(n_layers):
                pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0].cpu()
                for head in range(n_heads):
                    attn = pattern[head, pos, rand_src].item()
                    baseline_attention[(layer, head)].append(attn)
    
    # Semantic near-misses for FP analysis
    for exact_sent, sem_sent in zip(exact_repeat_sentences, semantic_near_sentences):
        exact_tokens = model.to_tokens(exact_sent)[0].cpu()
        sem_tokens = model.to_tokens(sem_sent)[0].cpu()
        exact_repeats = find_repeated_token_positions(exact_tokens)
        
        sem_input = model.to_tokens(sem_sent)
        _, sem_cache = model.run_with_cache(sem_input)
        
        for second_pos, first_pos, tok_id in exact_repeats:
            if second_pos < len(sem_tokens) and first_pos < len(sem_tokens):
                if sem_tokens[second_pos] != sem_tokens[first_pos]:
                    for layer in range(n_layers):
                        pattern = sem_cache[f"blocks.{layer}.attn.hook_pattern"][0].cpu()
                        for head in range(n_heads):
                            attn = pattern[head, second_pos, first_pos].item()
                            synonym_attention[(layer, head)].append(attn)
    
    # Compute per-head results
    results = []
    for layer in range(n_layers):
        for head in range(n_heads):
            key = (layer, head)
            hits = hit_attention.get(key, [])
            baselines = baseline_attention.get(key, [])
            synonyms = synonym_attention.get(key, [])
            
            if not hits or not baselines:
                continue
            
            mean_hit = np.mean(hits)
            mean_baseline = np.mean(baselines)
            mean_synonym = np.mean(synonyms) if synonyms else 0.0
            miss_rate = sum(1 for h in hits if h < 0.01) / len(hits) if hits else 1.0
            selectivity = mean_hit / mean_baseline if mean_baseline > 0.001 else mean_hit / 0.001
            fp_ratio = mean_synonym / mean_hit if mean_hit > 0.01 else 0.0
            
            results.append({
                "layer": layer,
                "head": head,
                "mean_hit": round(mean_hit, 4),
                "mean_baseline": round(mean_baseline, 4),
                "mean_synonym": round(mean_synonym, 4),
                "selectivity": round(selectivity, 2),
                "miss_rate": round(miss_rate, 4),
                "fp_ratio": round(fp_ratio, 4),
            })
    
    # Identify Bloom heads
    strong_bloom = [r for r in results if r["selectivity"] > 3 and r["miss_rate"] < 0.1 and r["mean_hit"] > 0.05]
    moderate_bloom = [r for r in results if r["selectivity"] > 2 and r["miss_rate"] < 0.2 and r["mean_hit"] > 0.03]
    
    elapsed = time.time() - start_time
    
    print(f"\n  Results ({elapsed:.0f}s):")
    print(f"  Total heads: {len(results)}")
    print(f"  Strong Bloom (>3x sel, <10% miss, >5% hit): {len(strong_bloom)}")
    print(f"  Moderate Bloom (>2x sel, <20% miss, >3% hit): {len(moderate_bloom)}")
    
    if strong_bloom:
        print(f"\n  Strong Bloom heads:")
        for r in sorted(strong_bloom, key=lambda x: x["selectivity"], reverse=True)[:10]:
            fp_note = " ← classic Bloom FP!" if 0.1 < r["fp_ratio"] < 0.8 else ""
            print(f"    L{r['layer']}H{r['head']}: {r['selectivity']:.1f}x selectivity, "
                  f"{r['miss_rate']:.1%} miss, FP ratio={r['fp_ratio']:.2f}{fp_note}")
    
    # Layer distribution
    bloom_layers = [r["layer"] for r in strong_bloom]
    if bloom_layers:
        early = sum(1 for l in bloom_layers if l < n_layers // 3)
        mid = sum(1 for l in bloom_layers if n_layers // 3 <= l < 2 * n_layers // 3)
        late = sum(1 for l in bloom_layers if l >= 2 * n_layers // 3)
        print(f"\n  Layer distribution: early={early}, mid={mid}, late={late}")
    
    # Clean up model from memory
    del model
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    
    return {
        "model": model_name,
        "label": model_label,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "d_head": d_head,
        "total_heads": n_layers * n_heads,
        "strong_bloom_count": len(strong_bloom),
        "moderate_bloom_count": len(moderate_bloom),
        "strong_bloom_heads": strong_bloom[:15],
        "bloom_layer_distribution": {
            "early": sum(1 for r in strong_bloom if r["layer"] < n_layers // 3),
            "mid": sum(1 for r in strong_bloom if n_layers // 3 <= r["layer"] < 2 * n_layers // 3),
            "late": sum(1 for r in strong_bloom if r["layer"] >= 2 * n_layers // 3),
        },
        "all_results": results,
        "elapsed_seconds": round(elapsed, 1),
    }


# Run all models
all_model_results = {}
for model_name, model_label in MODELS:
    result = analyze_model(model_name, model_label)
    if result:
        all_model_results[model_name] = result
    print()

# ============================================================
# Cross-model comparison
# ============================================================
print("\n" + "="*70)
print("CROSS-MODEL COMPARISON")
print("="*70)

print(f"\n{'Model':>25} {'Layers':>6} {'Heads':>6} {'Total':>6} {'Strong BF':>10} {'Moderate':>9} {'% BF':>6} {'Early/Mid/Late':>15}")
print("-" * 100)

for model_name, model_label in MODELS:
    if model_name not in all_model_results:
        print(f"  {model_label:>23}  FAILED")
        continue
    r = all_model_results[model_name]
    pct_bf = r["strong_bloom_count"] / r["total_heads"] * 100
    dist = r["bloom_layer_distribution"]
    print(f"  {r['label']:>23} {r['n_layers']:>6} {r['n_heads']:>6} {r['total_heads']:>6} "
          f"{r['strong_bloom_count']:>10} {r['moderate_bloom_count']:>9} {pct_bf:>5.1f}% "
          f"{dist['early']:>4}/{dist['mid']:>3}/{dist['late']:>3}")

# Key findings
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

models_with_bloom = [m for m, r in all_model_results.items() if r["strong_bloom_count"] > 0]
print(f"\n  Models with Bloom filter heads: {len(models_with_bloom)}/{len(all_model_results)}")

if len(models_with_bloom) == len(all_model_results):
    print("  🟢 BLOOM FILTER HEADS GENERALIZE ACROSS MODELS")
    print("     This is a universal feature of transformer attention, not a GPT-2 quirk.")
elif len(models_with_bloom) > 1:
    print("  🟡 BLOOM FILTER HEADS FOUND IN MULTIPLE MODELS")
    print("     Strong evidence for generality, but not universal.")
else:
    print("  🔴 BLOOM FILTER HEADS MAY BE MODEL-SPECIFIC")

# Check if bloom heads concentrate in early layers across models
all_early = all([r["bloom_layer_distribution"]["early"] > r["bloom_layer_distribution"]["late"]
                for r in all_model_results.values() if r["strong_bloom_count"] > 0])
if all_early:
    print("  ✅ Early-layer concentration holds across all models")

# Scaling analysis
model_sizes = []
bloom_counts = []
for model_name, _ in MODELS:
    if model_name in all_model_results:
        r = all_model_results[model_name]
        model_sizes.append(r["total_heads"])
        bloom_counts.append(r["strong_bloom_count"])

if len(model_sizes) > 2:
    corr = np.corrcoef(model_sizes, bloom_counts)[0, 1] if np.std(bloom_counts) > 0 else 0
    print(f"\n  Scaling: Correlation between model size and Bloom head count: r={corr:.3f}")
    if corr > 0.5:
        print("  ✅ More Bloom heads in larger models — consistent with distributed membership testing")

# Save results
results_path = "/Users/pabalogh/clawd/projects/bloom-filter-heads/results/multi_model_validation.json"
with open(results_path, "w") as f:
    json.dump(all_model_results, f, indent=2, default=str)
print(f"\nResults saved to {results_path}")
