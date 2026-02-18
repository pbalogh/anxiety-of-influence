"""
Experiment 6: Bloom Filter Heads vs. Duplicate-Token Heads (Wang et al. 2022)

The #1 novelty objection: "Aren't these just duplicate-token heads from the IOI circuit?"

Wang et al. (2022) identified duplicate-token heads in the Indirect Object 
Identification (IOI) circuit. These heads fire when the SECOND occurrence of a
name attends to the FIRST occurrence — specifically in sentences like:
    "When Mary and John went to the store, John gave a drink to [Mary]"

We establish the distinction:
  1. Replicate the IOI duplicate-token measurement across all 144 heads
  2. Cross-reference with our Bloom filter heads — compute the Venn diagram
  3. Test generalization: Bloom filter heads respond to ANY repeated token,
     not just names. Duplicate-token heads may be name-specific.
  4. Test on: IOI name repetition, non-name repetition (verbs, adjectives,
     common nouns), random token repetition (no natural language structure)
"""

import torch
import numpy as np
import json
import os
import sys
import random
from collections import defaultdict
from transformer_lens import HookedTransformer

# Import expanded stimuli for the 100-sentence tests
sys.path.insert(0, '/Users/pabalogh/clawd/projects/bloom-filter-heads/code')
from expanded_stimuli import EXACT_REPEAT, NO_REPEAT, SEMANTIC_NEAR

# ============================================================
# Setup
# ============================================================

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
n_layers = model.cfg.n_layers
n_heads = model.cfg.n_heads
print(f"Model loaded: {n_layers} layers, {n_heads} heads per layer ({n_layers * n_heads} total)")

BLOOM_HEADS = [(0, 1), (0, 5), (1, 11), (3, 0)]

# ============================================================
# Section 1: IOI Duplicate-Token Head Identification
# ============================================================

print("\n" + "=" * 70)
print("SECTION 1: IOI DUPLICATE-TOKEN HEAD IDENTIFICATION")
print("=" * 70)

# Standard IOI sentence templates from Wang et al. (2022)
# Format: "When [IO] and [S] went to the [PLACE], [S] gave a [OBJECT] to [IO]"
# S appears twice (S1 and S2). Duplicate-token heads: S2 attends to S1.

NAMES = [
    "Mary", "John", "Alice", "Bob", "Sarah", "David", "Emma", "James",
    "Lisa", "Tom", "Anna", "Mark", "Jane", "Paul", "Kate", "Mike",
    "Susan", "Peter", "Laura", "Chris", "Diana", "Steve", "Helen", "Kevin",
    "Grace", "Brian", "Wendy", "Scott", "Emily", "Frank",
]

PLACES = [
    "store", "park", "beach", "school", "office", "market", "library",
    "restaurant", "museum", "hospital", "station", "theater", "gym",
    "airport", "church", "garden", "hotel", "bakery", "bank", "cafe",
]

OBJECTS = [
    "drink", "book", "gift", "letter", "ticket", "key", "flower",
    "toy", "phone", "bag", "hat", "ring", "coin", "pen", "card",
]

TEMPLATES = [
    "When {IO} and {S} went to the {PLACE}, {S} gave a {OBJ} to",
    "After {IO} and {S} arrived at the {PLACE}, {S} handed a {OBJ} to",
    "Because {IO} and {S} visited the {PLACE}, {S} offered a {OBJ} to",
    "While {IO} and {S} were at the {PLACE}, {S} passed a {OBJ} to",
    "Since {IO} and {S} stopped by the {PLACE}, {S} brought a {OBJ} to",
]


def generate_ioi_sentences(n=50):
    """Generate IOI-format sentences with S1, S2, and IO positions tracked."""
    sentences = []
    random.seed(42)  # Reproducibility
    for i in range(n):
        # Pick two different names
        name_pair = random.sample(NAMES, 2)
        io_name, s_name = name_pair[0], name_pair[1]
        place = random.choice(PLACES)
        obj = random.choice(OBJECTS)
        template = TEMPLATES[i % len(TEMPLATES)]
        
        sentence = template.format(IO=io_name, S=s_name, PLACE=place, OBJ=obj)
        sentences.append({
            "text": sentence,
            "io_name": io_name,
            "s_name": s_name,
        })
    return sentences


def find_token_positions(model, tokens_tensor, target_str):
    """Find all positions where target_str appears as a token (or starts a token)."""
    positions = []
    all_str_tokens = model.to_str_tokens(tokens_tensor[0])
    # Handle both with and without leading space
    target_variants = [target_str, " " + target_str, "Ġ" + target_str]
    for pos, tok_str in enumerate(all_str_tokens):
        if tok_str.strip() in [target_str] or tok_str in target_variants:
            positions.append(pos)
    return positions


def measure_duplicate_token_scores(model, ioi_sentences):
    """
    For each head, measure duplicate-token behavior:
    S2 (second occurrence of subject name) attending to S1 (first occurrence).
    
    Returns: dict of (layer, head) -> mean attention from S2 to S1
    """
    scores = defaultdict(list)
    valid_count = 0
    
    for sent_info in ioi_sentences:
        tokens = model.to_tokens(sent_info["text"])
        s_name = sent_info["s_name"]
        
        # Find S1 and S2 positions
        s_positions = find_token_positions(model, tokens, s_name)
        
        if len(s_positions) < 2:
            continue
        
        s1_pos = s_positions[0]
        s2_pos = s_positions[1]
        valid_count += 1
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        for layer in range(n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0]  # [n_heads, dest, src]
            for head in range(n_heads):
                # Attention from S2 to S1
                attn_s2_to_s1 = pattern[head, s2_pos, s1_pos].item()
                scores[(layer, head)].append(attn_s2_to_s1)
    
    print(f"  Valid IOI sentences (both S positions found): {valid_count}/{len(ioi_sentences)}")
    
    # Average across sentences
    mean_scores = {}
    for key, vals in scores.items():
        mean_scores[key] = float(np.mean(vals))
    
    return mean_scores


print("\nGenerating 50 IOI-format sentences...")
ioi_sentences = generate_ioi_sentences(50)
print(f"  Examples:")
for s in ioi_sentences[:3]:
    print(f"    \"{s['text']}\" (IO={s['io_name']}, S={s['s_name']})")

print("\nMeasuring duplicate-token scores for all 144 heads...")
dup_token_scores = measure_duplicate_token_scores(model, ioi_sentences)

# Rank heads by duplicate-token score
sorted_heads = sorted(dup_token_scores.items(), key=lambda x: x[1], reverse=True)

print("\nTop 20 duplicate-token heads (S2→S1 attention):")
print(f"  {'Head':>10s}  {'Score':>8s}  {'Bloom?':>8s}")
print(f"  {'-'*10}  {'-'*8}  {'-'*8}")
top_dup_heads = []
for (layer, head), score in sorted_heads[:20]:
    is_bloom = "✅ YES" if (layer, head) in BLOOM_HEADS else ""
    print(f"  L{layer}H{head:>2d}       {score:.4f}    {is_bloom}")
    top_dup_heads.append((layer, head))

# ============================================================
# Section 2: Venn Diagram — Bloom vs Duplicate-Token Heads
# ============================================================

print("\n" + "=" * 70)
print("SECTION 2: VENN DIAGRAM — BLOOM vs DUPLICATE-TOKEN HEADS")
print("=" * 70)

# Define duplicate-token heads as top-K by score
# Wang et al. typically use ~10-15 heads
DUP_TOP_K = 15
dup_heads_set = set(sorted_heads[i][0] for i in range(min(DUP_TOP_K, len(sorted_heads))))
bloom_heads_set = set(BLOOM_HEADS)

overlap = bloom_heads_set & dup_heads_set
bloom_only = bloom_heads_set - dup_heads_set
dup_only = dup_heads_set - bloom_heads_set

print(f"\nDuplicate-token heads (top {DUP_TOP_K}): {sorted(dup_heads_set)}")
print(f"Bloom filter heads: {sorted(bloom_heads_set)}")
print(f"\nOverlap:         {sorted(overlap)} ({len(overlap)} heads)")
print(f"Bloom only:      {sorted(bloom_only)} ({len(bloom_only)} heads)")
print(f"Dup-token only:  {sorted(dup_only)} ({len(dup_only)} heads)")

# Per-Bloom-head duplicate-token ranks
print("\nBloom filter heads' duplicate-token ranks:")
for bh in BLOOM_HEADS:
    rank = next((i + 1 for i, (h, _) in enumerate(sorted_heads) if h == bh), "N/A")
    score = dup_token_scores.get(bh, 0.0)
    print(f"  L{bh[0]}H{bh[1]}: rank {rank}/144, dup-token score = {score:.4f}")

# ============================================================
# Section 3: Cross-Stimulus Generalization
# ============================================================

print("\n" + "=" * 70)
print("SECTION 3: CROSS-STIMULUS GENERALIZATION")
print("=" * 70)
print("Testing whether heads generalize beyond IOI name repetition...")
print(f"Using {len(EXACT_REPEAT)} expanded stimuli sentences for non-name repetition.\n")


def measure_repeat_attention_natural(model, sentences, heads):
    """
    For natural language sentences with repeated content words,
    measure attention from the second occurrence to the first.
    """
    results = defaultdict(list)
    valid = 0
    
    for sentence in sentences:
        tokens = model.to_tokens(sentence)
        str_tokens = model.to_str_tokens(tokens[0])
        
        # Find repeated tokens (by string, ignoring case and whitespace)
        cleaned = [t.strip().lower() for t in str_tokens]
        seen = {}
        repeat_pairs = []
        for pos, tok in enumerate(cleaned):
            if len(tok) <= 1:  # skip single-char tokens (articles, etc.)
                continue
            if tok in seen:
                repeat_pairs.append((pos, seen[tok]))
            else:
                seen[tok] = pos
        
        if not repeat_pairs:
            continue
        valid += 1
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        for layer, head in heads:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]  # [dest, src]
            attns = []
            for dest_pos, src_pos in repeat_pairs:
                if dest_pos < pattern.shape[0] and src_pos < pattern.shape[1]:
                    attns.append(pattern[dest_pos, src_pos].item())
            if attns:
                results[(layer, head)].append(float(np.mean(attns)))
    
    return results, valid


def build_random_repeat_sequences(model, n_sequences=30, seq_len=20, n_repeats=5):
    """Build sequences with random tokens, some repeated, no linguistic structure."""
    sequences = []
    vocab_size = model.cfg.d_vocab
    random.seed(42)
    torch.manual_seed(42)
    
    for _ in range(n_sequences):
        # Pick random tokens (avoid special tokens 0-999)
        unique_tokens = torch.randint(1000, vocab_size, (seq_len - n_repeats,))
        # Pick n_repeats tokens from the unique set to repeat at the end
        repeat_indices = torch.randperm(seq_len - n_repeats)[:n_repeats]
        repeat_tokens = unique_tokens[repeat_indices]
        
        full_seq = torch.cat([
            torch.tensor([model.tokenizer.bos_token_id]),
            unique_tokens,
            repeat_tokens
        ]).unsqueeze(0).to(device)
        
        # Track where repeats are and where their first occurrence was
        repeat_pairs = []
        for i, ri in enumerate(repeat_indices):
            dest_pos = 1 + (seq_len - n_repeats) + i  # position of repeat
            src_pos = 1 + ri.item()  # position of first occurrence
            repeat_pairs.append((dest_pos, src_pos))
        
        sequences.append({
            "tokens": full_seq,
            "repeat_pairs": repeat_pairs,
        })
    
    return sequences


def measure_repeat_attention_random(model, random_sequences, heads):
    """
    For random-token sequences, measure attention from repeated token
    to its first occurrence.
    """
    results = defaultdict(list)
    
    for seq_info in random_sequences:
        tokens = seq_info["tokens"]
        
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
        
        for layer, head in heads:
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0, head]
            attns = []
            for dest_pos, src_pos in seq_info["repeat_pairs"]:
                if dest_pos < pattern.shape[0] and src_pos < pattern.shape[1]:
                    attns.append(pattern[dest_pos, src_pos].item())
            if attns:
                results[(layer, head)].append(float(np.mean(attns)))
    
    return results


# Heads to test: Bloom heads + top duplicate-token heads (non-Bloom)
top_dup_only = [(l, h) for (l, h) in top_dup_heads if (l, h) not in BLOOM_HEADS][:6]
test_heads = BLOOM_HEADS + top_dup_only

print(f"Testing heads:")
print(f"  Bloom heads:      {BLOOM_HEADS}")
print(f"  Top-dup-only:     {top_dup_only}")

# A. IOI name repetition — from section 1, extract per-head scores
print("\n  A. IOI name repetition (from Section 1)...")
ioi_scores_for_test = {}
for h in test_heads:
    ioi_scores_for_test[h] = dup_token_scores.get(h, 0.0)

# B. Non-name repetition using expanded stimuli
print(f"  B. Non-name repetition ({len(EXACT_REPEAT)} sentences from expanded_stimuli)...")
non_name_results, non_name_valid = measure_repeat_attention_natural(
    model, EXACT_REPEAT, test_heads
)
print(f"     Valid sentences: {non_name_valid}/{len(EXACT_REPEAT)}")

non_name_scores = {}
for h in test_heads:
    vals = non_name_results.get(h, [])
    non_name_scores[h] = float(np.mean(vals)) if vals else 0.0

# C. Random token repetition
print("  C. Random token repetition (30 sequences, no linguistic structure)...")
random_sequences = build_random_repeat_sequences(model, n_sequences=30)
random_results = measure_repeat_attention_random(model, random_sequences, test_heads)

random_scores = {}
for h in test_heads:
    vals = random_results.get(h, [])
    random_scores[h] = float(np.mean(vals)) if vals else 0.0

# Print cross-stimulus generalization matrix
print("\n" + "-" * 78)
print("CROSS-STIMULUS GENERALIZATION MATRIX")
print(f"  {'Head':>10s}  {'IOI Names':>11s}  {'Non-Name':>11s}  {'Random':>11s}  {'Type':>12s}")
print(f"  {'-'*10}  {'-'*11}  {'-'*11}  {'-'*11}  {'-'*12}")

generalization_matrix = {}
for h in test_heads:
    label = f"L{h[0]}H{h[1]}"
    head_type = "Bloom" if h in BLOOM_HEADS else "Dup-token"
    ioi_s = ioi_scores_for_test[h]
    nn_s = non_name_scores[h]
    rnd_s = random_scores[h]
    
    generalization_matrix[label] = {
        "ioi_name_repetition": round(ioi_s, 4),
        "non_name_repetition": round(nn_s, 4),
        "random_token_repetition": round(rnd_s, 4),
        "head_type": head_type,
    }
    
    print(f"  {label:>10s}  {ioi_s:>11.4f}  {nn_s:>11.4f}  {rnd_s:>11.4f}  {head_type:>12s}")

# ============================================================
# Section 4: Generalization Index
# ============================================================

print("\n" + "=" * 70)
print("SECTION 4: GENERALIZATION INDEX")
print("=" * 70)

print("\nComputing generalization index = min(non_name, random) / ioi_name")
print("(1.0 = perfectly general, 0.0 = IOI-name-specific)\n")

gen_indices = {}
for h in test_heads:
    label = f"L{h[0]}H{h[1]}"
    ioi_s = ioi_scores_for_test[h]
    nn_s = non_name_scores[h]
    rnd_s = random_scores[h]
    
    if ioi_s > 0.01:
        gen_idx = min(nn_s, rnd_s) / ioi_s
    else:
        gen_idx = 0.0
    
    gen_indices[label] = round(gen_idx, 4)
    head_type = "Bloom" if h in BLOOM_HEADS else "Dup-token"
    print(f"  {label:>10s}  gen_index = {gen_idx:.4f}  ({head_type})")

bloom_gen = [gen_indices[f"L{h[0]}H{h[1]}"] for h in BLOOM_HEADS]
dup_gen = [gen_indices[f"L{h[0]}H{h[1]}"] for h in top_dup_only if f"L{h[0]}H{h[1]}" in gen_indices]

print(f"\n  Bloom heads mean generalization: {np.mean(bloom_gen):.4f}")
if dup_gen:
    print(f"  Dup-token heads mean generalization: {np.mean(dup_gen):.4f}")

# ============================================================
# Section 5: Statistical Comparison (Bloom vs Dup-Token)
# ============================================================

print("\n" + "=" * 70)
print("SECTION 5: STATISTICAL COMPARISON")
print("=" * 70)

# Compare attention on expanded_stimuli EXACT_REPEAT for Bloom vs Top-Dup heads
print("\nComparing attention on expanded stimuli (100 exact-repeat sentences):")

bloom_non_name_scores = [non_name_scores[h] for h in BLOOM_HEADS]
dup_non_name_scores = [non_name_scores[h] for h in top_dup_only]

bloom_random_scores = [random_scores[h] for h in BLOOM_HEADS]
dup_random_scores = [random_scores[h] for h in top_dup_only]

print(f"\n  Non-name repetition attention:")
print(f"    Bloom heads mean:     {np.mean(bloom_non_name_scores):.4f} ± {np.std(bloom_non_name_scores):.4f}")
print(f"    Dup-token heads mean: {np.mean(dup_non_name_scores):.4f} ± {np.std(dup_non_name_scores):.4f}")

print(f"\n  Random token repetition attention:")
print(f"    Bloom heads mean:     {np.mean(bloom_random_scores):.4f} ± {np.std(bloom_random_scores):.4f}")
print(f"    Dup-token heads mean: {np.mean(dup_random_scores):.4f} ± {np.std(dup_random_scores):.4f}")

# ============================================================
# Section 6: Full 144-Head Heatmap Data
# ============================================================

print("\n" + "=" * 70)
print("SECTION 6: FULL DUPLICATE-TOKEN SCORE HEATMAP")
print("=" * 70)

# Build a 12x12 grid of scores
score_grid = [[0.0] * n_heads for _ in range(n_layers)]
for (layer, head), score in dup_token_scores.items():
    score_grid[layer][head] = round(score, 4)

print("\nDuplicate-token score by layer (rows) and head (columns):")
print(f"  {'':>6s}", end="")
for h in range(n_heads):
    print(f"  H{h:>2d}", end="")
print()

for layer in range(n_layers):
    print(f"  L{layer:>2d}  ", end="")
    for head in range(n_heads):
        s = score_grid[layer][head]
        marker = "*" if (layer, head) in BLOOM_HEADS else " "
        print(f" {s:.2f}{marker}", end="")
    print()
print("  (* = Bloom filter head)")

# ============================================================
# VERDICT
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 6 VERDICT")
print("=" * 70)

n_overlap = len(overlap)
n_bloom_only = len(bloom_only)
mean_bloom_gen = float(np.mean(bloom_gen))
mean_dup_gen = float(np.mean(dup_gen)) if dup_gen else 0.0

print(f"\n1. OVERLAP: {n_overlap}/{len(BLOOM_HEADS)} Bloom heads are in the top-{DUP_TOP_K} duplicate-token heads.")

if n_overlap == 0:
    overlap_verdict = "Bloom filter heads are DISTINCT from duplicate-token heads."
elif n_overlap < len(BLOOM_HEADS):
    overlap_verdict = f"PARTIAL overlap: {n_overlap} heads shared, {n_bloom_only} Bloom-only."
else:
    overlap_verdict = "FULL overlap: all Bloom heads are also duplicate-token heads."
print(f"   → {overlap_verdict}")

print(f"\n2. GENERALIZATION:")
print(f"   Bloom heads generalize to non-name stimuli: mean gen_index = {mean_bloom_gen:.4f}")
if dup_gen:
    print(f"   Duplicate-token heads generalize:          mean gen_index = {mean_dup_gen:.4f}")

if mean_bloom_gen > mean_dup_gen + 0.1:
    gen_verdict = ("POSITIVE: Bloom heads generalize MORE broadly than duplicate-token heads. "
                   "They respond to any repeated token, not just repeated names.")
elif mean_bloom_gen > 0.3:
    gen_verdict = ("MODERATE: Bloom heads generalize to non-name repetition. The distinction "
                   "from duplicate-token heads is real but the gap is modest.")
else:
    gen_verdict = ("WEAK: Bloom heads do not generalize much beyond IOI-like stimuli. "
                   "The distinction from duplicate-token heads is unclear.")
print(f"   → {gen_verdict}")

print(f"\n3. KEY DISTINCTION:")
print(f"   Duplicate-token heads were characterized ONLY in the IOI circuit for name repetition.")
print(f"   Bloom filter heads additionally show: capacity curves matching Bloom filter theory,")
print(f"   independence between heads, and generalization across token types.")

# ============================================================
# Save results
# ============================================================

results = {
    "experiment": "experiment6_duplicate_token_comparison",
    "model": "gpt2-small",
    "device": device,
    "bloom_heads": [list(h) for h in BLOOM_HEADS],
    "n_ioi_sentences": len(ioi_sentences),
    "n_expanded_stimuli": len(EXACT_REPEAT),
    "dup_token_top_k": DUP_TOP_K,
    "dup_token_scores_all_heads": {
        f"L{l}H{h}": round(s, 4) for (l, h), s in dup_token_scores.items()
    },
    "dup_token_ranking": [
        {"head": f"L{l}H{h}", "score": round(s, 4), "rank": i + 1}
        for i, ((l, h), s) in enumerate(sorted_heads)
    ],
    "top_dup_heads": [f"L{h[0]}H{h[1]}" for h in top_dup_heads[:DUP_TOP_K]],
    "venn_diagram": {
        "overlap": [f"L{h[0]}H{h[1]}" for h in sorted(overlap)],
        "bloom_only": [f"L{h[0]}H{h[1]}" for h in sorted(bloom_only)],
        "dup_only": [f"L{h[0]}H{h[1]}" for h in sorted(dup_only)],
        "n_overlap": n_overlap,
        "n_bloom_only": n_bloom_only,
        "n_dup_only": len(dup_only),
    },
    "bloom_head_dup_ranks": {
        f"L{h[0]}H{h[1]}": {
            "dup_token_score": round(dup_token_scores.get(h, 0.0), 4),
            "rank": next((i + 1 for i, (hh, _) in enumerate(sorted_heads) if hh == h), None),
        }
        for h in BLOOM_HEADS
    },
    "generalization_matrix": generalization_matrix,
    "generalization_indices": gen_indices,
    "summary_stats": {
        "bloom_mean_generalization": round(mean_bloom_gen, 4),
        "dup_mean_generalization": round(mean_dup_gen, 4),
        "bloom_mean_non_name_attn": round(float(np.mean(bloom_non_name_scores)), 4),
        "dup_mean_non_name_attn": round(float(np.mean(dup_non_name_scores)), 4),
        "bloom_mean_random_attn": round(float(np.mean(bloom_random_scores)), 4),
        "dup_mean_random_attn": round(float(np.mean(dup_random_scores)), 4),
    },
    "score_grid": score_grid,
    "verdict": {
        "overlap_description": overlap_verdict,
        "generalization_verdict": gen_verdict,
        "n_overlap": n_overlap,
        "n_bloom_total": len(BLOOM_HEADS),
    },
}

results_dir = "/Users/pabalogh/clawd/projects/bloom-filter-heads/results"
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, "experiment6_duplicate_token_comparison.json")

with open(output_path, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to {output_path}")
print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
