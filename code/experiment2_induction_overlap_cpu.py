"""
Experiment 2: Are induction heads a special case of Bloom filter heads?

Induction heads implement the pattern: A B ... A → B
They work by:
  1. A "previous token head" attends from B to A (one layer earlier)
  2. An "induction head" matches current A to previous A, then copies B

The QK circuit of induction heads should show Bloom-filter behavior:
  - It's asking "have I seen this query token before?" — a membership test.

We test:
  1. Which heads are induction heads (using the standard A B ... A B test)
  2. Whether induction heads overlap with our Bloom filter heads
  3. Whether non-induction Bloom filter heads exist (membership without copying)
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from collections import defaultdict
import json

device = "cpu"  # FORCED CPU
print(f"Using device: {device}")

print("Loading GPT-2 small...")
model = HookedTransformer.from_pretrained("gpt2-small", device=device)
print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads")

# ============================================================
# Test 1: Standard induction head detection
# ============================================================

def detect_induction_heads(model, n_trials=50):
    """
    Standard induction head test: random token sequences repeated.
    [BOS] A B C D E ... A B C D E ...
    Induction heads attend from second A to first B (shifted by 1).
    """
    print("\nDetecting induction heads with random repeated sequences...")
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_half = 25  # half-sequence length
    
    # Accumulate per-head induction scores
    induction_scores = torch.zeros(n_layers, n_heads)
    
    for trial in range(n_trials):
        # Random token sequence, repeated
        rand_tokens = torch.randint(1000, 10000, (seq_half,))
        # [BOS] + seq + seq
        tokens = torch.cat([
            torch.tensor([model.tokenizer.bos_token_id]),
            rand_tokens,
            rand_tokens
        ]).unsqueeze(0).to(device)
        
        _, cache = model.run_with_cache(tokens)
        
        for layer in range(n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0].cpu()  # [head, dest, src]
            
            for head in range(n_heads):
                # For positions in the SECOND half, check if they attend to 
                # the position ONE AFTER the corresponding position in the FIRST half
                # i.e., position (1 + seq_half + i) should attend to position (1 + i + 1)
                # which is the "induction" pattern: A...A → attend to token after previous A
                
                score = 0.0
                count = 0
                for i in range(seq_half - 1):
                    dest = 1 + seq_half + i      # position in second half
                    src = 1 + i + 1              # position after match in first half
                    if dest < pattern.shape[1] and src < pattern.shape[2]:
                        score += pattern[head, dest, src].item()
                        count += 1
                
                if count > 0:
                    induction_scores[layer, head] += score / count
    
    induction_scores /= n_trials
    return induction_scores


def detect_previous_token_heads(model, n_trials=50):
    """
    Previous token heads attend from position i to position i-1.
    """
    print("Detecting previous-token heads...")
    
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    seq_len = 50
    
    prev_token_scores = torch.zeros(n_layers, n_heads)
    
    for trial in range(n_trials):
        tokens = torch.randint(1000, 10000, (1, seq_len + 1)).to(device)
        tokens[0, 0] = model.tokenizer.bos_token_id
        
        _, cache = model.run_with_cache(tokens)
        
        for layer in range(n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"][0].cpu()
            
            for head in range(n_heads):
                # Average attention to previous token (diagonal -1)
                score = 0.0
                count = 0
                for pos in range(2, seq_len + 1):
                    score += pattern[head, pos, pos - 1].item()
                    count += 1
                
                if count > 0:
                    prev_token_scores[layer, head] += score / count
    
    prev_token_scores /= n_trials
    return prev_token_scores


# Run detection
induction_scores = detect_induction_heads(model)
prev_token_scores = detect_previous_token_heads(model)

# Load Bloom filter results from Experiment 1
bloom_results_path = "/Users/peter/clawd/projects/bloom-filter-heads/results/experiment1_results.json"
with open(bloom_results_path) as f:
    bloom_data = json.load(f)

bloom_scores = {}
for r in bloom_data["results"]:
    bloom_scores[(r["layer"], r["head"])] = r

# ============================================================
# Analysis: Overlap between induction heads and Bloom filter heads
# ============================================================

print("\n" + "="*60)
print("INDUCTION HEAD SCORES (top 20)")
print("="*60)

all_scores = []
for layer in range(model.cfg.n_layers):
    for head in range(model.cfg.n_heads):
        key = (layer, head)
        bloom = bloom_scores.get(key, {})
        all_scores.append({
            "layer": layer,
            "head": head,
            "induction_score": round(induction_scores[layer, head].item(), 4),
            "prev_token_score": round(prev_token_scores[layer, head].item(), 4),
            "bloom_score": bloom.get("bloom_score", 0),
            "selectivity": bloom.get("selectivity", 0),
            "fp_ratio": bloom.get("fp_ratio", 0),
            "mean_hit_attention": bloom.get("mean_hit_attention", 0),
        })

# Sort by induction score
all_scores.sort(key=lambda x: x["induction_score"], reverse=True)

print(f"{'Layer':>5} {'Head':>4} {'Induction':>10} {'PrevTok':>8} {'Bloom':>8} {'Select.':>8} {'FP Ratio':>9} {'Classification':>20}")
print("-" * 95)

for s in all_scores[:20]:
    # Classify
    is_induction = s["induction_score"] > 0.3
    is_bloom = s["selectivity"] > 3 and s["mean_hit_attention"] > 0.05
    is_prev_tok = s["prev_token_score"] > 0.3
    
    if is_induction and is_bloom:
        cls = "INDUCTION+BLOOM ⭐"
    elif is_induction:
        cls = "INDUCTION"
    elif is_bloom:
        cls = "BLOOM ONLY"
    elif is_prev_tok:
        cls = "PREV TOKEN"
    else:
        cls = ""
    
    bloom_val = s["bloom_score"] if not np.isnan(s["bloom_score"]) else 0
    print(f"  L{s['layer']:>2}   H{s['head']:>2}   {s['induction_score']:>8.4f}   {s['prev_token_score']:>6.4f}   {bloom_val:>7.2f}   {s['selectivity']:>6.1f}x   {s['fp_ratio']:>7.4f}   {cls}")

# ============================================================
# Previous token heads
# ============================================================
print("\n" + "="*60)
print("PREVIOUS TOKEN HEADS (top 10)")
print("="*60)

all_scores.sort(key=lambda x: x["prev_token_score"], reverse=True)
for s in all_scores[:10]:
    is_bloom = s["selectivity"] > 3 and s["mean_hit_attention"] > 0.05
    cls = "PREV+BLOOM ⭐" if is_bloom else "PREV TOKEN"
    bloom_val = s["bloom_score"] if not np.isnan(s["bloom_score"]) else 0
    print(f"  L{s['layer']:>2}   H{s['head']:>2}   prev={s['prev_token_score']:>6.4f}   ind={s['induction_score']:>6.4f}   bloom={bloom_val:>7.2f}   sel={s['selectivity']:>5.1f}x   {cls}")

# ============================================================
# Key question: Are Bloom filter heads a SUPERSET of induction heads?
# ============================================================
print("\n" + "="*60)
print("OVERLAP ANALYSIS")
print("="*60)

induction_heads = [(s["layer"], s["head"]) for s in all_scores if s["induction_score"] > 0.3]
bloom_heads = [(s["layer"], s["head"]) for s in all_scores if s["selectivity"] > 3 and s["mean_hit_attention"] > 0.05]
prev_token_heads = [(s["layer"], s["head"]) for s in all_scores if s["prev_token_score"] > 0.3]

overlap_ind_bloom = set(induction_heads) & set(bloom_heads)
bloom_only = set(bloom_heads) - set(induction_heads)
induction_only = set(induction_heads) - set(bloom_heads)

print(f"\nInduction heads found: {len(induction_heads)}")
for h in induction_heads:
    print(f"  L{h[0]} H{h[1]}")

print(f"\nBloom filter heads found: {len(bloom_heads)}")
for h in bloom_heads:
    print(f"  L{h[0]} H{h[1]}")

print(f"\nPrevious token heads found: {len(prev_token_heads)}")
for h in prev_token_heads:
    print(f"  L{h[0]} H{h[1]}")

print(f"\nOverlap (Induction ∩ Bloom): {len(overlap_ind_bloom)}")
for h in overlap_ind_bloom:
    print(f"  L{h[0]} H{h[1]}")

print(f"\nBloom-only (not induction): {len(bloom_only)}")
for h in bloom_only:
    s = [x for x in all_scores if x["layer"] == h[0] and x["head"] == h[1]][0]
    print(f"  L{h[0]} H{h[1]} — selectivity={s['selectivity']:.1f}x, FP ratio={s['fp_ratio']:.2f}")
    print(f"    → This head does membership testing but NOT pattern completion")

print(f"\nInduction-only (not Bloom): {len(induction_only)}")
for h in induction_only:
    s = [x for x in all_scores if x["layer"] == h[0] and x["head"] == h[1]][0]
    print(f"  L{h[0]} H{h[1]} — induction={s['induction_score']:.4f}, selectivity={s['selectivity']:.1f}x")

# ============================================================
# VERDICT
# ============================================================
print("\n" + "="*60)
print("VERDICT")
print("="*60)

if overlap_ind_bloom:
    print("✅ Some induction heads ARE Bloom filter heads (as predicted)")
if bloom_only:
    print("✅ Bloom-only heads exist — membership testing WITHOUT pattern completion")
    print("   This is the novel finding: a new functional category of attention heads")
if induction_only:
    print("ℹ️  Some induction heads don't show Bloom behavior in our test")
    print("   (May use different membership strategies, or our test isn't capturing them)")

# ============================================================
# Optimization implications
# ============================================================
print("\n" + "="*60)
print("OPTIMIZATION IMPLICATIONS")
print("="*60)

print("""
If attention heads implement Bloom filters, several optimizations become possible:

1. SPARSE ATTENTION PRUNING
   Bloom filter heads only need to check membership — they don't need 
   full quadratic attention. Replace with actual Bloom filters or hash 
   tables for O(1) lookup instead of O(n) attention.
   
2. EARLY EXIT FOR KNOWN CONTEXT
   If Bloom filter heads confirm "all referenced entities are in context,"
   later layers can skip redundant context-checking. This enables 
   input-dependent early exit — shorter inference for simpler inputs.

3. KV-CACHE OPTIMIZATION  
   Bloom filter heads need KEYS but not VALUES from earlier positions.
   Their KV-cache entries could be compressed to key-only, halving memory
   for those heads during long-context inference.

4. ATTENTION HEAD PRUNING
   If multiple Bloom filter heads check the same membership (like 
   multiple hash functions), we might only need a subset. Reducing from
   4 Bloom heads to 2 could save compute with minimal accuracy loss.

5. HALLUCINATION DETECTION
   If Bloom filter heads show FALSE POSITIVE activation (high attention 
   to a position where the referenced token DOESN'T actually appear),
   that's a detectable signal that the model is about to hallucinate —
   it "thinks" something is in context when it isn't.
""")

# Save results
results = {
    "induction_heads": [{"layer": h[0], "head": h[1]} for h in induction_heads],
    "bloom_heads": [{"layer": h[0], "head": h[1]} for h in bloom_heads],
    "prev_token_heads": [{"layer": h[0], "head": h[1]} for h in prev_token_heads],
    "overlap": [{"layer": h[0], "head": h[1]} for h in overlap_ind_bloom],
    "bloom_only": [{"layer": h[0], "head": h[1]} for h in bloom_only],
    "induction_only": [{"layer": h[0], "head": h[1]} for h in induction_only],
    "all_scores": all_scores[:30],  # top 30
}

results_path = "/Users/peter/clawd/projects/bloom-filter-heads/results/experiment2_induction_overlap.json"
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {results_path}")
