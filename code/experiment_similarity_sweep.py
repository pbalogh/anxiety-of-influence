#!/usr/bin/env python3
"""
Experiment: Similarity Sweep for Bloom Filter Heads
=====================================================

Runs GPT-2 small on all sentence frames, extracting attention patterns
from all 144 heads to measure attention from the probe/synonym/control
position back to the first occurrence of the target word.

Aggregates by similarity bin to reveal the Bloom-filter response curve.

Output: ../results/similarity_sweep_test.json  (small test run)
        ../results/similarity_sweep_full.json  (full 100-word run)
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

import torch

# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--n-targets", type=int, default=5,
                    help="Number of target words to process (default=5 for test)")
parser.add_argument("--output", type=str, default=None,
                    help="Output file path")
parser.add_argument("--device", type=str, default=None,
                    help="Device (cpu/mps/cuda)")
args = parser.parse_args()

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_JSON = os.path.join(DATA_DIR, "sentence_frames.json")

if args.output:
    OUTPUT_JSON = args.output
elif args.n_targets <= 10:
    OUTPUT_JSON = os.path.join(RESULTS_DIR, "similarity_sweep_test.json")
else:
    OUTPUT_JSON = os.path.join(RESULTS_DIR, "similarity_sweep_full.json")

# Device
if args.device:
    device = args.device
elif torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Device: {device}")

# ──────────────────────────────────────────────
# Load TransformerLens model
# ──────────────────────────────────────────────
print("Loading GPT-2 small with TransformerLens...")
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device=device)
tokenizer = model.tokenizer
n_layers = model.cfg.n_layers   # 12
n_heads = model.cfg.n_heads     # 12
print(f"  Model: {n_layers} layers × {n_heads} heads = {n_layers * n_heads} total heads")

# ──────────────────────────────────────────────
# Load sentence frames
# ──────────────────────────────────────────────
print("Loading sentence frames...")
with open(INPUT_JSON) as f:
    frames_data = json.load(f)

all_frames = frames_data['frames']

# Limit to requested number of targets
if args.n_targets < len(all_frames):
    all_frames = all_frames[:args.n_targets]
    print(f"  Using {args.n_targets} target words (test run)")
else:
    print(f"  Using all {len(all_frames)} target words")

# Known Bloom filter heads (from paper)
BLOOM_HEADS = [(0, 1), (0, 5), (1, 11), (3, 0)]
BLOOM_HEAD_NAMES = ["L0H1", "L0H5", "L1H11", "L3H0"]

# Attention threshold for FP rate calculation
# NOTE: 0.1 is a conservative threshold. The paper uses attention > 0.1
# as "attending to" a position. This can be adjusted.
ATTENTION_THRESHOLD = 0.1

# ──────────────────────────────────────────────
# Run experiment
# ──────────────────────────────────────────────
print("Running experiment...")
start_time = time.time()

all_results = []
per_head_by_level = defaultdict(lambda: defaultdict(list))  # (layer, head) -> sim_level -> [attention_values]

for frame_idx, frame in enumerate(all_frames):
    target_word = frame['target_word']
    print(f"\n  [{frame_idx+1}/{len(all_frames)}] Target: {target_word} ({frame['target_pos']})")
    
    for sent_entry in frame['sentences']:
        sentence = sent_entry['sentence']
        condition = sent_entry['condition']
        slot2_word = sent_entry['slot2_word']
        sim_level = sent_entry['similarity_level']
        cosine_sim = sent_entry.get('cosine_similarity')
        
        # Tokenize
        tokens = tokenizer.encode(sentence, add_special_tokens=False)
        tokens_tensor = torch.tensor([tokens], device=device)
        
        # Find target positions (first occurrence) and probe position (second occurrence/replacement)
        target_token_id = tokenizer.encode(" " + target_word, add_special_tokens=False)
        target_token_id_nospace = tokenizer.encode(target_word, add_special_tokens=False)
        
        # Find first occurrence of target
        first_target_pos = None
        for pos_idx in range(len(tokens)):
            if tokens[pos_idx] in target_token_id or tokens[pos_idx] in target_token_id_nospace:
                first_target_pos = pos_idx
                break
        
        # Find second slot position (where probe/control/repeat goes)
        slot2_token_id = tokenizer.encode(" " + slot2_word, add_special_tokens=False)
        slot2_token_id_nospace = tokenizer.encode(slot2_word, add_special_tokens=False)
        
        second_slot_pos = None
        found_first = False
        for pos_idx in range(len(tokens)):
            is_target = tokens[pos_idx] in target_token_id or tokens[pos_idx] in target_token_id_nospace
            is_slot2 = tokens[pos_idx] in slot2_token_id or tokens[pos_idx] in slot2_token_id_nospace
            
            if condition == 'exact':
                # Both slots are target word; find second occurrence
                if is_target:
                    if found_first:
                        second_slot_pos = pos_idx
                        break
                    found_first = True
            else:
                # First slot is target, second slot is probe/control word
                if is_slot2 and pos_idx != first_target_pos:
                    second_slot_pos = pos_idx
                    break
        
        if first_target_pos is None or second_slot_pos is None:
            # Skip if we can't find positions
            continue
        
        # Forward pass — extract attention patterns
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_tensor, 
                                            names_filter=lambda name: "pattern" in name)
        
        # Extract attention from second_slot_pos -> first_target_pos for all heads
        head_attentions = {}
        for layer in range(n_layers):
            pattern = cache[f"blocks.{layer}.attn.hook_pattern"]  # (batch, heads, seq, seq)
            for head in range(n_heads):
                # Attention from probe position attending to target position
                attn_value = float(pattern[0, head, second_slot_pos, first_target_pos].cpu().item())
                head_attentions[f"L{layer}H{head}"] = attn_value
                
                # Aggregate
                per_head_by_level[(layer, head)][sim_level].append(attn_value)
        
        # Store per-sentence result
        result_entry = {
            'target_word': target_word,
            'slot2_word': slot2_word,
            'condition': condition,
            'similarity_level': sim_level,
            'cosine_similarity': cosine_sim,
            'first_target_pos': first_target_pos,
            'second_slot_pos': second_slot_pos,
            'n_tokens': len(tokens),
            'sentence': sentence,
            'bloom_head_attentions': {
                name: head_attentions[name] for name in BLOOM_HEAD_NAMES
            },
            'all_head_attentions': head_attentions,
        }
        
        if sent_entry.get('wordnet_similarity') is not None:
            result_entry['wordnet_similarity'] = sent_entry['wordnet_similarity']
        
        all_results.append(result_entry)

elapsed = time.time() - start_time
print(f"\n  Processed {len(all_results)} sentences in {elapsed:.1f}s")

# ──────────────────────────────────────────────
# Aggregate results
# ──────────────────────────────────────────────
print("\nAggregating results...")

aggregated = {}
for (layer, head), level_data in per_head_by_level.items():
    head_name = f"L{layer}H{head}"
    aggregated[head_name] = {}
    for sim_level, values in level_data.items():
        vals = np.array(values)
        aggregated[head_name][str(sim_level)] = {
            'mean_attention': round(float(np.mean(vals)), 6),
            'std_attention': round(float(np.std(vals)), 6),
            'median_attention': round(float(np.median(vals)), 6),
            'max_attention': round(float(np.max(vals)), 6),
            'fp_rate': round(float(np.mean(vals > ATTENTION_THRESHOLD)), 4),
            'n_samples': len(values),
        }

# ──────────────────────────────────────────────
# Print Bloom head results
# ──────────────────────────────────────────────
print("\n" + "="*80)
print("BLOOM FILTER HEAD ATTENTION BY SIMILARITY LEVEL")
print(f"(threshold for FP rate: {ATTENTION_THRESHOLD})")
print("="*80)

# Ordered levels for display
display_levels = ['exact', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.0', 'synonym', 'control']

for head_name in BLOOM_HEAD_NAMES:
    print(f"\n  {head_name}:")
    head_data = aggregated.get(head_name, {})
    for level in display_levels:
        if level in head_data:
            d = head_data[level]
            print(f"    sim={level:>8s}: mean_attn={d['mean_attention']:.4f}  "
                  f"std={d['std_attention']:.4f}  "
                  f"FP_rate={d['fp_rate']:.2f}  "
                  f"n={d['n_samples']}")

# ──────────────────────────────────────────────
# Summary across all heads
# ──────────────────────────────────────────────
print("\n" + "="*80)
print("TOP 10 HEADS BY EXACT-REPEAT ATTENTION (mean)")
print("="*80)

head_exact_attention = []
for head_name, levels in aggregated.items():
    if 'exact' in levels:
        head_exact_attention.append((head_name, levels['exact']['mean_attention']))

head_exact_attention.sort(key=lambda x: x[1], reverse=True)
for rank, (head_name, attn) in enumerate(head_exact_attention[:10], 1):
    is_bloom = " ★ BLOOM" if head_name in BLOOM_HEAD_NAMES else ""
    ctrl = aggregated[head_name].get('control', {}).get('mean_attention', 0)
    print(f"  {rank:2d}. {head_name:6s}: exact_attn={attn:.4f}  control_attn={ctrl:.4f}  "
          f"selectivity={attn-ctrl:.4f}{is_bloom}")

# ──────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────
print(f"\nSaving results to {OUTPUT_JSON}...")

output = {
    'config': {
        'n_targets': len(all_frames),
        'n_sentences': len(all_results),
        'device': device,
        'attention_threshold': ATTENTION_THRESHOLD,
        'bloom_heads': BLOOM_HEAD_NAMES,
        'elapsed_seconds': round(elapsed, 1),
    },
    'aggregated_by_head_and_level': aggregated,
    'per_sentence_results': all_results,
}

with open(OUTPUT_JSON, 'w') as f:
    json.dump(output, f, indent=2)

print(f"  Saved {len(all_results)} sentence results")
print("\nDone!")
