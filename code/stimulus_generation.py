#!/usr/bin/env python3
"""
Stimulus Generation for Bloom Filter Heads Paper
=================================================

Selects 100 target content words (stratified by POS and frequency),
finds probe words at 10 cosine-similarity levels in GPT-2 embedding space,
and identifies WordNet synonyms.

Output: ../data/target_probe_matrix.csv
"""

import os
import sys
import csv
import json
import random
import numpy as np
import torch
from collections import defaultdict

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from transformers import GPT2Tokenizer, GPT2Model
from nltk.corpus import wordnet as wn
from wordfreq import zipf_frequency

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, "target_probe_matrix.csv")
OUTPUT_META = os.path.join(DATA_DIR, "target_words_meta.json")

# ──────────────────────────────────────────────
# Step 1: Load tokenizer and embedding matrix
# ──────────────────────────────────────────────
print("Step 1: Loading GPT-2 tokenizer and embeddings...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2Model.from_pretrained("gpt2")
embedding_matrix = model.wte.weight.detach().clone()  # (50257, 768)
del model  # Free memory
print(f"  Embedding matrix shape: {embedding_matrix.shape}")

# ──────────────────────────────────────────────
# Step 2: Build vocabulary of valid single-token English words
# ──────────────────────────────────────────────
print("Step 2: Building valid single-token English word list...")

valid_tokens = {}  # word -> token_id

for token_id in range(tokenizer.vocab_size):
    raw_token = tokenizer.convert_ids_to_tokens(token_id)
    
    # Must start with Ġ (space prefix) to be a standalone word
    if not raw_token.startswith("Ġ"):
        continue
    word = raw_token[1:]
    
    # Must be purely alphabetic, lowercase, at least 4 chars
    if not word.isalpha() or not word.islower() or len(word) < 4:
        continue
    
    # Must have reasonable frequency (zipf >= 2.0 means roughly top 50k)
    freq = zipf_frequency(word, 'en')
    if freq < 2.0:
        continue
    
    # Must be in WordNet
    if not wn.synsets(word):
        continue
    
    # Verify single-token encoding
    encoded = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(encoded) == 1 and encoded[0] == token_id:
        valid_tokens[word] = token_id

print(f"  Found {len(valid_tokens)} valid single-token English words")

# ──────────────────────────────────────────────
# Step 3: Classify by POS and frequency
# ──────────────────────────────────────────────
print("Step 3: Classifying words by POS and frequency...")

def get_primary_pos(word):
    """Get the primary POS from WordNet."""
    synsets = wn.synsets(word)
    if not synsets:
        return None
    pos_counts = defaultdict(int)
    for s in synsets:
        pos_counts[s.pos()] += 1
    wn_to_cat = {'n': 'noun', 'v': 'verb', 'a': 'adj', 's': 'adj', 'r': 'adv'}
    best_pos = max(pos_counts, key=pos_counts.get)
    return wn_to_cat.get(best_pos)

word_info = {}
for word, token_id in valid_tokens.items():
    pos = get_primary_pos(word)
    if pos not in ('noun', 'verb', 'adj'):
        continue
    freq = zipf_frequency(word, 'en')
    word_info[word] = {
        'token_id': token_id,
        'pos': pos,
        'zipf_freq': freq,
    }

# Sort by frequency to assign ranks
all_words_by_freq = sorted(word_info.keys(), key=lambda w: word_info[w]['zipf_freq'], reverse=True)
for rank, word in enumerate(all_words_by_freq):
    word_info[word]['freq_rank'] = rank + 1

def get_freq_bin(word):
    z = word_info[word]['zipf_freq']
    if z >= 4.5:
        return 'high'
    elif z >= 3.0:
        return 'mid'
    else:
        return 'low'

# Organize into bins
bins = defaultdict(list)
for word in word_info:
    pos = word_info[word]['pos']
    fb = get_freq_bin(word)
    bins[(pos, fb)].append(word)

print("  Word counts by (POS, frequency):")
for key in sorted(bins.keys()):
    print(f"    {key}: {len(bins[key])}")

# ──────────────────────────────────────────────
# Step 4: Select 100 target words with stratification
# ──────────────────────────────────────────────
print("Step 4: Selecting 100 stratified target words...")

targets_needed = {
    ('noun', 'high'): 17, ('noun', 'mid'): 17, ('noun', 'low'): 16,
    ('verb', 'high'): 8,  ('verb', 'mid'): 9,  ('verb', 'low'): 8,
    ('adj', 'high'):  8,  ('adj', 'mid'):  9,  ('adj', 'low'):  8,
}

selected_targets = []
for (pos, fb), count in targets_needed.items():
    available = bins[(pos, fb)]
    if len(available) < count:
        print(f"  WARNING: Only {len(available)} words available for ({pos}, {fb}), need {count}")
        count = min(len(available), count)
    random.shuffle(available)
    selected = available[:count]
    selected_targets.extend(selected)
    print(f"  Selected {len(selected)} {pos}/{fb}: {selected[:5]}...")

print(f"  Total selected: {len(selected_targets)} target words")

# ──────────────────────────────────────────────
# Step 5: Compute cosine similarities for targets
# ──────────────────────────────────────────────
print("Step 5: Computing cosine similarities...")

# Normalize embeddings
emb_norm = torch.nn.functional.normalize(embedding_matrix, dim=1)

# Prepare valid token arrays
valid_words_list = list(valid_tokens.keys())
valid_ids_list = [valid_tokens[w] for w in valid_words_list]
valid_emb = emb_norm[valid_ids_list]  # (N_valid, 768)

print(f"  Valid probe pool: {len(valid_words_list)} words")

SIMILARITY_LEVELS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

target_probes = {}

for i, target_word in enumerate(selected_targets):
    if (i + 1) % 10 == 0 or i == 0:
        print(f"  Target {i+1}/{len(selected_targets)}: {target_word}")
    
    target_id = valid_tokens[target_word]
    target_emb_vec = emb_norm[target_id].unsqueeze(0)  # (1, 768)
    
    # Compute cosine sims to all valid words (vectorized)
    sims = torch.mm(target_emb_vec, valid_emb.t()).squeeze(0).numpy()  # (N_valid,)
    
    probes = []
    used_words = {target_word}
    
    for level in SIMILARITY_LEVELS:
        # Compute distance to target level for all words
        dists = np.abs(sims - level)
        # Sort by distance
        sorted_indices = np.argsort(dists)
        
        # Find best unused word
        for idx in sorted_indices[:100]:  # Check top 100 candidates
            w = valid_words_list[idx]
            if w not in used_words:
                actual_sim = float(sims[idx])
                probes.append({
                    'probe_word': w,
                    'cosine_similarity': round(actual_sim, 4),
                    'target_similarity_level': level,
                    'probe_token_id': valid_ids_list[idx],
                    'probe_freq_rank': word_info.get(w, {}).get('freq_rank', -1),
                    'probe_pos': word_info.get(w, {}).get('pos', 'unknown'),
                    'wordnet_similarity': None,
                    'is_synonym': False,
                })
                used_words.add(w)
                break
    
    target_probes[target_word] = probes

# ──────────────────────────────────────────────
# Step 6: Find WordNet synonyms (EFFICIENT version)
# ──────────────────────────────────────────────
print("Step 6: Finding WordNet synonyms...")

synonym_count = 0
for i, target_word in enumerate(selected_targets):
    if (i + 1) % 20 == 0:
        print(f"  Synonym search {i+1}/{len(selected_targets)}...")
    
    target_synsets = wn.synsets(target_word)
    if not target_synsets:
        continue
    
    target_id = valid_tokens[target_word]
    target_emb_vec = emb_norm[target_id]
    
    used = {p['probe_word'] for p in target_probes[target_word]}
    used.add(target_word)
    
    best_synonym = None
    best_wup = 0.0
    best_cosine = None
    
    # For each target synset, look at lemmas from RELATED synsets
    # But only check synsets reachable via hypernym/hyponym paths (efficient)
    checked_synsets = set()
    for ts in target_synsets:
        # Check the synset's own lemmas
        for related_ss in [ts] + ts.hypernyms() + ts.hyponyms() + ts.also_sees() + ts.similar_tos():
            if related_ss in checked_synsets:
                continue
            checked_synsets.add(related_ss)
            
            wup = ts.wup_similarity(related_ss)
            if wup is None or wup <= 0.7:
                continue
            
            for lemma in related_ss.lemmas():
                lemma_name = lemma.name().lower()
                if '_' in lemma_name or lemma_name == target_word:
                    continue
                if lemma_name not in valid_tokens or lemma_name in used:
                    continue
                
                if wup > best_wup:
                    syn_id = valid_tokens[lemma_name]
                    cosine = float(torch.dot(target_emb_vec, emb_norm[syn_id]).item())
                    best_wup = wup
                    best_synonym = lemma_name
                    best_cosine = cosine
        
        # Also check 2-hop hypernyms/hyponyms
        for hyper in ts.hypernyms():
            for sibling in hyper.hyponyms():
                if sibling in checked_synsets:
                    continue
                checked_synsets.add(sibling)
                wup = ts.wup_similarity(sibling)
                if wup is None or wup <= 0.7:
                    continue
                for lemma in sibling.lemmas():
                    lemma_name = lemma.name().lower()
                    if '_' in lemma_name or lemma_name == target_word:
                        continue
                    if lemma_name not in valid_tokens or lemma_name in used:
                        continue
                    if wup > best_wup:
                        syn_id = valid_tokens[lemma_name]
                        cosine = float(torch.dot(target_emb_vec, emb_norm[syn_id]).item())
                        best_wup = wup
                        best_synonym = lemma_name
                        best_cosine = cosine
    
    if best_synonym:
        synonym_count += 1
        target_probes[target_word].append({
            'probe_word': best_synonym,
            'cosine_similarity': round(best_cosine, 4),
            'target_similarity_level': 'synonym',
            'probe_token_id': valid_tokens[best_synonym],
            'probe_freq_rank': word_info.get(best_synonym, {}).get('freq_rank', -1),
            'probe_pos': word_info.get(best_synonym, {}).get('pos', 'unknown'),
            'wordnet_similarity': round(best_wup, 4),
            'is_synonym': True,
        })

print(f"  Found WordNet synonyms for {synonym_count}/{len(selected_targets)} targets")

# ──────────────────────────────────────────────
# Step 7: Save outputs
# ──────────────────────────────────────────────
print("Step 7: Saving outputs...")

rows = []
for target_word in selected_targets:
    t_info = word_info[target_word]
    for probe in target_probes[target_word]:
        rows.append({
            'target_word': target_word,
            'probe_word': probe['probe_word'],
            'cosine_similarity': probe['cosine_similarity'],
            'wordnet_similarity': probe['wordnet_similarity'] if probe['wordnet_similarity'] else '',
            'target_similarity_level': probe['target_similarity_level'],
            'is_synonym': probe['is_synonym'],
            'target_freq_rank': t_info['freq_rank'],
            'target_zipf_freq': round(t_info['zipf_freq'], 2),
            'probe_freq_rank': probe['probe_freq_rank'],
            'target_pos': t_info['pos'],
            'probe_pos': probe['probe_pos'],
        })

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=[
        'target_word', 'probe_word', 'cosine_similarity', 'wordnet_similarity',
        'target_similarity_level', 'is_synonym', 'target_freq_rank', 'target_zipf_freq',
        'probe_freq_rank', 'target_pos', 'probe_pos'
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"  Saved {len(rows)} rows to {OUTPUT_CSV}")

# Save metadata
meta = {
    'n_targets': len(selected_targets),
    'targets': [],
}
for target_word in selected_targets:
    t_info = word_info[target_word]
    meta['targets'].append({
        'word': target_word,
        'token_id': t_info['token_id'],
        'pos': t_info['pos'],
        'freq_rank': t_info['freq_rank'],
        'zipf_freq': round(t_info['zipf_freq'], 2),
        'n_probes': len(target_probes[target_word]),
        'has_synonym': any(p['is_synonym'] for p in target_probes[target_word]),
    })

with open(OUTPUT_META, 'w') as f:
    json.dump(meta, f, indent=2)

print(f"  Saved metadata to {OUTPUT_META}")

# ──────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY STATISTICS")
print("="*60)
pos_counts = defaultdict(int)
freq_counts = defaultdict(int)
for t in meta['targets']:
    pos_counts[t['pos']] += 1
    if t['zipf_freq'] >= 4.5:
        freq_counts['high'] += 1
    elif t['zipf_freq'] >= 3.0:
        freq_counts['mid'] += 1
    else:
        freq_counts['low'] += 1

print(f"Total targets: {len(selected_targets)}")
print(f"POS distribution: {dict(pos_counts)}")
print(f"Frequency distribution: {dict(freq_counts)}")
print(f"Synonym coverage: {synonym_count}/{len(selected_targets)}")
print(f"Total probe pairs: {len(rows)}")

# Cosine similarity distribution across levels
print("\nCosine similarity by target level:")
level_sims = defaultdict(list)
for target_word in selected_targets:
    for probe in target_probes[target_word]:
        level_sims[probe['target_similarity_level']].append(probe['cosine_similarity'])

for level in SIMILARITY_LEVELS + ['synonym']:
    sims = level_sims.get(level, [])
    if sims:
        print(f"  Level {level:>8}: n={len(sims):3d}, mean={np.mean(sims):.4f}, std={np.std(sims):.4f}, range=[{min(sims):.4f}, {max(sims):.4f}]")

# Sample target-probe pairs
print("\nSample target-probe pairs:")
for target_word in selected_targets[:3]:
    print(f"\n  Target: {target_word} (pos={word_info[target_word]['pos']}, zipf={word_info[target_word]['zipf_freq']:.2f})")
    for probe in target_probes[target_word]:
        syn_str = f", wn_sim={probe['wordnet_similarity']}" if probe['wordnet_similarity'] else ""
        print(f"    -> {probe['probe_word']:15s} cos={probe['cosine_similarity']:+.4f}  level={probe['target_similarity_level']}{syn_str}")

print("\nDone!")
