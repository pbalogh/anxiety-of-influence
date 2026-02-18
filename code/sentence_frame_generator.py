#!/usr/bin/env python3
"""
Sentence Frame Generator for Bloom Filter Heads Paper
======================================================

For each target word, generates template sentences with TWO occurrences
of the target, suitable for testing repeated-token attention.

Creates three versions:
  EXACT: both slots have the target word
  PROBE: second slot replaced with probe word (one per similarity level)
  CONTROL: second slot replaced with an unrelated frequency-matched word

Output: ../data/sentence_frames.json
"""

import os
import sys
import json
import csv
import random
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

from transformers import GPT2Tokenizer
from wordfreq import zipf_frequency

# Reproducibility
random.seed(42)

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
OUTPUT_JSON = os.path.join(DATA_DIR, "sentence_frames.json")
INPUT_CSV = os.path.join(DATA_DIR, "target_probe_matrix.csv")
INPUT_META = os.path.join(DATA_DIR, "target_words_meta.json")

# ──────────────────────────────────────────────
# Load tokenizer
# ──────────────────────────────────────────────
print("Loading GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def is_single_token(text):
    """Check if every word in text encodes to a single GPT-2 token."""
    # Tokenize the full text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    # Each word (space-separated) should be one token
    words = text.split()
    return len(tokens) == len(words)

def get_token_count(text):
    """Return number of GPT-2 tokens for text."""
    return len(tokenizer.encode(text, add_special_tokens=False))

def verify_single_token_word(word):
    """Verify a word is a single token when preceded by space."""
    tokens = tokenizer.encode(" " + word, add_special_tokens=False)
    return len(tokens) == 1

# ──────────────────────────────────────────────
# Load target/probe data
# ──────────────────────────────────────────────
print("Loading target/probe matrix...")
with open(INPUT_META) as f:
    meta = json.load(f)

# Read CSV
probes_by_target = defaultdict(list)
with open(INPUT_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        probes_by_target[row['target_word']].append(row)

target_info = {t['word']: t for t in meta['targets']}
print(f"  Loaded {len(target_info)} targets with probes")

# ──────────────────────────────────────────────
# Build pool of single-token filler words
# ──────────────────────────────────────────────
print("Building filler word pools...")

# Pre-verified single-token function words and content words for templates
# Each of these has been chosen to be common and reliably single-token in GPT-2
SINGLE_TOKEN_NOUNS = [
    "man", "woman", "child", "dog", "cat", "house", "car", "book", "door",
    "room", "table", "chair", "tree", "road", "city", "team", "game", "food",
    "water", "fire", "king", "boat", "bird", "fish", "song", "film", "shop",
    "park", "ball", "stone", "horse", "farm", "gold", "iron", "wood", "glass",
    "rain", "snow", "wind", "hill", "lake", "star", "moon", "sun", "hand",
    "face", "head", "eye", "foot", "arm", "back", "heart", "mind", "soul",
]

SINGLE_TOKEN_VERBS = [
    "saw", "found", "made", "took", "gave", "told", "left", "felt", "kept",
    "brought", "lost", "paid", "met", "ran", "held", "called", "turned",
    "moved", "lived", "played", "worked", "tried", "asked", "needed",
    "started", "watched", "walked", "served", "loved", "reached",
]

SINGLE_TOKEN_ADVERBS = [
    "quickly", "slowly", "carefully", "gently", "quietly", "loudly",
    "often", "always", "never", "soon", "still", "already", "again",
    "here", "there", "now", "then", "once", "twice",
]

SINGLE_TOKEN_ADJECTIVES = [
    "small", "large", "old", "new", "great", "good", "long", "high",
    "big", "dark", "young", "real", "full", "deep", "strong", "hard",
    "clean", "soft", "warm", "cold", "sweet", "wild", "bright", "fresh",
]

# Verify all fillers are single tokens
def verify_pool(name, words):
    valid = []
    for w in words:
        if verify_single_token_word(w):
            valid.append(w)
        else:
            print(f"  WARNING: '{w}' is NOT single-token in GPT-2, removing from {name}")
    return valid

SINGLE_TOKEN_NOUNS = verify_pool("nouns", SINGLE_TOKEN_NOUNS)
SINGLE_TOKEN_VERBS = verify_pool("verbs", SINGLE_TOKEN_VERBS)
SINGLE_TOKEN_ADVERBS = verify_pool("adverbs", SINGLE_TOKEN_ADVERBS)
SINGLE_TOKEN_ADJECTIVES = verify_pool("adjectives", SINGLE_TOKEN_ADJECTIVES)

# ──────────────────────────────────────────────
# Sentence templates
# ──────────────────────────────────────────────
# Templates use {TARGET} as placeholder for the target/probe word
# Each template has a 'pos' field indicating which POS it's designed for
# Templates are designed so the second {TARGET} can be swapped with a synonym

NOUN_TEMPLATES = [
    "The {TARGET} was quite impressive and the {TARGET} caught everyone off guard",
    "A {TARGET} appeared near the building and the {TARGET} drew much attention",
    "The {TARGET} seemed very unusual and the {TARGET} surprised the whole crowd",
    "Every {TARGET} has some value and each {TARGET} deserves close study",
    "That {TARGET} was found outside and the {TARGET} was brought inside quickly",
    "One {TARGET} stood in the corner and another {TARGET} sat on the shelf",
    "The first {TARGET} looked quite old and the second {TARGET} looked brand new",
    "His {TARGET} was well maintained and her {TARGET} needed some repairs soon",
    "A bright {TARGET} lit the whole room and the {TARGET} shone through the night",
    "This {TARGET} costs very little money and that {TARGET} costs much more today",
    "The large {TARGET} blocked the whole path and the {TARGET} had to be moved",
    "No {TARGET} could solve the problem but one {TARGET} came very close indeed",
    "My {TARGET} arrived early this morning and your {TARGET} should arrive by noon",
    "The new {TARGET} worked just fine but the old {TARGET} needed to be fixed",
    "Some {TARGET} can be found here while more {TARGET} exist over there somewhere",
    "The lost {TARGET} turned up last week and the {TARGET} was in great shape",
    "Each {TARGET} plays a key role and every {TARGET} helps the team succeed",
]

VERB_TEMPLATES = [
    "The man {TARGET} the task quickly and the woman {TARGET} the job slowly",
    "She {TARGET} the work with care and he {TARGET} the project with skill",
    "They {TARGET} the food for dinner and we {TARGET} the meal for lunch",
    "The team {TARGET} the game plan early and the coach {TARGET} the whole strategy",
    "He {TARGET} the door with force and she {TARGET} the window with ease",
    "I {TARGET} the old book first and then {TARGET} the new one right after",
    "The child {TARGET} the toy gently and the dog {TARGET} the bone quickly",
    "We {TARGET} the issue last month and they {TARGET} the matter this week",
    "The teacher {TARGET} the class all day and the student {TARGET} the lesson well",
    "People {TARGET} the event each year and crowds {TARGET} the show every time",
    "First she {TARGET} the letter twice and then she {TARGET} the note once more",
    "The bird {TARGET} the seed from the ground and the cat {TARGET} the fish off the plate",
]

ADJ_TEMPLATES = [
    "The {TARGET} house stood on the hill and the {TARGET} building sat by the lake",
    "A {TARGET} light filled the whole room and a {TARGET} glow spread through the hall",
    "His {TARGET} voice carried through the air and her {TARGET} tone filled the space",
    "The {TARGET} road stretched for many miles and the {TARGET} path wound through the trees",
    "That {TARGET} song played on the radio and this {TARGET} tune rang in my head",
    "A {TARGET} wind blew from the north and a {TARGET} storm came from the east",
    "The {TARGET} child ran through the park and the {TARGET} dog chased close behind",
    "Some {TARGET} days feel quite long while most {TARGET} nights pass very fast",
    "His {TARGET} smile lit the whole room and her {TARGET} laugh filled the air",
    "The {TARGET} book sat on the shelf and the {TARGET} film played on the screen",
    "One {TARGET} wave crashed on the shore and each {TARGET} tide swept the coast",
    "A very {TARGET} meal was served first and a very {TARGET} dish came out next",
]

def pick_template(pos, target_word, used_templates):
    """Pick a template suitable for the POS, avoiding reuse."""
    if pos == 'noun':
        pool = NOUN_TEMPLATES
    elif pos == 'verb':
        pool = VERB_TEMPLATES
    elif pos == 'adj':
        pool = ADJ_TEMPLATES
    else:
        pool = NOUN_TEMPLATES
    
    # Try to find an unused template
    available = [t for t in pool if t not in used_templates]
    if not available:
        available = pool  # Reuse if exhausted
    
    # Pick randomly
    template = random.choice(available)
    return template

# ──────────────────────────────────────────────
# Validate and fix sentences for single-token property
# ──────────────────────────────────────────────
def validate_sentence(sentence, target_word, replacement_word=None):
    """
    Check that every word in the sentence is a single GPT-2 token.
    Returns (is_valid, token_count, problem_words).
    """
    words = sentence.split()
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    
    # Check each word
    problem_words = []
    for word in words:
        enc = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(enc) != 1:
            # Try without space (first word)
            enc2 = tokenizer.encode(word, add_special_tokens=False)
            if len(enc2) != 1:
                problem_words.append((word, len(enc)))
    
    return len(problem_words) == 0, len(tokens), problem_words

# ──────────────────────────────────────────────
# Build control word pool (frequency-matched unrelated words)
# ──────────────────────────────────────────────
print("Building control word pool...")

# Load all valid single-token words from the vocabulary
control_pool = {}  # word -> zipf_freq
for token_id in range(tokenizer.vocab_size):
    raw_token = tokenizer.convert_ids_to_tokens(token_id)
    if not raw_token.startswith("Ġ"):
        continue
    word = raw_token[1:]
    if not word.isalpha() or not word.islower() or len(word) < 4:
        continue
    freq = zipf_frequency(word, 'en')
    if freq < 2.0:
        continue
    # Verify single-token
    encoded = tokenizer.encode(" " + word, add_special_tokens=False)
    if len(encoded) == 1 and encoded[0] == token_id:
        control_pool[word] = freq

print(f"  Control pool: {len(control_pool)} words")

def find_control_word(target_word, target_pos, exclude_words):
    """Find a frequency-matched unrelated word."""
    target_freq = zipf_frequency(target_word, 'en')
    
    candidates = []
    for word, freq in control_pool.items():
        if word in exclude_words:
            continue
        freq_diff = abs(freq - target_freq)
        if freq_diff < 0.5:  # Within 0.5 zipf
            candidates.append((word, freq_diff))
    
    if not candidates:
        # Relax constraint
        for word, freq in control_pool.items():
            if word in exclude_words:
                continue
            freq_diff = abs(freq - target_freq)
            if freq_diff < 1.5:
                candidates.append((word, freq_diff))
    
    if not candidates:
        return None
    
    # Sort by frequency match, pick randomly from top 20
    candidates.sort(key=lambda x: x[1])
    top = candidates[:20]
    random.shuffle(top)
    return top[0][0]

# ──────────────────────────────────────────────
# Generate sentence frames
# ──────────────────────────────────────────────
print("Generating sentence frames...")

all_frames = []
used_templates = defaultdict(set)  # pos -> set of used templates
validation_failures = 0
total_sentences = 0

for i, target_data in enumerate(meta['targets']):
    target_word = target_data['word']
    pos = target_data['pos']
    
    if (i + 1) % 20 == 0:
        print(f"  Processing target {i+1}/{len(meta['targets'])}: {target_word}")
    
    # Pick a template
    template = pick_template(pos, target_word, used_templates[pos])
    used_templates[pos].add(template)
    
    # Generate EXACT version (both slots = target word)
    exact_sentence = template.replace("{TARGET}", target_word)
    
    # Validate
    is_valid, n_tokens, problems = validate_sentence(exact_sentence, target_word)
    if not is_valid:
        # Try another template
        for _ in range(10):
            template = pick_template(pos, target_word, set())
            exact_sentence = template.replace("{TARGET}", target_word)
            is_valid, n_tokens, problems = validate_sentence(exact_sentence, target_word)
            if is_valid:
                break
    
    if not is_valid:
        validation_failures += 1
        print(f"  WARNING: Could not find valid template for '{target_word}': {problems}")
    
    # Find positions of target word in tokenized sentence
    tokens = tokenizer.encode(exact_sentence, add_special_tokens=False)
    target_token_id = tokenizer.encode(" " + target_word, add_special_tokens=False)
    # Also check without space (first word)
    target_token_id_no_space = tokenizer.encode(target_word, add_special_tokens=False)
    
    target_positions = []
    for pos_idx, t in enumerate(tokens):
        if t in target_token_id or t in target_token_id_no_space:
            target_positions.append(pos_idx)
    
    # Build frame entry
    frame = {
        'target_word': target_word,
        'target_pos': pos,
        'target_token_id': target_data['token_id'],
        'template': template,
        'exact_sentence': exact_sentence,
        'target_positions': target_positions,
        'n_tokens': n_tokens,
        'is_valid': is_valid,
        'sentences': [],
    }
    
    # EXACT version
    frame['sentences'].append({
        'condition': 'exact',
        'sentence': exact_sentence,
        'slot2_word': target_word,
        'cosine_similarity': 1.0,
        'similarity_level': 'exact',
    })
    total_sentences += 1
    
    # PROBE versions (one per similarity level)
    probes = probes_by_target.get(target_word, [])
    exclude_words = {target_word} | {p['probe_word'] for p in probes}
    
    for probe_row in probes:
        probe_word = probe_row['probe_word']
        sim_level = probe_row['target_similarity_level']
        cosine_sim = float(probe_row['cosine_similarity'])
        
        # Replace only the SECOND occurrence of target in template
        parts = template.split("{TARGET}")
        if len(parts) >= 3:
            probe_sentence = parts[0] + target_word + parts[1] + probe_word + parts[2]
        else:
            probe_sentence = template.replace("{TARGET}", probe_word, 1)
            # This replaces the first, but we want to replace the second
            # Let's be more careful
            first_idx = template.find("{TARGET}")
            second_idx = template.find("{TARGET}", first_idx + len("{TARGET}"))
            if second_idx >= 0:
                probe_sentence = (template[:first_idx] + target_word + 
                                 template[first_idx + len("{TARGET}"):second_idx] + 
                                 probe_word + 
                                 template[second_idx + len("{TARGET}"):])
            else:
                probe_sentence = template.replace("{TARGET}", target_word, 1).replace("{TARGET}", probe_word, 1)
        
        is_synonym = probe_row.get('is_synonym', 'False')
        wn_sim = probe_row.get('wordnet_similarity', '')
        
        condition = 'synonym' if is_synonym == 'True' else 'probe'
        
        frame['sentences'].append({
            'condition': condition,
            'sentence': probe_sentence,
            'slot2_word': probe_word,
            'cosine_similarity': cosine_sim,
            'similarity_level': sim_level,
            'wordnet_similarity': float(wn_sim) if wn_sim else None,
        })
        total_sentences += 1
    
    # CONTROL version
    control_word = find_control_word(target_word, pos, exclude_words)
    if control_word:
        # Same template replacement as probe
        parts = template.split("{TARGET}")
        if len(parts) >= 3:
            control_sentence = parts[0] + target_word + parts[1] + control_word + parts[2]
        else:
            first_idx = template.find("{TARGET}")
            second_idx = template.find("{TARGET}", first_idx + len("{TARGET}"))
            if second_idx >= 0:
                control_sentence = (template[:first_idx] + target_word + 
                                   template[first_idx + len("{TARGET}"):second_idx] + 
                                   control_word + 
                                   template[second_idx + len("{TARGET}"):])
            else:
                control_sentence = template.replace("{TARGET}", target_word, 1).replace("{TARGET}", control_word, 1)
        
        frame['sentences'].append({
            'condition': 'control',
            'sentence': control_sentence,
            'slot2_word': control_word,
            'cosine_similarity': None,  # Not computed
            'similarity_level': 'control',
            'control_freq': round(zipf_frequency(control_word, 'en'), 2),
            'target_freq': round(zipf_frequency(target_word, 'en'), 2),
        })
        total_sentences += 1
    
    all_frames.append(frame)

# ──────────────────────────────────────────────
# Final validation pass
# ──────────────────────────────────────────────
print("\nFinal validation pass...")
invalid_count = 0
for frame in all_frames:
    for sent_entry in frame['sentences']:
        is_valid, n_tok, problems = validate_sentence(sent_entry['sentence'], frame['target_word'])
        if not is_valid:
            invalid_count += 1
            if invalid_count <= 10:
                print(f"  INVALID: '{sent_entry['sentence'][:60]}...' problems: {problems}")

print(f"  Invalid sentences: {invalid_count}/{total_sentences}")

# ──────────────────────────────────────────────
# Save output
# ──────────────────────────────────────────────
print("Saving output...")
with open(OUTPUT_JSON, 'w') as f:
    json.dump({
        'n_targets': len(all_frames),
        'n_total_sentences': total_sentences,
        'frames': all_frames,
    }, f, indent=2)

print(f"  Saved {len(all_frames)} frames ({total_sentences} sentences) to {OUTPUT_JSON}")

# ──────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
conditions = defaultdict(int)
for frame in all_frames:
    for sent in frame['sentences']:
        conditions[sent['condition']] += 1

print(f"Targets: {len(all_frames)}")
print(f"Total sentences: {total_sentences}")
print(f"By condition: {dict(conditions)}")
print(f"Validation failures (template level): {validation_failures}")
print(f"Invalid sentences (token level): {invalid_count}")

# Sample
print("\nSample frames:")
for frame in all_frames[:3]:
    print(f"\n  Target: {frame['target_word']} ({frame['target_pos']})")
    print(f"  Template: {frame['template']}")
    for sent in frame['sentences'][:4]:
        print(f"    [{sent['condition']:8s}] {sent['sentence'][:80]}...")
    if len(frame['sentences']) > 4:
        print(f"    ... and {len(frame['sentences'])-4} more sentences")

print("\nDone!")
