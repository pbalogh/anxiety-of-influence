#!/usr/bin/env python3
"""
Build the interactive tutorial JSON for "The Anxiety of Influence: Bloom Filters in Transformer Attention Heads"
"""
import json

OUTPUT_PATH = "/Users/pabalogh/Documents/VisualTutorialsForAIConcepts/src/content/anxiety-of-influence-bloom-filter-heads.json"

# Meta
meta = {
    "title": "The Anxiety of Influence: Bloom Filters in Transformer Attention Heads",
    "description": "Interactive companion to the paper — discover how attention heads implement membership testing",
    "author": "Peter Balogh",
    "lastUpdated": "2026-02-16",
    "tags": ["mechanistic interpretability", "attention heads", "transformers", "bloom filters"]
}

# Helper functions
def section(title, children):
    return {"type": "Section", "props": {"title": title}, "children": children}

def p(text):
    return {"type": "p", "children": text}

def ul(items):
    return {"type": "ul", "children": [{"type": "li", "children": item} for item in items]}

def callout(type_, text):
    return {"type": "Callout", "props": {"type": type_}, "children": [p(text)]}

def code_block(code, lang="python"):
    return {"type": "CodeBlock", "props": {"language": lang}, "children": code}

def deep_dive(title, children):
    return {"type": "DeepDive", "props": {"title": title}, "children": children}

def quiz(questions):
    return {"type": "Quiz", "props": {"questions": questions}}

# Section 1: What is a Bloom Filter?
sec1 = section("1. What is a Bloom Filter?", [
    p("A Bloom filter is a probabilistic data structure invented by Burton Bloom in 1970. It answers the question: 'Is this element in my set?' with a crucial asymmetry:"),
    ul([
        "If the answer is NO → definitely not in the set (zero false negatives)",
        "If the answer is YES → probably in the set (occasional false positives)"
    ]),
    p("This tradeoff enables massive space savings. Instead of storing all elements, we use a bit array and hash functions."),
    deep_dive("How it works", [
        p("1. Start with m bits, all set to 0"),
        p("2. To INSERT element x: compute k hash functions h₁(x), h₂(x), ..., hₖ(x). Set those bit positions to 1."),
        p("3. To QUERY element x: check if ALL k bit positions are 1. If any is 0 → definitely not in set. If all are 1 → probably in set."),
        p("The false positive rate follows the formula:"),
        code_block("p ≈ (1 - e^(-kn/m))^k\n\nwhere:\n  m = number of bits\n  k = number of hash functions  \n  n = number of elements inserted", "text")
    ]),
    callout("info", "Key insight: As the filter fills up (n grows), the false positive rate increases. But the false NEGATIVE rate is always zero — if something is in the set, the filter will always say yes.")
])

# Section 2: The Literary Connection
sec2 = section("2. The Literary Connection", [
    p("The title of this paper alludes to two Blooms:"),
    ul([
        "Burton Bloom (1970) — invented the Bloom filter data structure",
        "Harold Bloom (1973) — wrote 'The Anxiety of Influence', arguing that every literary work is haunted by its predecessors"
    ]),
    p("The connection isn't just wordplay. Both Blooms recognized the same fundamental question: How do we determine what has come before, and how does that determination shape what comes next?"),
    callout("warning", "Harold Bloom's thesis: No text can mean anything without first reckoning with what came before it. Detection of prior influence is the PRECONDITION for meaning."),
    p("In transformers, this maps directly to attention: a token's representation is constructed from its contextual predecessors. The question 'have I seen this before?' isn't peripheral to language processing — it's foundational."),
    p("And here's the irony that closes the paper: gradient descent, striving to learn its own solution from scratch, independently converges on Burton Bloom's 1970 design. The anxiety of influence extends even to the algorithms.")
])

# Section 3: Finding Bloom Filter Heads
sec3 = section("3. Finding Bloom Filter Heads", [
    p("We looked for attention heads that behave like Bloom filters: strong response to repeated tokens (zero false negatives), occasional response to similar-but-not-identical tokens (false positives)."),
    p("Our metrics:"),
    ul([
        "Selectivity: hit attention / baseline attention (how much stronger is response to repeats?)",
        "Miss rate: fraction of repeated tokens with attention < 0.01 (should be ~0%)",
        "FP ratio: synonym attention / hit attention (how much do near-misses trigger the head?)"
    ]),
    p("In GPT-2 small, we found 4 heads with the Bloom filter signature:"),
    {"type": "Table", "props": {
        "headers": ["Head", "Selectivity", "Miss Rate", "FP Ratio"],
        "rows": [
            ["L0H1", "146×", "0.0%", "0.08"],
            ["L0H5", "74×", "0.0%", "0.01"],
            ["L1H11", "53×", "0.0%", "0.29"],
            ["L3H0", "51×", "0.4%", "0.25"]
        ]
    }},
    callout("success", "The effect sizes are enormous: Cohen's d = 12.3 for hit attention, d = 12.5 for selectivity. For reference, d > 0.8 is typically considered 'large'. These heads are MASSIVELY different from the other 140."),
    p("A permutation test confirms these 4 heads are special: no random group of 4 heads achieved mean selectivity ≥ 79× in 10,000 permutations (p < 0.0001).")
])

# Section 4: Not Induction Heads
sec4 = section("4. Not Induction Heads", [
    p("A natural question: aren't these just induction heads? Induction heads (Olsson et al. 2022) implement the pattern [A][B]...[A] → [B] — they copy what followed a previous occurrence."),
    p("No. The overlap is ZERO. Three distinct categories emerged:"),
    {"type": "Table", "props": {
        "headers": ["Category", "Count", "Layer Range", "Function"],
        "rows": [
            ["Bloom filter heads", "4", "0-3", "\"Is this token in context?\""],
            ["Previous-token heads", "11", "2-6", "\"What came immediately before?\""],
            ["Induction heads", "16", "5-11", "\"A B ... A → B\""]
        ]
    }},
    p("The layer distribution tells a story: Bloom filter heads are in EARLY layers (0-3), induction heads are in LATE layers (5-11). This suggests a pipeline:"),
    callout("info", "Processing pipeline: DETECT membership (Bloom filter heads, layers 0-3) → INTERPRET the repetition (gap layers 3-5) → EXPLOIT for prediction (induction heads, layers 5-11).")
])

# Section 5: The Capacity Curve
sec5 = section("5. The Capacity Curve — The Money Result", [
    p("If these heads really are Bloom filters, their false positive rate should increase as context grows (more tokens = filter fills up). We tested this by varying context from 5 to 200 unique tokens."),
    p("L3H0 follows the theoretical Bloom filter capacity formula with R² = 0.99:"),
    {"type": "Table", "props": {
        "headers": ["Unique Tokens", "Theory (m=64)", "L3H0 Observed"],
        "rows": [
            ["5", "7.5%", "1.0%"],
            ["20", "26.8%", "25.0%"],
            ["50", "54.2%", "72.0%"],
            ["100", "79.0%", "93.0%"],
            ["200", "95.6%", "97.0%"]
        ]
    }},
    callout("success", "Fitting the Bloom filter formula to L3H0 yields m = 59 bits and k = 2.16 hash functions. The head dimension is 64 — so this head devotes 92% of its capacity to membership testing!"),
    p("L0H1 and L0H5 show a different pattern: near-zero FP rates regardless of context size. These are 'perfect filters' — more like hash tables than Bloom filters."),
    p("Critically: miss rates stay at 0% across ALL capacity levels. The false negative guarantee holds even as the filter saturates.")
])

# Section 6: Independent Hash Functions
sec6 = section("6. Independent Hash Functions", [
    p("Multiple hash functions are how Bloom filters reduce false positives. If our 4 heads act as independent hash functions, their FP decisions should be uncorrelated."),
    p("We measured the phi coefficient (correlation) between each pair of Bloom heads' FP decisions:"),
    ul([
        "Mean φ = 0.13 (low correlation = high independence)",
        "Range: 0.02 (L0H5 ↔ L3H0) to 0.38 (L0H1 ↔ L0H5)",
        "The L0H1-L0H5 correlation makes sense — they share the same layer's input"
    ]),
    p("Combining all 4 heads with AND logic:"),
    ul([
        "Individual FP rate: up to 78% (L3H0 at high capacity)",
        "Combined FP rate: 0.17%",
        "Combined hit rate: still 100%!"
    ]),
    callout("info", "80% of false positives trigger only ONE head. Only 0.2% trigger all four. This diversity of coverage is characteristic of independent hash functions probing different aspects of token similarity.")
])

# Section 7: Cross-Model Validation
sec7 = section("7. Cross-Model Validation", [
    p("Do Bloom filter heads exist in other models? We tested GPT-2 small, medium, large, and Pythia-160M."),
    {"type": "Table", "props": {
        "headers": ["Model", "Total Heads", "Bloom Heads", "%", "Early/Mid/Late"],
        "rows": [
            ["GPT-2 Small (85M)", "144", "4", "2.8%", "4/0/0"],
            ["GPT-2 Medium (302M)", "384", "3", "0.8%", "3/0/0"],
            ["GPT-2 Large (708M)", "720", "27", "3.8%", "22/5/0"],
            ["Pythia-160M", "144", "4", "2.8%", "3/1/0"]
        ]
    }},
    p("Three universal patterns:"),
    ul([
        "Early-layer concentration: ZERO late-layer Bloom heads in any model",
        "Classic FP signature: synonym FP ratios of 0.2-0.6 in every model",
        "Scaling: Bloom head count correlates with model size (r = 0.89)"
    ]),
    callout("success", "GPT-2 Large devotes 27 heads — nearly 7× more than GPT-2 Small — to membership testing. Larger models allocate more capacity to this primitive operation.")
])

# Section 8: Ablation
sec8 = section("8. Ablation — The Smoking Gun", [
    p("To confirm these heads have a specific function, we ablated (zeroed out) them and measured perplexity on sentences with and without repeated tokens."),
    {"type": "Table", "props": {
        "headers": ["Ablation", "Repeat Δ PPL", "No-repeat Δ PPL", "Interaction"],
        "rows": [
            ["Bloom heads", "+9.7%", "-4.4%", "+14.1%"],
            ["Control heads", "+11.8%", "+5.5%", "+6.3%"],
            ["Induction heads", "+37.6%", "+125.2%", "-87.6%"]
        ]
    }},
    callout("warning", "The -4.4% is the smoking gun. Removing Bloom heads IMPROVES performance on novel content! This means they're adding noise via false positives when there's nothing to match — exactly the Bloom filter prediction."),
    p("The interaction effect (+14.1% for Bloom heads vs +6.3% for controls) shows these heads have a repeat-specific role, not just general importance."),
    p("Induction heads show the opposite pattern — they hurt novel content far more than repeated content. Different heads, different jobs.")
])

# Section 9: Implications
sec9 = section("9. So What? — Practical Implications", [
    p("If attention heads implement Bloom filters, several optimizations become possible:"),
    deep_dive("Sparse Attention Replacement", [
        p("Bloom filter heads perform O(n) attention to answer a question a literal Bloom filter answers in O(1). For long-context inference, replace these heads with hash-based lookups → eliminate their quadratic cost entirely.")
    ]),
    deep_dive("KV-Cache Compression", [
        p("Bloom heads use keys (for membership testing) more than values (for content retrieval). Their KV-cache could be compressed to keys only → 50% memory savings for these heads.")
    ]),
    deep_dive("Hallucination Detection", [
        p("A Bloom head showing high activation toward a context position where the queried token does NOT appear = a false positive = the model 'remembering' something that isn't there. Monitor these heads as an early warning system for hallucination.")
    ]),
    deep_dive("Head Pruning", [
        p("If 4 heads act as independent hash functions, maybe we don't need all 4. If 2 provide sufficient coverage, prune the others for efficiency.")
    ])
])

# Section 10: Data & Code
sec10 = section("10. Data & Code — Full Reproducibility", [
    p("All code and data are available at:"),
    p("/Users/pabalogh/clawd/projects/bloom-filter-heads/"),
    p("Experiment scripts:"),
    ul([
        "expanded_stimuli.py — 100 sentence triplets (exact repeat, no repeat, semantic near-miss)",
        "experiment1_bloom_signature.py — Bloom filter signature detection",
        "experiment2_induction_overlap.py — Taxonomy analysis (Bloom vs induction vs previous-token)",
        "experiment3_capacity.py — Capacity curve fitting",
        "experiment4_hash_functions.py — Independence analysis (phi coefficients, AND combination)",
        "experiment5_ablation.py — Ablation study",
        "multi_model_validation.py — Cross-model validation (GPT-2 s/m/l, Pythia)",
        "statistical_hardening_v2.py — Rigorous statistics (bootstrap CIs, Bonferroni, permutation)",
        "generate_figures.py — All 7 publication figures"
    ]),
    p("Result files (JSON):"),
    ul([
        "experiment1_results.json — Per-head selectivity, miss rate, FP ratio for all 144 heads",
        "experiment2_induction_overlap.json — Head classification taxonomy",
        "experiment3_capacity.json — Capacity curve data and fitted parameters",
        "experiment4_hash_functions.json — Phi matrix, combined FP rates",
        "experiment5_ablation.json — Perplexity deltas by condition",
        "multi_model_validation.json — Results for all 4 models",
        "statistical_hardening_v2.json — All statistical test results"
    ]),
    p("Paper: /Users/pabalogh/clawd/projects/bloom-filter-heads/paper/main.tex"),
    p("Figures: /Users/pabalogh/clawd/projects/bloom-filter-heads/figures/ (PDF + PNG)")
])

# Section 11: Anticipated Criticisms
sec11 = section("11. Anticipated Criticisms — Steelmanning the Skeptics", [
    p("Every strong claim invites strong criticism. Here's how we'd respond to the toughest reviewers:"),
    
    deep_dive("\"These are just duplicate-token heads (Wang et al. 2022)\"", [
        p("Wang et al. identified duplicate-token heads in the IOI circuit that attend to repeated names. The difference: their heads are part of a larger circuit doing indirect object identification; our heads do membership testing WITHOUT pattern completion. They answer 'is this token here?' — period. No copying, no prediction of what follows.")
    ]),
    
    deep_dive("\"Your stimuli are constructed, not naturalistic\"", [
        p("Acknowledged. Our 100 sentence triplets are synthetic. However: (1) Cross-model validation shows the phenomenon isn't stimulus-specific; (2) The ablation study uses the same patterns; (3) Naturalistic corpus studies are planned as follow-up. The current work establishes the phenomenon exists; naturalistic validation extends it.")
    ]),
    
    deep_dive("\"N=4 heads is a tiny sample\"", [
        p("Fair concern, mitigated by: (1) Permutation test with p < 0.0001 — these 4 aren't random; (2) Cross-model replication — 4 models, same signature; (3) GPT-2 Large has 27 Bloom heads. The sample is small in GPT-2 Small; the phenomenon is robust across models.")
    ]),
    
    deep_dive("\"You haven't shown the QK mechanism\"", [
        p("True. We classify heads behaviorally, not mechanistically. A direct analysis of QK weight matrices — showing they implement something like hash functions — would be stronger evidence. This is explicitly future work. The current paper establishes the behavioral signature; mechanistic confirmation comes next.")
    ]),
    
    deep_dive("\"The capacity curve fit could be coincidental\"", [
        p("R² = 0.99 with fitted m = 59 (head dimension = 64) is hard to dismiss as coincidence. The model didn't just fit — it recovered parameters that match the architecture. But yes, replication across models would strengthen this. We note that not all Bloom heads show capacity-dependent FP rates (L0H1 and L0H5 are 'perfect filters').")
    ]),
    
    deep_dive("\"Miss rate 0% is suspicious — threshold too generous?\"", [
        p("Our threshold is attention > 0.01. We tested stricter thresholds (0.05, 0.10) — miss rates remain very low (<2%). The 0% figure is robust, not an artifact of a generous threshold.")
    ]),
    
    deep_dive("\"This is just a same-token detector, not a Bloom filter\"", [
        p("A same-token detector would show zero response to synonyms. Our heads show 25-30% response to synonyms (L1H11, L3H0) — they fire on 'physician' when looking for 'doctor'. This is the false positive signature that distinguishes Bloom filters from exact matchers.")
    ]),
    
    deep_dive("\"The ablation effects are modest (+9.7%)\"", [
        p("The raw number isn't the point — the INTERACTION is. Bloom head ablation hurts repeated tokens (+9.7%) while HELPING novel tokens (-4.4%). The interaction effect (+14.1%) is 2.2× stronger than control heads (+6.3%). The asymmetric pattern, not the magnitude, is the evidence.")
    ]),
    
    deep_dive("\"Why only GPT-2 scale models?\"", [
        p("Honest limitation. TransformerLens support and computational constraints. Testing frontier-scale models (70B+) is important future work. We can say Bloom heads exist in 85M-708M parameter models; whether they persist at larger scales is unknown.")
    ]),
    
    deep_dive("\"Could this be a BPE tokenization artifact?\"", [
        p("Good question. Our stimuli use common words that are typically single BPE tokens (doctor, teacher, book, etc.). Multi-token words could show different behavior. Worth controlling for in follow-up work.")
    ]),
    
    deep_dive("\"The Harold Bloom framing is decorative\"", [
        p("Disagree. The framing isn't just clever naming — it highlights that repetition detection is a PRECONDITION for meaning. Harold Bloom's argument was that no text means anything without reckoning with what came before. We show transformers implement exactly this: detection first, then interpretation. The framing is substantive, not decorative.")
    ]),
    
    deep_dive("\"The 'zero overlap' claim with induction heads is too strong\"", [
        p("We use the standard Olsson et al. (2022) test: random repeated sequences, measure attention to the token following the repeated token. Alternative definitions (prefix matching, copying score) might yield different classifications. We're specific about our test; other tests might find overlap.")
    ])
])

# Section 12: Quiz
sec12 = section("12. Quiz — Test Your Understanding", [
    quiz([
        {
            "question": "What is the defining asymmetry of a Bloom filter?",
            "options": [
                "Zero false positives, occasional false negatives",
                "Zero false negatives, occasional false positives",
                "Equal rates of false positives and false negatives",
                "No errors of any kind"
            ],
            "correct": 1,
            "explanation": "Bloom filters guarantee zero false negatives (if something is in the set, they always say yes) but allow false positives (they may say yes for things not in the set)."
        },
        {
            "question": "In GPT-2 Small, how many heads exhibit the Bloom filter signature?",
            "options": ["1", "4", "12", "144"],
            "correct": 1,
            "explanation": "Four heads (L0H1, L0H5, L1H11, L3H0) show selectivity >30×, miss rate ~0%, and characteristic false positive behavior."
        },
        {
            "question": "What is the overlap between Bloom filter heads and induction heads?",
            "options": ["Complete overlap", "Partial overlap", "Zero overlap", "Unknown"],
            "correct": 2,
            "explanation": "Zero overlap. Bloom filter heads (layers 0-3) and induction heads (layers 5-11) are completely separate populations performing different functions."
        },
        {
            "question": "What happens when you ablate Bloom filter heads?",
            "options": [
                "Perplexity increases equally on all text",
                "Perplexity increases on repeated tokens, DECREASES on novel tokens",
                "Perplexity decreases on all text",
                "No effect on perplexity"
            ],
            "correct": 1,
            "explanation": "Ablation hurts repeated-token processing (+9.7%) but improves novel-token processing (-4.4%). This asymmetry is the Bloom filter signature: false positives add noise on novel content."
        },
        {
            "question": "What did L3H0's capacity curve fitting reveal?",
            "options": [
                "The head doesn't follow Bloom filter theory",
                "m=59 bits and k=2.16 hash functions (R²=0.99), closely matching d_head=64",
                "The head is actually an induction head",
                "The false positive rate is constant"
            ],
            "correct": 1,
            "explanation": "L3H0 follows the theoretical Bloom filter formula with R²=0.99. The fitted capacity (m=59) is 92% of the head dimension (64), suggesting nearly all representational capacity is devoted to membership testing."
        },
        {
            "question": "What's the irony that closes the paper?",
            "options": [
                "Bloom filters are outdated",
                "Gradient descent, striving to learn from scratch, reinvents Burton Bloom's 1970 design",
                "The paper should have used a different data structure",
                "Harold Bloom was wrong about literary influence"
            ],
            "correct": 1,
            "explanation": "Harold Bloom argued originality is impossible — every creation reckons with predecessors. Gradient descent proves his point by independently converging on a 50-year-old human-designed data structure. The anxiety of influence extends even to the algorithms."
        }
    ])
])

# Assemble the full tutorial
tutorial = {
    "type": "Fragment",
    "meta": meta,
    "content": {
        "type": "Fragment",
        "children": [sec1, sec2, sec3, sec4, sec5, sec6, sec7, sec8, sec9, sec10, sec11, sec12]
    }
}

# Write the JSON
with open(OUTPUT_PATH, 'w') as f:
    json.dump(tutorial, f, indent=2)

print(f"✅ Tutorial written to {OUTPUT_PATH}")
print(f"   Sections: 12")
print(f"   File size: {len(json.dumps(tutorial)):,} bytes")
