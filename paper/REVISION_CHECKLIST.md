# Revision Checklist — "The Anxiety of Influence"

Compiled from 11-pass critical review (2026-02-16 to 2026-02-17).
Ordered by reviewer-damage potential.

---

## 🔴 TIER 1: Manuscript-Breaking (fix before arXiv posting)

### 1. Text-Figure Contradictions [Pass 6 #1, #2, #3]
- [x] **Fig 4 phi values vs §4.4 text.** Text said φ range 0.02–0.38. Figure shows 0.08–0.18. → **FIXED:** Text now says 0.08–0.18, matching figure. Raw data confirms avg=0.129.
- [x] **Fig 6 FP distribution vs §4.4 text.** Text said 63%/17%/0.2%. Figure shows 19.7%/44%/24%/12%/0.2%. → **FIXED:** Text now reports the aggregate categories from raw data (19.7% none, 80.2% mixed, 0.2% all four). Figure caption notes that 1-3 breakdown is estimated. Figure regeneration with real per-probe counts is a follow-up task.
- [x] **Orphan Fig 5 (combined_fp.pdf).** → **FIXED:** Now included as Figure with honest discussion of observed vs predicted FP rates. Text acknowledges ~2 orders of magnitude departure from independence prediction.
- [x] **Mean phi ($\bar{\phi} = 0.13$)** — confirmed consistent with figure and raw data. No change needed.

### 2. L3H0 Miss Rate Contradiction [Pass 1 #1, Pass 8 #1]
- [x] **FIXED.** Raw data confirms: L3H0 has 1 miss out of 238 observations = 0.4%. Table 1 was correct. All prose now says "near-zero" instead of "zero." §4.1 specifies "0.4\% (1/238 token-level observations)" for L3H0 and 0% for the other three. Footnote explains where 238 comes from. Fig 3 caption updated. Abstract already said "near-zero miss rates."

### 3. L0H1/L0H5 Identity Crisis [Pass 8 #14]
- [x] §4.1 calls them Bloom filter heads. §4.3 says they're "approximate hash tables rather than Bloom filters." → **IN PROGRESS (sub-agent).** Reframe as a subtype — high-precision Bloom filters with near-zero FP. Keep them in the Bloom category with a note about their distinct FP profile.

### 4. Missing Critical Citations [Pass 3 #1-3, Pass 9 #1]
- [x] **IN PROGRESS (sub-agent).** Adding:
  - Kraska et al. (2018) "Learned Index Structures"
  - Rae et al. (2019) "Meta-Learning Neural Bloom Filters"
  - Mitzenmacher (2018) "Learned Bloom Filters"
  - McDougall et al. (2023) "Copy Suppression"
  - Merullo et al. (2024) retrieval heads
  - Gould et al. (2023) successor heads
- [x] **IN PROGRESS.** Rewriting Related Work to engage with learned data structure literature and strengthen Wang et al. duplicate-token head discussion.

### 5. "Implement" → "Behave Like" Throughout [Pass 2 #1, Pass 7 #1]
- [x] **IN PROGRESS (sub-agent).** Global replacement in Sections 2–5.
- [x] **IN PROGRESS (sub-agent).** Abstract/intro/conclusion being rewritten separately.

---

## 🟠 TIER 2: Major Revision Triggers (fix before venue submission)

### 6. Selectivity Threshold Mismatch: 3× vs 30× [Pass 8 #17]
- [ ] Methods §3.3 says threshold >3×. Contributions say ">30× over baseline." → Clarify: 3× is the threshold; observed values are 51×–146×. Fix contributions item 1.
  - *Note: abstract/intro sub-agent is addressing this in contributions list.*

### 7. Duplicate-Token Heads Not Distinguished [Pass 2 #2, Pass 3 #8]
- [ ] **New experiment needed.** Run Wang et al.'s IOI duplicate-token head identification procedure on GPT-2 small. Show which heads overlap with our Bloom filter heads. If they overlap, own it and argue the Bloom filter framing adds the capacity theory + independence analysis. If different, show it.
- [ ] Update Related Work with comparison results.
  - *Note: Related Work sub-agent is strengthening the textual discussion, but the experiment still needs doing.*

### 8. Capacity Experiment Confound [Pass 5 #7]
- [ ] **New experiment needed.** Current design varies both unique-token count AND sequence length simultaneously. → Add control: hold sequence length constant at 200 tokens, vary unique-token proportion. Verify FP rate still tracks the Bloom formula.

### 9. r = 0.89 Scaling Claim [Pass 1 #4, Pass 8 #13]
- [x] **FIXED everywhere.** Removed from abstract, contributions, AND §4.5 body text. §4.5 now says "GPT-2 large allocates substantially more heads (27)... though with only four models tested we cannot establish a reliable scaling relationship."
- [ ] **Additional models still desirable.** Testing Pythia suite (8 sizes) would provide real scaling evidence, but the claim is no longer made without it.

### 10. Ablation Lacks CIs and Controls [Pass 1 #9, Pass 8 #15-16]
- [ ] Report bootstrap CIs on ablation deltas (Table 5).
- [ ] Specify control head selection procedure (random? layer-matched? how many?).
- [ ] Address head-count mismatch: 4 Bloom heads vs 16 induction heads. Normalize per-head or match counts.
- [ ] Consider adding a figure showing per-sentence perplexity distributions.

### 11. Implausible p-values [Pass 1 #2]
- [x] **FIXED.** p < 10⁻¹³⁴ values replaced with $\ll 10^{-20}$ throughout Table 1. Caption now notes these are asymptotic approximations capped at $10^{-20}$ since the extreme tail is unreliable at this sample size.

### 12. Ablation Method Unspecified [Pass 5 #9]
- [ ] State whether zero ablation, mean ablation, or resample ablation was used. → Add to §4.6 methods description. Ideally run mean ablation as robustness check.

---

## 🟡 TIER 3: Strengthening (before camera-ready)

### 13. Naturalistic Validation [Pass 5 #8]
- [ ] Run Bloom head analysis on WikiText or OpenWebText with naturally-occurring repetitions. Show the pattern holds beyond constructed stimuli.

### 14. Harold Bloom Framing [Pass 2 #4, Pass 4 #2,5, Pass 7 #6]
- [x] **IN PROGRESS (sub-agent).** Cutting to 1-2 sentences in intro. Removing philosophical paragraph from conclusion.

### 15. R² = 0.99 on 5 Data Points [Pass 1 #5, Pass 6 #4]
- [ ] Report adjusted R². Run capacity curve at finer granularity (every 10 tokens from 5 to 200). Clarify whether R² is computed on 5 table values or ~10 figure values.

### 16. Synonym Source Unspecified [Pass 5 #3]
- [ ] Add to §3.2: how synonyms were chosen (WordNet? LLM? manual?). This is a significant methodology gap.

### 17. Miss Threshold 0.01 Justification [Pass 5 #4]
- [ ] Derive 0.01 from the null distribution of attention values rather than choosing ad hoc. Or: at minimum, show results are robust to threshold choices (0.005, 0.01, 0.02).

### 18. Abstract Quality [Pass 4 #1, Pass 7 #2,8]
- [x] **IN PROGRESS (sub-agent).** Leading with plain English, moving numbers to body.

### 19. Promote Ablation to Contributions [Pass 7 #3]
- [x] **IN PROGRESS (sub-agent).** Adding ablation as contribution #5, demoting cross-model scaling.

### 20. Bibliography Fixes [Pass 9 #2, #12]
- [x] **IN PROGRESS (sub-agent).** Fixing biderman @article→@inproceedings, nanda @article→@misc, citing or removing orphan conmy2023.

### 21. L1H11 Non-Monotonic FP Rates [Pass 6 #9]
- [ ] Table 4 shows 8%→35%→22%→26%→33% — the dip at 50 contradicts Bloom filter theory. → Discuss this anomaly in §4.3 or a footnote. Don't ignore it.

### 22. L5H5 Control in Fig 3 [Pass 6 #5]
- [ ] A non-Bloom head tracks the theoretical curve in the figure but is never discussed in text. → Either discuss (and explain why a non-Bloom head follows the curve — this is potentially damaging to the claim) or remove from figure.

### 23. Fig 7 Layer Range Inconsistency [Pass 6 #6]
- [ ] Table 3 says previous-token heads layers 2–6. Fig 7 shows 2–7. → Check data, align.

### 24. Multiple Prior Occurrences [Pass 5 #14]
- [ ] **New experiment.** If a token appears 3+ times, does the head attend to ALL prior occurrences? This would be strong Bloom filter evidence (membership ≠ recency).

### 25. 16 Induction Heads Seems High [Pass 8 #10]
- [ ] Olsson et al. typically report 4–6 strong induction heads. State threshold used or explain discrepancy.

### 26. Phi Binarization Threshold [Pass 5 #15]
- [ ] §4.4 phi coefficient requires binary FP decisions. What attention threshold turns continuous values into "FP yes/no"? Never stated. Add to methods.

### 27. Observation Count 238 [Pass 1 #6]
- [ ] Where does 238 come from? 100 triplets × 1 repeat = 100 observations per head, not 238. Clarify derivation.

### 28. Position Confound [Pass 5 #1]
- [ ] Repeated token always appears as second occurrence. Need control for positional relationship (same gap, different tokens).

### 29. Token Frequency Control [Pass 5 #6]
- [ ] No control for word frequency in stimuli. Report frequency distribution or stratify results.

### 30. Cross-Model Tokenization [Pass 11 #6]
- [ ] Different tokenizers (GPT-2 vs Pythia) may tokenize stimuli differently. Was this controlled?

### 31. Reproducibility Package [Pass 11 #1-3]
- [ ] Release: 100 sentence triplets, analysis code (TransformerLens scripts), raw attention matrices.
- [ ] Specify stimulus generation procedure (hand-written? template? LLM-generated?).
- [ ] Define "content word" (nouns only? verbs? adjectives?).
- [ ] Prepare NeurIPS reproducibility checklist.

### 32. Writing Polish [Pass 4]
- [ ] Fix "surgeon/doctor" synonym example in Appendix A — "doctor" is a hypernym of "surgeon," not a synonym. Use "physician."
- [ ] Fix \textsc{true} → \texttt{true} or italicize
- [ ] Standardize "near-miss" vs "near miss"
- [ ] Make Discussion §5.1 paragraph lengths less formulaic
- [ ] Label error bars in Fig 2 (SD? SE? 95% CI?)
- [ ] Add a figure for ablation results (Table 5)

### 33. Randomly Initialized Control [Pass 5 #13, Pass 11 #13]
- [ ] Report: same architecture? same stimulus set? random seed? → Run 5 seeds, report mean ± SD of detected Bloom heads.

---

## Estimated Timeline

| Phase | Items | Time |
|-------|-------|------|
| **Now (sub-agents)** | #3, #4, #5, #6 (partial), #9 (partial), #14, #18, #19, #20 | In progress |
| **Day 1** | #1 (text-figure), #2 (miss rate), #11 (p-values) | 3-4 hours |
| **Day 2** | #7 (dup-token experiment), #8 (capacity confound) | 6-8 hours |
| **Day 3** | #10 (ablation CIs), #12 (ablation method), #15 (R²) | 4-6 hours |
| **Week 2** | #13 (naturalistic), #24 (multi-occurrence), #31 (reproducibility) | 2-3 days |
| **Polish** | Everything else | 1-2 days |

---

*Last updated: 2026-02-17*
*Source: review-notes.md passes 1–11*
