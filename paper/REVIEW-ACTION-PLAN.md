# Review Action Plan — "The Anxiety of Influence"

Consolidated from 18 review passes (2026-02-16 to 2026-02-17).
Each item references the pass number for full context in `review-notes.md`.

---

## 🔴 Critical (Must fix before arXiv)

### 1. Run competing model fits on capacity curve — THE make-or-break experiment
- **What's wrong:** The Bloom filter capacity curve ($R^2=0.99$) is the paper's central quantitative claim, but simpler models (softmax dilution, logistic, power law) may fit equally well. If they do, the entire Bloom filter interpretation is "unfalsifiable ornamentation." (Pass 12, 13, 14)
- **Where:** §4.3, Fig 3, Table 4
- **Fix:** Fit 5 competing models (Bloom, softmax dilution 1-param, logistic 3-param, power law 2-param, linear 2-param) on L3H0 + all 4 heads. Compare AIC/BIC. Bootstrap 1000× for robustness. Plot residuals. Run at finer granularity (every 10 tokens, ~20 data points) for better discrimination.
- **Estimated time:** 4–6 hours
- **Impact:** Determines whether paper keeps its title or gets reframed entirely

### 2. Fix text-figure contradictions (3 remaining)
- **What's wrong:** Fig 4 φ values don't match text (text said 0.02–0.38, figure shows max 0.18). Fig 6 FP distribution doesn't match text (text said 63%/17%, figure shows 44%/24%). These are now partially fixed per Pass 15, but verify ALL numbers match after any figure regeneration.
- **Where:** §4.4, Fig 4, Fig 6
- **Fix:** Already mostly fixed per Pass 15. Do a final cross-check of every number in §4.4 against the actual figures. (Pass 6 #1–3, Pass 8 #1–3)
- **Estimated time:** 30 min verification pass
- **Status:** ✅ Mostly fixed — verify final state

### 3. Abstract still cherry-picks the 0.17% combined FP without the independence caveat
- **What's wrong:** Fig 5 (now included) shows observed combined FP is ~60× worse than independence predicts. Abstract still presents the flattering 0.17% without this context. (Pass 16 #3)
- **Where:** Abstract
- **Fix:** Either add "though exceeding the independence prediction by two orders of magnitude" or remove the 0.17% from abstract entirely.
- **Estimated time:** 15 min

### 4. Abstract framing not updated for learned-index literature
- **What's wrong:** Citations for Kraska/Rae/Mitzenmacher were added to Related Work, but the abstract's closing ("gradient descent converges on a fifty-year-old data structure, without explicit design") doesn't acknowledge this prior work exists. (Pass 16 #5)
- **Where:** Abstract, final sentence
- **Fix:** Revise to something like: "...converges on a solution that quantitatively matches a classical data structure — not through explicit optimization for this objective (as in the learned index literature), but as an emergent subroutine of language modeling."
- **Estimated time:** 15 min

### 5. No code/data availability statement
- **What's wrong:** The 100 sentence triplets ARE the experiment. Without them, nothing is reproducible. No code, data, or supplementary materials referenced. NeurIPS requires this. (Pass 11 #1, Pass 15, Pass 16 #12)
- **Where:** New section before References, or end of paper
- **Fix:** Create GitHub repo with: stimulus set (100 triplets), analysis scripts, raw attention data. Add 3-sentence availability statement to paper.
- **Estimated time:** 2 hours (repo setup + statement)

### 6. §4.3 miss rate sentence still missing L3H0 caveat
- **What's wrong:** "miss rates remain near zero across all capacity levels for all Bloom heads" — needs parenthetical about L3H0's single miss not recurring. (Pass 15, Pass 16 #11)
- **Where:** §4.3
- **Fix:** Add "(L3H0's single miss at the base load does not recur at higher loads)"
- **Estimated time:** 5 min

### 7. Cohen's d p-value near combinatorial floor — reframe
- **What's wrong:** $p = 5.8 \times 10^{-8}$ is the exact minimum for Mann-Whitney with 4 vs 140. More informative to state: "the four Bloom filter heads occupy the four highest ranks among all 144 heads." (Pass 8 #8, Pass 16 #10, Pass 18 #15)
- **Where:** §4.1
- **Fix:** Restate as rank-based finding + report as $p < 10^{-7}$.
- **Estimated time:** 10 min

---

## 🟡 Important (Should fix)

### 8. Duplicate-token head overlap never tested empirically
- **What's wrong:** Wang et al. (2022) identified duplicate-token heads. Paper discusses the distinction verbally but never runs the IOI circuit analysis to check overlap. Primary "isn't this just a rename?" vulnerability. (Pass 2 #2, Pass 3 #8, Pass 7 #7, Pass 13 W2, Pass 15 item A)
- **Where:** §6 Related Work, new analysis
- **Fix:** Run IOI duplicate-token head identification on GPT-2 small. Report which (if any) of L0H1, L0H5, L1H11, L3H0 overlap. If they overlap, argue Bloom filter framing adds something (capacity theory, independence, resolution profiles). If different, show it.
- **Estimated time:** 3–4 hours

### 9. Capacity experiment confound — unique tokens conflated with sequence length
- **What's wrong:** "Varying context size from 5 to 200 unique tokens" changes both filter load AND sequence length, positional effects, softmax width. Massive confound. (Pass 5 #7, Pass 15 item B)
- **Where:** §4.3
- **Fix:** Add control: hold sequence length constant at 200 tokens, vary the proportion that are unique (5, 20, 50, 100, 200 unique out of 200 total). If FP rate still tracks Bloom formula, confound is ruled out.
- **Estimated time:** 3–4 hours

### 10. Ablation section tells two contradictory stories
- **What's wrong:** Zero ablation says repeat-specific (+14.6% interaction). Mean ablation says general-purpose (interaction = −3.7%). Paper hand-waves: "specialization coexists with broader roles." Reviewer will ask which method you trust. (Pass 16 #1)
- **Where:** §4.6, Contribution #5
- **Fix:** State clearly that mean ablation is the more principled method. Make mean-ablation the primary result. Relegate zero ablation to supplementary or footnote. Reframe contribution #5 accordingly.
- **Estimated time:** 45 min

### 11. L0H1/L0H5 heterogeneity undermines independence analysis
- **What's wrong:** §4.3 says L0H1/L0H5 resemble "perfect hash tables" (FP ≈ 1%). §4.4 treats all 4 heads as equivalent "independent hash functions." But AND-combining a near-perfect detector with a capacity-limited one isn't combining hash functions of the same filter. (Pass 16 #2)
- **Where:** §4.4
- **Fix:** Acknowledge the heterogeneity. Note that the two "subtypes" contribute differently to the AND-combination and that the low combined FP is dominated by the near-perfect heads.
- **Estimated time:** 30 min

### 12. Optimal-k analysis — major missed opportunity
- **What's wrong:** Bloom filter theory predicts optimal $k^* = (m/n) \ln 2$. For $m=59$, typical $n=20$: $k^* = 2.05$, almost exactly matching fitted $k=2.16$. This is a *quantitative prediction* no competing model makes — far stronger evidence than the capacity curve alone. Paper completely misses this. (Pass 18 #9)
- **Where:** New analysis in §4.3
- **Fix:** Compute optimal k for typical context lengths. Show fitted k matches. This could be the single strongest argument for Bloom filter interpretation.
- **Estimated time:** 1 hour

### 13. $k$ parameter inconsistency: within-head (2.16) vs between-head (4)
- **What's wrong:** §4.3 fits $k=2.16$ per head. §4.4 says 4 heads = 4 "independent hash functions" (implying $k=4$). These are different $k$'s. Never reconciled. (Pass 18 #1)
- **Where:** §4.3, §4.4
- **Fix:** Explicitly distinguish: "within-head $k \approx 2$ (internal hash resolution)" vs "system-level $K=4$ (number of parallel filters)." Discuss whether total effective hash functions = $k \times K \approx 8$.
- **Estimated time:** 30 min

### 14. Two different FP metrics used interchangeably
- **What's wrong:** §4.1 uses FP ratio (continuous: $\bar{a}_\text{synonym}/\bar{a}_\text{hit}$). §4.3/§4.4 use FP rate (binary: fraction exceeding threshold). Same name, different metrics. (Pass 18 #5)
- **Where:** §3.3, throughout
- **Fix:** Use "FP ratio" for continuous, "FP rate" for binary. Define both in §3.3.
- **Estimated time:** 20 min

### 15. Synonym source unspecified
- **What's wrong:** "Semantic near-miss" synonyms — chosen how? WordNet? LLM? Author judgment? Uncontrolled and potentially circular. (Pass 5 #3, Pass 15 item F)
- **Where:** §3.2
- **Fix:** State the source. If WordNet, say so. If manual, acknowledge subjectivity.
- **Estimated time:** 10 min

### 16. Retrieval heads overlap not tested
- **What's wrong:** Wu et al. (2024) retrieval heads may overlap with Bloom filter heads. Currently claimed as distinct but never empirically tested. (Pass 16 #4)
- **Where:** §6 Related Work
- **Fix:** Add sentence acknowledging overlap hasn't been tested empirically, or run the comparison.
- **Estimated time:** 10 min (text) or 2 hours (experiment)

### 17. Add QK projection caveat to hash resolution section
- **What's wrong:** §5 measures cosine similarity in INPUT embedding space, but heads operate in QK-projected space. For layer-0 heads this is ~equivalent, but for L3H0 (layer 3) the residual stream has been transformed by 3 layers. (Pass 18 #2, #10)
- **Where:** §5
- **Fix:** Note this limitation. State that input-space similarity is a proxy; the analysis is a lower bound on the true relationship in QK space.
- **Estimated time:** 15 min

---

## 🟢 Nice-to-have (Polish)

### 18. L1H11 non-monotonic FP rates undiscussed
- Table 4: 8%→35%→22%→26%→33%. The dip at 50 tokens contradicts Bloom filter theory. (Pass 6 #9)
- **Fix:** Add 1–2 sentences discussing possible causes (statistical noise, capacity regime transition).
- **Time:** 10 min

### 19. L5H5 control in Fig 3 unexplained
- A non-Bloom head tracks the theoretical curve. If other heads also fit, Bloom specificity weakens. (Pass 6 #5)
- **Fix:** Add sentence explaining why this head is shown and why it doesn't invalidate the claim.
- **Time:** 10 min

### 20. Fig 7 layer range: "2–7" vs "2–6"
- Table 3 says previous-token heads in layers 2–6; figure says 2–7. (Pass 6 #6)
- **Fix:** Check data, align.
- **Time:** 5 min

### 21. $R^2$: ordinary or adjusted? Report adjusted.
- With 2 params on 5–10 points, adjusted $R^2$ could be notably lower. (Pass 1 #5, Pass 18 #12)
- **Time:** 10 min

### 22. Add error bars to Fig 8 (similarity sweep)
- Currently no CIs on the resolution profiles. (Pass 17 #11)
- **Time:** 30 min

### 23. Fit sigmoid to resolution profiles for formal bandwidth metric
- "Ultra-precise" vs "broad" is currently subjective. (Pass 17 #4)
- **Time:** 1 hour

### 24. Fix Appendix A synonym example: "surgeon/doctor" is hypernym, not synonym
- Use a true synonym pair (e.g., "physician/doctor"). (Pass 4 #14)
- **Time:** 5 min

### 25. $\phi$ coefficient: add formula or citation
- Used but never formally defined. (Pass 18 #11)
- **Time:** 5 min

### 26. Bootstrap scheme: specify resampling unit
- Sentence-level vs token-level matters when baseline ≈ 0. (Pass 18 #4)
- **Time:** 10 min

### 27. Add "behavioral characterization" caveat at start of Results
- §2.2 disclaimer is too early; readers forget by Results. (Pass 16 #7)
- **Time:** 5 min

### 28. Reproducibility checklist preparation
- NeurIPS requires it. Most answers are straightforward. (Pass 11 #8)
- **Time:** 1 hour

### 29. Biderman bib entry: @article → @inproceedings
- (Pass 9 #12)
- **Time:** 2 min

### 30. Conmy2023 orphan bib entry: cite or remove
- (Pass 9 #2)
- **Time:** 5 min

### 31. Multiple random seeds for initialized control
- Currently one seed. Run 5 and report mean ± SD. (Pass 11 #13)
- **Time:** 30 min

### 32. Naturalistic validation on WikiText/OpenWebText
- All results on constructed stimuli. Natural text would strengthen generalizability. (Pass 5 #8)
- **Time:** 4–6 hours

### 33. Test attention to multiple prior occurrences
- If token appears 3+ times, does head attend to ALL priors? Strong Bloom filter evidence. (Pass 5 #14)
- **Time:** 2 hours

---

## 🧪 Experiments Needed

### EXP-1: Competing model fits (HIGHEST PRIORITY) ⭐
- **Design:** Pass 14, full specification
- Fit Bloom, softmax-dilution, logistic, power law, linear to capacity data
- Run on all 4 heads, not just L3H0
- Finer granularity (every 10 tokens, 5–200)
- Compare AIC/BIC, bootstrap 1000×, plot residuals
- **Outcome determines paper framing.** If Bloom wins → strengthen. If ties → pivot.
- **Time:** 4–6 hours
- **Blocks:** Everything else (framing decisions depend on this)

### EXP-2: Optimal-k prediction test
- Compute $k^* = (m/n) \ln 2$ for typical context lengths
- Compare to fitted $k = 2.16$
- If match → strongest single evidence for Bloom interpretation
- **Time:** 1 hour
- **Depends on:** EXP-1 data

### EXP-3: Duplicate-token head overlap test
- Run IOI duplicate-token head identification (Wang et al. 2022 procedure) on GPT-2 small
- Report overlap with L0H1, L0H5, L1H11, L3H0
- **Time:** 3–4 hours

### EXP-4: Constant-length capacity control
- Hold sequence at 200 tokens, vary unique proportion
- Rules out the sequence-length confound
- **Time:** 3–4 hours
- **Depends on:** EXP-1 (same data pipeline)

### EXP-5: Naturalistic validation
- Run Bloom head analysis on WikiText-103 / OpenWebText passages with natural repetitions
- Show selectivity holds outside constructed stimuli
- **Time:** 4–6 hours

### EXP-6: Multi-occurrence attention test
- Tokens appearing 3+ times: does head attend to ALL priors equally?
- Bloom filters don't distinguish members → all priors should get equal attention
- **Time:** 2 hours

### EXP-7 (stretch): QK subspace analysis
- Examine QK weight matrices for hashing-like structure
- Would upgrade "behaves like" to "mechanistically implements"
- **Time:** 8+ hours
- **Impact:** Transforms paper from borderline-reject to strong accept

---

## 📊 Summary Stats

| Metric | Count |
|--------|-------|
| Total issues found across 18 passes | ~85 |
| Fixed (verified in Pass 15/16) | ~35 |
| Remaining critical (🔴) | 7 |
| Remaining important (🟡) | 10 |
| Remaining nice-to-have (🟢) | 16 |
| Experiments needed | 7 (4 high-priority) |
| Text-figure contradictions found | 3 (mostly fixed) |
| Factual errors found | 5 (mostly fixed) |

**Simulated NeurIPS score: 4/10 (borderline reject)**
- Path to 5–6: Fix all 🔴 items + EXP-1
- Path to 7+: All above + EXP-2,3,4,5 + QK analysis

**Estimated remaining revision time:**
- Text fixes only (🔴 + 🟡 writing): ~5 hours
- Essential experiments (EXP-1,2,3,4): ~12 hours
- Full revision (everything): ~3–4 focused days

---

## 🗓️ Suggested Revision Schedule

### Day 1: The Make-or-Break Day
| Order | Task | Time | Blocks |
|-------|------|------|--------|
| 1 | **EXP-1: Competing model fits** | 4–6h | Everything — outcome determines framing |
| 2 | **EXP-2: Optimal-k analysis** | 1h | Uses EXP-1 data |
| 3 | Evaluate results → decide: keep Bloom framing or pivot | 30min | — |

**If Bloom wins:** proceed with strengthening. **If not:** reframe paper as "repeated-token detection heads with capacity limitations" and adjust title/abstract/conclusion.

### Day 2: Close the Gaps
| Order | Task | Time | Depends on |
|-------|------|------|------------|
| 4 | **EXP-3: Duplicate-token head overlap** | 3–4h | Independent |
| 5 | **EXP-4: Constant-length capacity control** | 3–4h | EXP-1 pipeline |
| 6 | All 🔴 text fixes (#2–7) | 1.5h | EXP-1 outcome (for framing) |
| 7 | 🟡 items #10–17 (text fixes) | 2h | Independent |

### Day 3: Polish & Package
| Order | Task | Time | Depends on |
|-------|------|------|------------|
| 8 | Code/data repo setup (#5) | 2h | Independent |
| 9 | 🟢 items #18–31 | 2h | Independent |
| 10 | Reproducibility checklist (#28) | 1h | All experiments done |
| 11 | Fresh read-through for consistency | 1.5h | Everything |
| 12 | Final compile + figure check | 30min | Everything |

### Day 4 (if available): Stretch Goals
- EXP-5: Naturalistic validation (4–6h)
- EXP-6: Multi-occurrence test (2h)
- EXP-7: QK subspace analysis (8h) — only if going for top venue

### Key Dependencies
```
EXP-1 (model fits) ──→ Framing decision ──→ All text revisions
                   ──→ EXP-2 (optimal-k)
                   ──→ EXP-4 (constant-length)
EXP-3 (dup-token) ──→ Related Work revision
Code repo (#5)    ──→ Availability statement
All experiments    ──→ Reproducibility checklist
```

---

## Top 3 Priorities (TL;DR)

1. **Run competing model fits (EXP-1).** This single experiment determines whether the paper's central claim survives. If softmax dilution fits as well as the Bloom formula, the paper needs major reframing. If Bloom wins on AIC/BIC, the paper is dramatically strengthened. Do this first; everything else is secondary.

2. **Fix the abstract.** It still cherry-picks (0.17% without independence caveat), overpromises (unvalidated applications), and doesn't reflect the learned-index literature now cited in the body. The abstract is what 90% of readers see. 30 minutes of rewriting here has more impact than hours elsewhere.

3. **Run the duplicate-token head overlap test (EXP-3).** "Aren't these just Wang et al.'s duplicate-token heads with a new name?" is the single most likely first question from any reviewer familiar with mechanistic interpretability. An empirical answer — overlap or not — closes this gap decisively.
