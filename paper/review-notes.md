# Review Notes: "The Anxiety of Influence"

Critical review notes for arXiv submission readiness.
Each review pass is dated and appended below.

---

## 2026-02-16 — Pass 1: Statistical Claims

**Focus:** Are all numbers, p-values, effect sizes, and statistical procedures defensible?

### Critical Issues

1. **L3H0 miss rate inconsistency.** Table 1 says L3H0 has miss rate 0.4%, but the text in §4.1 says "All four have 0% miss rate (0/238 observations)." These contradict each other. If 0.4% is correct on 238 obs, that's ~1 miss — which is NOT 0%. Fix the prose or explain the discrepancy. Also: §4.3 says "miss rates remain at 0% across all capacity levels for all Bloom heads" — again inconsistent with 0.4%.

2. **p-values reported as $< 10^{-134}$.** These are absurdly small and likely artifacts of asymptotic approximations in Mann-Whitney U with n=238. At sample sizes this small, the exact minimum achievable p-value for Mann-Whitney is much larger than $10^{-134}$. Either (a) the test is being run on pooled/expanded data not described in the methods, or (b) the software is computing an asymptotic approximation that's unreliable at these extremes. Report exact permutation p-values or cap at $< 10^{-20}$ with a note. Reviewers WILL flag this.

3. **Cohen's d = 12.3 and 12.5.** These are extraordinary effect sizes (anything above 2 is "huge"). While they may be real given the bimodal distribution, reporting them without comment invites skepticism. Add a sentence acknowledging these are extreme and explaining why (bimodal population: 4 Bloom heads vs 140 near-zero heads).

4. **Correlation $r = 0.89$ for count scaling with model size.** This is computed on **four data points** (4 models). An r of 0.89 on n=4 is not statistically significant at conventional thresholds (critical r ≈ 0.95 for α=0.05, df=2). Do not present this as evidence of scaling without a massive caveat or more models. Currently reads as a confident finding — it isn't.

5. **Capacity curve $R^2 = 0.99$.** How many data points? The text says "5 to 200 unique tokens" and Table 4 shows 5 data points. An R² of 0.99 on 5 points with a 3-parameter model (m, k, and the implicit proportionality) is not impressive — it's nearly saturated. Report adjusted R² or, better, run the curve at finer granularity (every 10 tokens) to make this convincing.

6. **"0/238 observations" for miss rate.** Where does 238 come from? 100 sentence triplets × 1 repeat condition = 100 observations per head, not 238. Clarify the observation count derivation.

### Minor Issues

7. **Bonferroni correction.** Using α = 0.05/144 is appropriate for GPT-2 small. But the cross-model results (§4.5) don't mention any correction for testing across 4 models × varying head counts. 

8. **Bootstrap CIs on selectivity.** The CIs in Table 1 are very wide (e.g., 105–201 for L0H1). This is fine to report but worth acknowledging — the ratio is noisy because baseline attention is near-zero (dividing by ~0.003).

9. **Ablation lacks CIs or significance tests.** Table 5 reports point estimates (+9.7%, -4.4%) with no confidence intervals, standard errors, or p-values. How many sentences? What's the variance? The "2.2× stronger than control" claim is untested.

10. **FP rate of 0.17% for AND-combination.** Is this theoretical (product of individual rates assuming independence) or empirical? If theoretical, the φ=0.38 correlation between L0H1↔L0H5 means the independence assumption doesn't hold perfectly. If empirical, report CI.

### Summary

The headline findings are likely real — the selectivity differences are enormous and visually obvious. But the statistical reporting has several soft spots that a careful reviewer will exploit. The most damaging are: the L3H0 miss rate contradiction (#1), the implausible p-values (#2), and the n=4 correlation presented without caveats (#4). Fix these before arXiv.

**Next pass suggestion:** Logical flow / argument structure.

---

## 2026-02-16 — Pass 2: Logical Flow & Argument Structure

**Focus:** Does the argument build coherently? Any unjustified leaps?

### Critical Issues

1. **The central analogy is never stress-tested.** The paper claims heads "implement" Bloom filters, but the argument is: (a) these heads attend to repeated tokens, (b) they have low miss rates and some false positives, (c) one head's FP curve fits the Bloom filter capacity formula. That's a behavioral analogy, not a mechanistic identification. A Bloom filter has a specific internal structure (bit array + hash functions). The paper never examines QK weight matrices to show anything resembling hashing into a bit array. §5.2 acknowledges this ("behavioral criteria rather than mechanistic analysis") but it's buried in Limitations. The gap between "behaves like" and "implements" is the paper's central logical vulnerability, and it needs to be confronted head-on, not footnoted. A reviewer could argue these are simply "duplicate-token heads" (which already exist in the literature — see Wang et al. 2022) that happen to have a monotonic FP-vs-load curve, which any saturating detector would show.

2. **Duplicate-token heads vs. Bloom filter heads — insufficiently distinguished.** §2.3 mentions Wang et al.'s duplicate-token heads and says they were "not characterized as Bloom filters." But the paper never shows that Bloom filter heads are doing something *functionally different* from duplicate-token heads. The new contribution appears to be (a) the Bloom filter framing/analogy and (b) the capacity curve fit. If a reviewer asks "aren't these just duplicate-token heads with a new name?", the current text has no strong rebuttal. You need an explicit comparison: run the IOI duplicate-token head identification procedure, show which heads overlap, and articulate what Bloom filter heads do that duplicate-token heads don't (or vice versa).

3. **"Independent hash functions" claim is a logical stretch.** §4.4 argues the four heads act as independent hash functions because their FP decisions have low φ correlation. But low correlation between detectors doesn't imply they're implementing hash functions — it just means they're making errors on different inputs. Any four weakly-correlated binary classifiers would show this. The hash function framing adds no explanatory power beyond "they're somewhat independent detectors." The AND-combination result is interesting on its own without the hash function claim.

4. **The Harold Bloom framing does real argumentative damage.** The intro spends a full paragraph on Harold Bloom's "Anxiety of Influence" and the conclusion returns to it. This is charming but it creates a logical problem: the paper implies that what these heads do is somehow analogous to literary influence anxiety, but the actual mechanism is just "detect repeated tokens." The literary parallel suggests something deeper about meaning-making that the experiments don't support. A hostile reviewer will read the Harold Bloom material as dressing up a modest finding. Consider: keep the title (it's catchy), drastically cut the Harold Bloom discussion in the intro to 1-2 sentences, and remove the concluding paragraph about Harold Bloom entirely. Let the Burton Bloom results speak for themselves.

5. **Logical gap: early-layer concentration is presented as confirming a "pipeline" but the pipeline is never established.** §4.2 says Bloom heads in layers 0–3 is "consistent with a processing pipeline where membership testing precedes pattern completion." But this is circular — you'd need to show that downstream heads *use* the membership signal from Bloom heads. The ablation (§4.6) partially addresses this but doesn't trace information flow. Without a causal circuit analysis showing Bloom head outputs feed into induction heads, the "pipeline" language overstates the evidence.

6. **The Discussion oversells practical applications.** §5.1 proposes sparse attention replacement, KV-cache compression, hallucination detection, and head pruning — four significant engineering claims, none with any experimental validation. This is fine as speculation but the paragraph framing ("several optimizations become available") implies they're ready to use. Reframe as "potential directions" and add a sentence noting none have been validated.

### Minor Issues

7. **Section ordering.** Related Work (§6) comes after Discussion (§5), which is unusual for ML papers (typically Related Work is §2 or early). This means the reader encounters the duplicate-token head comparison very late. Moving Related Work earlier — or at minimum, expanding §2.3 — would let you preempt the "isn't this just duplicate-token heads?" objection before presenting results.

8. **The randomly-initialized model control (end of §4.1)** is good but needs more detail. Was it the same architecture? Same stimulus set? How many heads showed selectivity > 3×? A single sentence isn't enough for a control experiment.

9. **Contributions list (end of §1) doesn't mention ablation.** The ablation is arguably the strongest evidence for functional specificity and deserves a bullet point.

### Summary

The paper's biggest structural weakness is the gap between the behavioral evidence and the mechanistic claim. "These heads behave like Bloom filters" is well-supported. "These heads implement Bloom filters" is not — and that's the claim the title, abstract, and framing make. Either add mechanistic evidence (QK circuit analysis) or soften the language throughout. The duplicate-token head distinction is the second major gap — a reviewer familiar with Wang et al. will see it immediately.

**Next pass suggestion:** Related work / missing citations.

---

## 2026-02-17 — Pass 3: Related Work & Missing Citations

**Focus:** Are there obvious citations missing? Claims that contradict or ignore known results?

### Critical Omissions

1. **Kraska et al. (2018) "The Case for Learned Index Structures" — MUST CITE.** This is the most important missing reference. Kraska et al. showed that neural networks can learn to replace classical data structures including Bloom filters. Your paper's central claim — that gradient descent independently arrives at a Bloom filter — is a *direct descendant* of this line of work. Not citing it will look like either ignorance or avoidance. Also cite the follow-up: Rae et al. (2019) "Meta-Learning Neural Bloom Filters" (ICML), which explicitly trains neural networks as Bloom filter replacements. The framing "gradient descent independently arrived at a fifty-year-old data structure" is significantly weakened if there's a whole literature on neural networks learning data structures — it's less surprising than the paper implies.

2. **Mitzenmacher (2018) "A Model for Learned Bloom Filters and Optimizing by Sandwiching" (NeurIPS).** Provides theoretical analysis of when learned Bloom filters outperform classical ones. Directly relevant to your capacity analysis in §4.3.

3. **The IAAR-Shanghai "Awesome Attention Heads" survey and associated paper.** There is now a comprehensive survey/taxonomy of attention head types (2024-2025, continuously updated). Your claim of establishing "a new functional category in the attention head taxonomy" needs to engage with this broader taxonomy effort. The survey catalogs dozens of head types beyond induction/previous-token/duplicate-token. If your Bloom filter heads overlap with any cataloged type (e.g., "retrieval heads," "copy heads"), a reviewer who knows this literature will notice.

4. **McDougall et al. (2023) "Copy Suppression: Comprehensively Understanding an Attention Head."** Analyzes a single attention head in GPT-2 small in mechanistic detail — exactly the level of circuit analysis your paper lacks. Cite as an example of the kind of mechanistic work that would strengthen the "implements" claim, and as methodological precedent.

5. **Conmy et al. (2023) "Towards Automated Circuit Discovery for Mechanistic Interpretability" (ACDC).** Your manual head classification approach could be compared against automated circuit discovery methods. Reviewers familiar with ACDC will wonder why you didn't use it.

6. **Merullo et al. (2024) and related work on retrieval heads.** Recent work identifies "retrieval heads" that are responsible for extracting information from context. These may overlap with or subsume your Bloom filter heads. If retrieval heads and Bloom filter heads are the same heads, that's a problem for novelty. If they're different, that's an important distinction to make explicitly.

### Claims That May Contradict Known Results

7. **"Zero overlap" between Bloom filter heads and induction heads.** Olsson et al. (2022) describe a two-head circuit where the first head (which they call a "previous token head" but which functions as a matching/lookup head) feeds into the induction head. Your Bloom filter heads in layers 0–3 could be exactly these "first heads" in the induction circuit, just measured with a different test. The paper needs to explicitly rule this out — run the induction *circuit* test (not just the induction head score) and show Bloom heads aren't the upstream component of induction circuits.

8. **"Taxonomically distinct from duplicate-token heads."** This was flagged in Pass 2 but deserves re-emphasis as a citation issue. Wang et al. (2022) identified duplicate-token heads in the IOI circuit. Your §2.3 waves this away in one sentence. You need to either (a) run the IOI circuit analysis and show your heads aren't the same ones, or (b) acknowledge they likely overlap and argue the Bloom filter framing adds something. Currently the paper pretends this prior work is minor — it isn't.

9. **GPT-2 is heavily studied.** Every head in GPT-2 small has been analyzed by dozens of papers. If L0H1, L0H5, L1H11, and L3H0 haven't been identified before as noteworthy, that's actually suspicious — or they have been identified under different names. Search TransformerLens documentation, Neel Nanda's blog posts, and the Anthropic transformer circuits thread for any prior characterization of these specific heads. If they've been discussed before, you must cite it.

### Minor Gaps

10. **No citation for the "randomly initialized model" control.** Is there precedent for this control in the mech interp literature? Cite it or describe methodology more fully.

11. **Missing: Gould et al. (2023) "Successor Heads."** Another functional head category discovered in transformers. Relevant to completeness of the taxonomy discussion.

12. **Missing: Quirke & Barez (2023) on training dynamics of induction heads.** Could inform your discussion of when/how Bloom filter heads emerge during training — a question the paper doesn't address but reviewers will ask.

13. **The Harold Bloom citation (1973) is to a literary criticism monograph.** While charming, some ML venues will view this as padding the bibliography. More importantly, if you're going to invoke Harold Bloom's theory, cite it correctly and engage with it — the current treatment is superficial. "The Anxiety of Influence" argues poets *misread* their predecessors as a creative act. Your heads don't misread anything — they accurately detect repetition. The analogy actually undermines itself on close inspection.

### Summary

The related work section is thin for a paper making a strong novelty claim. The most damaging omission is the learned index / neural Bloom filter literature (Kraska, Rae, Mitzenmacher) — this directly contextualizes and partially deflates the "gradient descent reinvented a data structure" narrative. The duplicate-token head overlap remains unresolved. And the explosion of attention head taxonomy work in 2024–2025 means the paper's claim of "a new functional category" needs to be defended against a much larger landscape than the three categories currently cited.

**Next pass suggestion:** Writing quality / AI-detectable patterns.

---

## 2026-02-17 — Pass 4: Writing Quality & AI-Detectable Patterns

**Focus:** Awkward phrasing, unclear sentences, overwriting, and patterns that read as AI-generated.

### Significant Issues

1. **The abstract is 189 words of dense results-dumping.** It front-loads jargon ($R^2 = 0.99$, $\bar{\phi} = 0.13$, fitted $m = 59$) that means nothing without context. Abstracts should tell a story: problem → approach → key finding → implication. Restructure: sentence 1 = what you found (plain English), sentence 2 = how you found it, sentence 3 = strongest single result, sentence 4 = why it matters. Move the numerical details to the body. The final sentence ("Or, to make the case more directly:") is an unusual rhetorical move in an abstract — it reads as the author breaking character, which is either charming or unprofessional depending on the reviewer.

2. **The intro's second paragraph is overwritten and unfocused.** It attempts to connect Burton Bloom, Harold Bloom, Mark Twain, coreference, surprisal, and the attention mechanism in a single paragraph. The result is dense, allusive, and hard to follow on first read. The Mark Twain parenthetical ("*Pace* Mark Twain...") adds nothing. The sentence beginning "This is the core insight that connects Burton Bloom's engineering and Harold Bloom's literary theory" is doing too much work — the reader hasn't been told why Harold Bloom is relevant yet. Consider: cut this paragraph to 3-4 sentences max, state the key idea plainly (transformers need to know what they've seen before), then move to the contribution.

3. **"*Pace* Mark Twain" is pretentious.** Using Latin rhetorical markers in an ML paper signals either erudition or affectation — most readers will land on the latter. Cut it.

4. **Inconsistent voice between sections.** The intro and conclusion are literary and discursive. The methods and results are terse and technical. This tonal whiplash is jarring. The Discussion (§5) is somewhere in between. Pick a register and maintain it. Recommendation: keep the technical sections as-is, tone down the literary flourishes in intro/conclusion to 1-2 moments max (the title is already doing this work — you don't need to keep re-justifying the Harold Bloom connection).

5. **Conclusion's final paragraph is purple prose.** "Every text, Harold Bloom argued, is a response to what came before..." through "The anxiety of influence is not, it seems, a uniquely human affliction." This reads like it was written to be quotable rather than informative. It's the kind of paragraph that divides reviewers — some will love it, others will see it as padding. Given that you're targeting NeurIPS (per the style file), lean toward restraint. The last sentence in particular ("Gradient descent appears to agree: given no blueprints and no prior knowledge, it reinvents Burton Bloom's 1970 filter as if it had no choice") is a strong closer — cut everything before it in that paragraph and use it as the final sentence after a brief factual summary.

6. **Repeated claim phrasing.** The phrase "gradient descent independently arrived at / reinvented a fifty-year-old data structure" appears in both the abstract and the conclusion in near-identical form. Once is a thesis statement; twice is a slogan. Vary the phrasing or cut one instance.

7. **"Or, to make the case more directly:" in the abstract.** This is a meta-commentary on the paper's own rhetoric *inside the abstract*. It breaks the fourth wall in a way that's unusual for academic writing. Either commit to the direct version (cut the preceding dense sentence) or commit to the technical version (cut this pivot). Having both reads as hedging.

8. **AI-detectable patterns.** The paper is generally well-written and doesn't have the usual AI tells (no "delve," no "it's important to note," no "in the rapidly evolving landscape of"). However:
   - The Discussion §5.1 uses a formulaic structure (four bold-headed paragraphs of roughly equal length, each proposing an application) that reads as list-generated. Vary paragraph length and merge related ideas.
   - The phrase "the defining properties of Bloom filters" appears in both the abstract and conclusion — this kind of exact repetition across sections is an AI pattern.
   - The Limitations section is suspiciously well-organized into exactly three limitations of roughly equal weight. Real limitations sections are usually lopsided — one big acknowledged gap and some minor caveats. The current version reads like a checklist.

### Minor Issues

9. **"Near-miss" vs. "near miss."** Used hyphenated in §3.2 ("semantic near-miss") but this isn't standard. "Near miss" (no hyphen) is the standard noun form; "near-miss" as adjective before a noun is acceptable but inconsistent with "semantic near-misses" used as a noun later.

10. **"$\textsc{true}$" in §1.** The `\textsc` command produces small caps, which is an unusual choice for a Boolean value. Use `\texttt{true}` or just "true" in italics.

11. **Table 1 caption:** "All tests significant at Bonferroni-corrected α = 3.47 × 10⁻⁴" — this is fine but the specificity of 3.47 is false precision. Use α ≈ 3.5 × 10⁻⁴ or just say α = 0.05/144.

12. **"That's a behavioral analogy, not a mechanistic identification" is something I wrote in Pass 2, but the paper itself never uses language this clear.** The Limitations section says "behavioral criteria rather than a mechanistic analysis of the QK circuit weights" — this is close but buried. Promote this honesty to the intro or the transition into Results.

13. **§2.2 last sentence:** "We hypothesize that for Bloom filter heads, the QK circuit implements the membership test... while the value circuit is secondary." This hypothesis is never tested. Either test it (compare QK vs. OV circuit contributions) or remove the hypothesis and just describe the attention mechanism.

14. **Appendix A stimulus example uses "surgeon/doctor" as the synonym pair.** "Doctor" is a hypernym of "surgeon," not a synonym. This is a minor but embarrassing inaccuracy in the one example readers will actually look at. Use a true synonym pair (e.g., "physician").

### Summary

The writing is above average for ML papers — the literary framing is distinctive and the technical prose is clear. The main risks are: (1) the abstract tries to do too much, (2) the Harold Bloom material is charming but will alienate some reviewers and currently occupies too much space, (3) a few AI-pattern flags in structural regularity. The single biggest improvement would be rewriting the abstract to lead with a plain-English statement of the finding and saving the numbers for the body.

**Next pass suggestion:** Methodology / experimental design.

---

## 2026-02-17 — Pass 5: Methodology & Experimental Design

**Focus:** Experimental design issues, confounds, missing controls, stimulus validity.

### Critical Issues

1. **Stimulus design confounds position with repetition.** The repeated token always appears as the *second* occurrence of a word. But attention naturally varies by position — tokens later in a sentence attend differently than early ones regardless of repetition. The "baseline" condition uses "attention from a unique token to a random earlier position," which doesn't control for the *specific positional relationship* between first and second occurrence. You need a control where the same positional gap exists but the tokens are different. Without this, selectivity could partly reflect positional patterns rather than membership testing.

2. **100 sentence triplets is underpowered for cross-head comparisons.** You're testing 144 heads (GPT-2 small) with 100 stimuli each. That's fine for identifying the top-4 Bloom heads (the effect is enormous). But the ablation (§4.6), capacity analysis (§4.3), and cross-model comparisons (§4.5) all rest on this same 100-sentence set. For ablation, you're measuring perplexity differences of 4–10% — what's the variance across sentences? With 100 sentences, a few outliers could drive the effect. Report per-sentence distributions, not just means.

3. **Synonym selection is uncontrolled and potentially circular.** The "semantic near-miss" condition uses synonyms, but how were synonyms chosen? By the authors? By WordNet? By an LLM? Different synonym sources produce different similarity levels. If synonyms were chosen by GPT-2-era embeddings, you're measuring GPT-2's own similarity space — circular for claims about what GPT-2 "confuses." The methods section says nothing about synonym source or validation. This is a significant omission.

4. **The 0.01 attention threshold for "miss" is arbitrary.** Miss rate is defined as "fraction of repeated tokens receiving < 0.01 attention to their first occurrence." Why 0.01? Attention in a 128-token sequence has a uniform baseline of ~0.008. So 0.01 is barely above chance. A head could be doing nothing special and still clear this threshold on most tokens just by softmax noise. The threshold needs justification — ideally derived from the null distribution of attention values, not chosen ad hoc.

5. **Selectivity metric is unstable when baseline → 0.** Selectivity = hit/baseline. When baseline attention is 0.003 (Table 1, L0H1), tiny fluctuations in baseline produce huge swings in selectivity. This explains the wide CIs (105–201×). The metric is fine for identifying Bloom heads qualitatively, but the specific selectivity numbers are unreliable. Consider reporting hit − baseline (absolute difference) alongside the ratio, which is more stable.

6. **No control for token frequency.** Repeated tokens in the stimulus set are "content words" but there's no control for frequency. Common words (e.g., "doctor") and rare words (e.g., "ophthalmologist") likely produce different attention patterns. If the 100 stimuli skew toward common words, the results may not generalize to rare tokens. Report the frequency distribution of target words, or stratify results by frequency.

7. **Capacity experiment (§4.3) conflates unique tokens with sequence length.** "Varying context size from 5 to 200 unique tokens" — but does this mean 5-token sequences vs. 200-token sequences? If so, you're varying both the number of unique tokens AND the sequence length, attention distribution width, positional encoding effects, etc. To isolate the "filter filling" effect, you'd need to hold sequence length constant and vary the proportion of unique vs. repeated tokens. The current design has a massive confound.

8. **No naturalistic validation.** All experiments use constructed stimuli. This is acknowledged in Limitations but understated. The constructed sentences have a very specific structure ("The X did Y and the X did Z"). Real text has much more varied repetition patterns — pronouns, partial matches, morphological variants ("run"/"running"), etc. Do Bloom heads respond to these? A validation pass on WikiText or OpenWebText with naturally-occurring repetitions would substantially strengthen the paper.

9. **TransformerLens intervention methodology unstated.** The ablation "zeros out" Bloom filter heads, but how exactly? Zero the attention output? The QK product? The full head output including the residual stream contribution? Different ablation methods (zero, mean, resample) give different results — this is well-established in the mech interp literature (see Chan et al. 2022, Conmy et al. 2023). The paper needs to specify exactly what was zeroed and justify the choice. Mean ablation is generally considered more reliable than zero ablation for causal claims.

10. **"Strong Bloom filter head" threshold is post-hoc.** The criteria (selectivity > 3×, miss rate < 10%, mean hit attention > 0.05) are presented as a definition but are clearly chosen to capture exactly the heads the authors already identified. There's no principled derivation of these thresholds. This isn't necessarily wrong — exploratory analysis is fine — but the paper presents it as a definition rather than a discovery procedure. Be transparent: "we observed a clear bimodal distribution and set thresholds to separate the modes" (if true).

### Minor Issues

11. **No train/test split on stimuli.** If any aspect of the analysis (e.g., threshold selection) was tuned on the same 100 sentences used for reporting, the results are overfit. Even a simple 80/20 split would help.

12. **Induction head test uses "random repeated sequences, n=50 trials" — why only 50?** The Bloom filter analysis uses 100 triplets. Using different sample sizes for different head categories makes the taxonomy comparison uneven.

13. **The randomly-initialized control (end of §4.1) needs more detail.** Same random seed? Multiple seeds? Same tokenizer? Same stimulus set? One sentence isn't a methods description.

14. **No analysis of attention to non-first occurrences.** If "doctor" appears at positions 5, 20, and 35, does position 35 attend to position 5, position 20, or both? Bloom filters don't distinguish between set members — if these heads behave like Bloom filters, they should attend to ALL prior occurrences, not just the first. This is testable and would be strong evidence.

15. **The phi coefficient (§4.4) measures correlation in binary FP decisions, but the binarization threshold is unstated.** When is a non-repeat attention value counted as a "false positive"? The FP ratio in §3.3 is continuous (a ratio), but phi requires binary classification. What cutoff was used? This threshold choice could substantially affect the independence result.

### Summary

The experimental design has one critical confound (§4.3 capacity experiment conflating unique tokens with sequence length) and several underspecified methodological choices (synonym source, miss threshold, ablation method, FP binarization) that make results hard to evaluate or replicate. The constructed-stimulus-only approach limits generalizability. The most impactful fixes would be: (1) redo the capacity experiment with constant sequence length, (2) add a naturalistic validation pass, (3) specify all thresholds and their justification, and (4) use mean ablation instead of (or alongside) zero ablation.

**Next pass suggestion:** Figures/tables clarity and correctness.

---

## 2026-02-17 — Pass 6: Figures & Tables

**Focus:** Are figures/tables clear, necessary, correctly referenced, and consistent with the text?

### Critical Issues

1. **Fig 4 phi matrix contradicts the text — MAJOR.** The paper states: "Individual pairs range from φ = 0.02 (L0H5 ↔ L3H0) to φ = 0.38 (L0H1 ↔ L0H5)." But the actual figure shows L0H1↔L0H5 = **0.08** and L0H5↔L3H0 = **0.18**. The maximum off-diagonal value in the figure is 0.18 (L0H5↔L3H0), not 0.38. Either the figure was regenerated after the text was written, or the text describes a different analysis. This is a factual error that will destroy credibility if a reviewer checks the figure against the prose. Fix immediately — update the text to match the figure (or vice versa), and verify which version is correct from the raw data.

2. **Fig 6 FP distribution contradicts the text — MAJOR.** The paper claims: "63% trigger exactly one head, 17% trigger two, and only 0.2% trigger all four." The figure clearly shows: **44.0%** trigger one, **24.0%** trigger two, **12.0%** trigger three, **0.2%** trigger four, and **19.7%** trigger zero. The 63% and 17% numbers are wrong. These appear to be computed on a different denominator (perhaps excluding the 19.7% true negatives? 44/80.3 ≈ 55% — still not 63%). Regardless, the text doesn't match the figure. Correct the text to match, and be explicit about whether percentages are of all probes or only false-positive probes.

3. **Fig 5 (combined_fp.pdf) is an orphan figure — never referenced in the paper.** The file exists in the figures directory but no `\ref` or `\includegraphics` points to it. This figure shows something critically important: at 4 heads combined, the **observed** FP rate is ~3 orders of magnitude HIGHER than the independence-predicted rate. This directly undermines the "independent hash functions" claim in §4.4. Either (a) the figure was intentionally excluded because it's inconvenient, in which case the independence claim is dishonest, or (b) it was accidentally dropped, in which case it needs to be added with honest discussion. Either way, this needs attention. The text's claim of 0.17% combined FP should be reconciled with what this figure shows.

4. **Fig 3 capacity curve has more data points than Table 4.** The figure shows ~9-10 data points per head (roughly every 25 tokens from 5 to 200), but Table 4 only reports 5 values (5, 20, 50, 100, 200). The text says "varying context size from 5 to 200 unique tokens" without specifying granularity. The $R^2 = 0.99$ claim — is it computed on the 5 table values or the ~10 figure values? This matters enormously (see Pass 1, issue #5). Report the exact data points used for the fit.

5. **Fig 3 shows a "L5H5 Control" line not mentioned anywhere in the text or tables.** This control head has FP rates that track close to L3H0 and the theoretical curve. If a non-Bloom-filter head also follows the Bloom filter capacity curve, that seriously weakens the claim that the capacity fit is specific to Bloom heads. Why is this control head shown in the figure but never discussed? This looks like it was added for completeness during figure generation and then overlooked — but a reviewer will notice and ask hard questions.

6. **Fig 7 taxonomy: layer range inconsistency.** The figure subtitle says "Previous Token Heads (Layers 2-7)" but Table 3 says the layer range is "2–6." The figure visually shows previous-token heads extending to layer 7. One of these is wrong. Check the data and make them consistent.

### Moderate Issues

7. **Fig 1 heatmap uses log10 scale without mentioning it in the caption.** The colorbar says "Bloom Score (log10)" but the caption describes the metric as "selectivity (hit attention / baseline attention)." The caption should mention the log scale, and clarify that "Bloom Score" = selectivity. Also: the caption says "selectivity >30×" for the highlighted heads, but the log10 colorbar shows the highlighted heads at ~2.0–2.2, which is 100×–158×. The >30× threshold isn't visible on the plot — there's no indication where 30× falls on the colorbar.

8. **Fig 2 error bars are asymmetric and enormous for Bloom heads.** The hit attention bar for Bloom heads has error bars spanning roughly 0.33–0.52. This is a huge range for a mean of ~0.42. The caption doesn't specify what the error bars represent (SD? SE? 95% CI? Bootstrap?). Always label error bars.

9. **Table 4 vs. Fig 3: L1H11 numbers are non-monotonic.** Table 4 shows L1H11 FP rates of 8%, 35%, 22%, 26%, 33% at 5, 20, 50, 100, 200 tokens. The drop from 35% to 22% at 50 tokens is unexpected — Bloom filter FP rates should monotonically increase with load. This non-monotonicity undermines the "follows Bloom filter theory" claim for this head. The figure confirms this dip. Discuss it, or it looks like you're hiding inconvenient data.

10. **Table 5 (ablation) has no figure.** The ablation results are arguably the strongest evidence for functional specificity, yet they're presented only as a small table with no visualization. A figure showing per-sentence perplexity distributions (with and without Bloom heads, split by repeat/no-repeat) would be much more convincing and would address the missing-CIs issue from Pass 1.

11. **No table for cross-model detailed results.** Table 5 gives summary counts per model but no per-head details for GPT-2 medium, large, or Pythia. The reader can't verify the cross-model claims. At minimum, put detailed tables in the appendix.

### Minor Issues

12. **Figure ordering: Fig 5 is skipped.** Figures go 1, 2, 3, 4 (subfig a), 6 (subfig b), 7. The reader sees fig labels jump from 4 to 6, with 5 missing. This is because Fig 5 exists as a file but isn't included. Either add it or renumber.

13. **Fig 4 + Fig 6 are combined as subfigures (a) and (b) under one figure environment**, but they're conceptually different (correlation matrix vs. count distribution). This works but the shared caption is doing a lot of work. Consider whether they're better as separate figures.

14. **All figures appear to use the default matplotlib style.** This is fine for preprint but consider a consistent, publication-quality style (seaborn, or a custom stylesheet) before camera-ready. Font sizes in Fig 7 are notably small given the wide aspect ratio.

15. **Fig 3 inset (miss rate) is very small and hard to read.** The inset shows miss rates near zero for all heads, which is the point — but the y-axis goes to 0.15 and all data is at 0.0, making it visually empty. Consider just stating "miss rate = 0% at all load levels" in the caption rather than using an inset that communicates nothing visual.

### Summary

Two figures directly contradict the text (Fig 4 phi values, Fig 6 FP distribution percentages) — these are manuscript-breaking errors that must be fixed before submission. The orphan Fig 5 reveals a potential problem with the independence claim that's currently hidden. The unexplained L5H5 control in Fig 3 and the L1H11 non-monotonicity both need discussion. Overall, the figures are clear and well-designed, but the text-figure consistency checking was clearly insufficient.

**Next pass suggestion:** Abstract / intro / conclusion alignment with actual contributions.

---

## 2026-02-17 — Pass 7: Abstract / Intro / Conclusion Alignment

**Focus:** Do the abstract, introduction, and conclusion accurately reflect what the paper actually demonstrates?

### Critical Issues

1. **The abstract claims "implement approximate membership testing" — the paper shows behavioral similarity, not implementation.** This was flagged in Pass 2 but persists as a framing problem across all three bookend sections. The abstract says heads "implement approximate membership testing, analogous to Bloom filters." The intro says heads "function as Bloom filters." The conclusion says heads "implement approximate membership testing with the defining properties of Bloom filters." Each formulation subtly overstates: "implement" implies mechanistic identity, but the evidence is behavioral (high selectivity + capacity curve fit). The strongest defensible claim is: "exhibit behavior consistent with approximate membership testing, quantitatively matching Bloom filter theory." Use that or something equivalently careful. "Implement" should appear only when you have QK circuit evidence — which you don't.

2. **Abstract's "practical consequences for sparse attention, KV-cache design, and hallucination detection" are unvalidated.** The abstract lists three applications as if they follow from the findings. None are tested. The Discussion (§5.1) at least frames them as optimizations that "become available," but the abstract drops the hedge entirely. Either add "potential" or remove the application claims from the abstract. Reviewers routinely penalize abstracts that promise more than the paper delivers.

3. **The contributions list omits the ablation but includes cross-model generality.** The four enumerated contributions in §1 are: identification, taxonomic independence, theoretical match, cross-model generality. The ablation (§4.6) is arguably stronger evidence than cross-model generality (which rests on n=4 models with an insignificant correlation — see Pass 1 #4). Meanwhile, the ablation provides causal evidence of functional specificity. Swap: demote cross-model to a supporting result discussed in-text, promote ablation to contribution #4.

4. **Conclusion introduces a claim not in the abstract or intro: "count scaling with model size."** The conclusion states Bloom head count scales with model size, but this finding (r=0.89, n=4) was already flagged as statistically unsupported in Pass 1. The abstract mentions it parenthetically ("$r = 0.89$"). The intro doesn't mention it. If it's a real contribution, put it in the contributions list and defend it. If it's not robust enough (it isn't, at n=4), downgrade it everywhere — especially the conclusion, where readers form final impressions.

5. **Abstract and conclusion both end with the "gradient descent reinvented" framing — but the intro buries it.** The intro's final paragraph lists four technical contributions. The grand narrative ("gradient descent independently arrived at a fifty-year-old data structure") appears only in the abstract's last sentence and the conclusion's last paragraph. The intro never states this as a thesis. This creates a structural mismatch: the abstract and conclusion are making a big philosophical argument, but the intro is making four modest technical claims. Pick one framing and align all three sections. Recommendation: the philosophical framing is the paper's distinctive contribution to the discourse — make it the explicit thesis in the intro (one clear sentence after the contributions list), keep it in the conclusion, but ensure the language acknowledges the behavioral-vs-mechanistic gap throughout.

6. **The intro promises more than it delivers on the Harold Bloom connection.** The second paragraph of the intro builds an elaborate parallel between Burton Bloom's engineering and Harold Bloom's literary theory, suggesting that attention heads solve "Harold's problem" — recognizing a text's relationship to its predecessors. But the experiments test something much narrower: do attention heads detect exact token repetition? That's Burton Bloom's problem, not Harold's. Harold Bloom's "anxiety of influence" is about how later works *creatively transform* their predecessors, not about detecting repetition. The intro's framing implies the paper will show something about how transformers process meaning in light of prior context — but it actually shows they detect string matches. This gap between the promised scope and the delivered evidence is the paper's biggest framing problem. Either (a) narrow the Harold Bloom discussion to a brief stylistic nod (keep the title, cut the rest), or (b) add experiments that actually probe the "semantic influence" angle (e.g., do Bloom heads respond differently to paraphrases vs. exact repeats at a semantic level, beyond the simple synonym test?).

7. **The conclusion's claim "distinct from induction and previous-token heads" understates the Wang et al. overlap concern.** The conclusion says Bloom filter heads are "a new functional category... distinct from induction and previous-token heads." It doesn't mention duplicate-token heads at all. But Passes 2 and 3 both identified the Wang et al. duplicate-token head overlap as a major vulnerability. The conclusion should either (a) explicitly claim distinction from duplicate-token heads (if you've done the analysis) or (b) acknowledge the relationship. Ignoring it in the conclusion looks evasive.

### Moderate Issues

8. **Abstract: "2–4% of attention heads are dedicated membership testers concentrated in early layers."** The cross-model data (Table 5) shows: GPT-2 small 2.8%, medium 0.8%, large 3.8%, Pythia 2.8%. The "2–4%" range excludes GPT-2 medium's 0.8%. Either report "0.8–3.8%" (accurate but less punchy) or explain why medium is low (perhaps its larger head dimension changes the threshold?). Misrepresenting the range in the abstract is a small but unnecessary credibility risk.

9. **Intro §1: "These heads function as Bloom filters: they attend strongly to repeated tokens (zero false negatives)."** But L3H0 has 0.4% miss rate (Table 1), which is NOT zero false negatives. The parenthetical "zero false negatives" is an oversimplification that contradicts the paper's own data. Say "near-zero false negatives" or handle the L3H0 exception explicitly.

10. **Conclusion doesn't mention limitations.** Best practice for NeurIPS is to have the conclusion briefly acknowledge the main limitation (here: behavioral vs. mechanistic evidence) before the closing statement. Currently the conclusion is pure victory lap. One sentence of intellectual honesty would improve it substantially.

### Summary

The three bookend sections consistently overstate the evidence in three ways: (1) "implement" vs. "behave like," (2) unvalidated applications presented as consequences, (3) the Harold Bloom framing promises semantic depth that the experiments don't deliver. The most impactful single fix is replacing "implement" with "exhibit behavior consistent with" throughout, which defuses the strongest reviewer objection while preserving all the actual findings. The second most impactful fix is confronting the duplicate-token head relationship in the conclusion rather than only in Related Work.

**Next pass suggestion:** Full consistency audit — cross-check every number between text, tables, and figures.

---

## 2026-02-17 — Pass 8: Full Consistency Audit

**Focus:** Cross-check every number between abstract, text, tables, and figures. Flag any mismatch.

### Already-Known Contradictions (from prior passes, verified still unfixed)

1. **L3H0 miss rate: 0.0% vs 0.4%.** Abstract says "zero false negatives." §4.1 prose says "All four have 0% miss rate (0/238 observations)." Table 1 says L3H0 = 0.4%. §4.3 says "miss rates remain at 0%." Three sources say 0%, one says 0.4%. **Status: still contradictory.**

2. **Fig 4 phi values vs text.** Text: φ range 0.02–0.38. Figure: different values (max 0.18). **Status: still contradictory.**

3. **Fig 6 FP distribution vs text.** Text: 63%/17%/0.2%. Figure: 44%/24%/12%/0.2%/19.7%. **Status: still contradictory.**

### New Cross-Reference Checks

4. **Abstract: "2–4% of attention heads."** Table 5 values: 2.8%, 0.8%, 3.8%, 2.8%. Range is 0.8–3.8%, not 2–4%. The 0.8% (GPT-2 medium) falls outside the claimed range. **MISMATCH.**

5. **Abstract: "$\bar{\phi} = 0.13$."** §4.4 text says "$\bar{\phi} = 0.13$." But if Fig 4 is correct (values differ from text), the mean phi computed from the figure's actual values would be different. **Cannot verify without raw data; depends on resolving Fig 4 contradiction first.**

6. **Abstract: "fitted parameters closely matching the head dimension."** §4.3: fitted m = 59, d_head = 64. That's 92%, which is "close" but not "closely matching" in the way a reader might expect. Defensible but slightly oversold.

7. **Abstract: "combined false positive rates dropping from 78% to 0.17%."** §4.4: "FP rate of 0.17%, down from 78.3% for L3H0 alone." Abstract rounds 78.3% to 78%. Minor but inconsistent. Also: if Fig 5 (the orphan figure) shows observed combined FP much higher than 0.17%, this number may be theoretical, not empirical. **The 0.17% claim is potentially misleading if it assumes perfect independence that Fig 5 contradicts.**

8. **§4.1: "Cohen's d = 12.3 for hit attention and d = 12.5 for selectivity ($p = 5.8 \times 10^{-8}$, Mann-Whitney U)."** The p-value here is for the Mann-Whitney U test comparing Bloom vs non-Bloom populations. With 4 vs 140 observations, the exact minimum p-value for Mann-Whitney U is C(144,4)⁻¹ ≈ 5.7 × 10⁻⁸. So p = 5.8 × 10⁻⁸ is essentially the minimum possible value — meaning ALL 10,000+ possible rank arrangements were more extreme. This is plausible given the separation, but worth noting: this is a one-sided exact test at its floor, not a continuous p-value. Report it as p < 10⁻⁷ or note it's the exact minimum.

9. **§4.1: "p < 10⁻⁴ permutation test."** Consistent with "10,000 permutations" — you can't get lower than 10⁻⁴ with 10,000 permutations. Fine, but report as p < 1/10000 to make the resolution limit explicit.

10. **§4.2 Table 3: Induction heads count = 16, layer range 5–11.** §4.2 text says induction heads identified via "standard induction head test (random repeated sequences, n=50 trials)." The count of 16 induction heads in GPT-2 small is high — Olsson et al. (2022) typically report 4–6 strong induction heads. Are you using a lower threshold? If so, state it. If 16 is correct, it should be compared to prior counts.

11. **Table 5 "GPT-2 Large (708M)" with "720 total heads."** 36 layers × 20 heads = 720. ✓. But 27 strong BF heads = 3.8%. 27/720 = 3.75%, rounds to 3.8%. ✓.

12. **Table 5: GPT-2 Medium "384 total heads."** 24 × 16 = 384. ✓. 3 strong BF = 0.78%, reported as 0.8%. ✓.

13. **§4.5: "count scales with model size (r = 0.89)."** Data points: (85M, 4), (160M, 4), (302M, 3), (708M, 27). Computing Pearson r on these: the 708M/27 point dominates; without it, (85M, 4), (160M, 4), (302M, 3) would give negative r. This isn't "scaling" — it's one outlier. **The r = 0.89 is driven entirely by GPT-2 Large. Three of four models have 3–4 Bloom heads regardless of size. This correlation is meaningless and should not be reported as a finding.**

14. **§4.3: "L0H1 and L0H5 show near-zero FP rates regardless of context size."** Table 4 shows L0H1 FP rates: 2.0%, 2.0%, 0.0%, 1.0%, 1.0%. These are near-zero. ✓. But the text then says they "function as approximate hash tables rather than Bloom filters" — this is an interesting claim that deserves more than one sentence. If they're NOT Bloom filters, why are they in Table 1 as "Bloom filter heads"?  **Logical inconsistency: §4.1 identifies four "Bloom filter heads" but §4.3 reclassifies two of them as "approximate hash tables." The paper can't have it both ways.**

15. **§4.6 ablation: "Control heads: +11.8% repeat, +5.5% no-repeat."** Which heads are the controls? How were they selected? Same number (4)? Random? Layer-matched? This is never specified. Without knowing the control selection procedure, the comparison is uninterpretable.

16. **§4.6: Induction head ablation "+37.6% repeat, +125.2% no-repeat."** Ablating 16 induction heads (per Table 3) vs 4 Bloom heads — you're ablating 4× more heads. The larger perplexity impact could simply reflect removing more capacity. The interaction term partly controls for this, but the absolute numbers aren't comparable without matching head count.

17. **Contributions list item 1: "selectivity >30× over baseline."** Table 1 shows minimum selectivity = 51× (L3H0). The >30× threshold appears to come from §3.3's "strong Bloom filter head" definition (selectivity > 3×). Wait — >3× or >30×? The contributions say >30×, the methods say >3×. **MISMATCH: 3× vs 30×.** Which is the actual threshold? If 3×, the contributions overstate ("30×" describes the observed values, not the threshold). If 30×, the methods section is wrong. Either way, confusing.

18. **§3.3: "strong Bloom filter head if selectivity > 3×, miss rate < 10%, and mean hit attention > 0.05."** But §4.1 says "selectivity ranging from 51× to 146×." If the threshold is 3×, there might be more heads between 3× and 51× that were excluded — are there? The bimodal claim (huge gap between Bloom and non-Bloom) needs the actual distribution shown. What's the 5th-highest selectivity head? If it's 25×, the >30× claim in contributions is doing suspicious work.

### Summary

This audit found **4 new substantive issues** beyond the already-known contradictions:
- The r = 0.89 scaling correlation is driven by a single data point (#13)
- Two of four "Bloom filter heads" are reclassified mid-paper as "not Bloom filters" (#14)
- The selectivity threshold is inconsistent: 3× in methods vs 30× in contributions (#17)
- Ablation control head selection is unspecified, and head count isn't matched (#15, #16)

Combined with the 3 known text-figure contradictions still unfixed, that's **7 inconsistencies** a reviewer could flag. The most damaging new finding is #14 — if L0H1 and L0H5 are "hash tables, not Bloom filters," the paper's claim of 4 Bloom filter heads reduces to 2, which substantially weakens every result that depends on having multiple independent heads (the hash function analysis, the AND-combination result, the cross-model counts).

**Next pass suggestion:** Check references.bib for completeness and formatting; verify all \citet/\citep resolve correctly.

---

## 2026-02-17 — Pass 9: References & Bibliography

**Focus:** Are all citations present in references.bib? Are any cited but undefined, or defined but uncited? Formatting issues? Missing critical references from Pass 3?

### Critical Issues

1. **Six references flagged in Pass 3 are still missing from references.bib.** The following were identified as critical omissions and have not been added:
   - **Kraska et al. (2018)** "The Case for Learned Index Structures" — the most important missing reference per Pass 3
   - **Rae et al. (2019)** "Meta-Learning Neural Bloom Filters" (ICML)
   - **Mitzenmacher (2018)** "A Model for Learned Bloom Filters" (NeurIPS)
   - **McDougall et al. (2023)** "Copy Suppression" — mechanistic single-head analysis precedent
   - **Merullo et al. (2024)** — retrieval heads, potential overlap with Bloom filter heads
   - **Quirke & Barez (2023)** — training dynamics of induction heads
   
   These aren't just "nice to have." Kraska/Rae/Mitzenmacher directly address the "neural networks learning data structures" narrative. A reviewer who knows this literature will see their absence as a red flag. **Add these to references.bib AND cite them in the paper before submission.**

2. **conmy2023towards is in references.bib but never cited in main.tex.** It was added (presumably after Pass 3 flagged it) but no `\citet` or `\citep` references it. Either cite it in Related Work / Methods or remove it. Orphan bib entries are sloppy.

3. **geva2023dissecting citation format mismatch.** It's cited in the paper as `\citet{geva2023dissecting}` in Related Work, but the bib entry lists it as an arXiv preprint. It was actually published at EMNLP 2023 — update the venue.

### Moderate Issues

4. **nanda2022transformerlens is a GitHub repo, not a paper.** The bib entry is `@article` with a URL field. This is technically fine for `plainnat` style but will render oddly — no journal, no pages, just a URL. Consider using `@misc` or `@software` and adding a note field: "Software available at \url{...}".

5. **elhage2021mathematical is the Transformer Circuits Thread, not a peer-reviewed venue.** The bib entry lists `journal={Transformer Circuits Thread}` which is accurate but unusual. This is standard practice in mech interp papers — fine as-is, but be aware some reviewers may not consider it a "real" citation.

6. **radford2019language is "OpenAI Blog" — again, not peer-reviewed.** Same concern. Standard for GPT-2 citation, but note that GPT-2's technical report was never formally published. Some venues now prefer citing the model card or a specific version.

7. **olsson2022context: "and others" truncation.** The bib entry uses `and others` for the long author list. `plainnat` should handle this with `et al.` rendering, but verify the compiled output doesn't look odd. Some styles render "and others" literally.

8. **Inconsistent use of \citet vs \citep.** Quick scan of main.tex shows the paper uses `\citet` for all citations (parenthetical author-year). Check whether any should be `\citep` (citations as parenthetical notes rather than sentence subjects). E.g., "...as shown by previous work \citep{voita2019analyzing, michel2019sixteen}" is more natural than "...as shown by \citet{voita2019analyzing} and \citet{michel2019sixteen}."

### Minor Issues

9. **bloom1973anxiety: publisher is Oxford University Press.** Verify edition — the 1973 first edition is correct, but a 2nd edition (1997) exists with a new preface. Since the paper references the original theory, 1973 is fine.

10. **No DOIs on any entry.** Adding DOIs would improve discoverability and is increasingly expected. Low priority for arXiv preprint but worth adding for camera-ready.

11. **wang2022interpretability uses `{GPT}-2` with braces for capitalization protection.** Good. But `clark2019what` uses `{BERT}` with the same pattern. Consistent — fine.

12. **biderman2023pythia is listed as ICML 2023 proceedings.** Verify: yes, it appeared at ICML 2023. But the bib entry says `journal={Proceedings of the 40th International Conference on Machine Learning}` — should be `@inproceedings` with `booktitle`, not `@article` with `journal`. This will render incorrectly in most styles.

### Compilation Check Needed

13. **Run `pdflatex` + `bibtex` and check for warnings.** Specifically look for:
    - "Citation `X` undefined" warnings (indicates \cite in tex but missing from bib)
    - "Empty `journal`" or similar field warnings
    - Any bib entries that render as `[?]` in the PDF

### Summary

The bibliography has one orphan entry (conmy2023), one venue error (biderman as @article), and is missing the six most important references flagged two passes ago. The missing Kraska/Rae/Mitzenmacher citations remain the single biggest related-work vulnerability. Adding them will also require revising the "gradient descent independently arrived at" framing throughout the paper, since that claim is less remarkable when there's an existing literature on neural networks learning Bloom filters.

**Next pass suggestion:** Priority fix list — compile all critical issues across all 9 passes into a ranked action list for revision.

---

## 2026-02-17 — Pass 10: Priority Fix List (Compilation)

**Focus:** Rank all critical issues across passes 1–9 into an actionable revision checklist. Ordered by reviewer-damage potential: issues most likely to trigger a reject or major revision request come first.

### 🔴 TIER 1: Manuscript-Breaking (fix before arXiv posting)

**1. Text-figure contradictions (Pass 6 #1, #2, #3)**
- Fig 4 phi values don't match §4.4 text (text says 0.02–0.38; figure shows max 0.18)
- Fig 6 FP distribution doesn't match §4.4 text (text says 63%/17%; figure says 44%/24%)
- Orphan Fig 5 shows observed combined FP rate MUCH higher than the 0.17% claimed in text, undermining the independence claim
- **Action:** Go back to raw data. Regenerate figures OR fix text. Reconcile Fig 5 with the 0.17% claim — if independence doesn't hold empirically, say so honestly.

**2. L3H0 miss rate contradiction (Pass 1 #1, Pass 8 #1)**
- Table 1: 0.4%. Abstract, §4.1 prose, §4.3: 0%. These can't all be true.
- **Action:** Check raw data. If 0.4% is real, fix all prose to say "near-zero" not "zero." Update abstract's "zero false negatives" claim.

**3. Two of four "Bloom filter heads" reclassified as "not Bloom filters" (Pass 8 #14)**
- §4.3 says L0H1 and L0H5 are "approximate hash tables rather than Bloom filters." But §4.1, the abstract, and all statistics treat them as Bloom filter heads.
- **Action:** Either (a) maintain all four as Bloom filter heads and reframe L0H1/L0H5 as a subtype (low-FP Bloom filters), or (b) reclassify them honestly and redo all statistics with n=2 Bloom heads. Option (a) is easier but (b) is more honest. Either way, the current contradictory treatment is indefensible.

**4. Missing critical citations: learned Bloom filter literature (Pass 3 #1-2, Pass 9 #1)**
- Kraska et al. (2018), Rae et al. (2019), Mitzenmacher (2018) — the "neural networks learning data structures" literature exists and directly contextualizes this paper. Not citing it looks ignorant or evasive.
- **Action:** Add all three to references.bib, cite in Related Work, and soften the "independently arrived at" framing. The finding is still interesting — "gradient descent converges on Bloom filters *within a larger model trained on language*" is a different (and arguably more interesting) claim than "gradient descent discovers Bloom filters for the first time."

**5. "Implement" vs "behave like" throughout (Pass 2 #1, Pass 7 #1)**
- The paper claims heads "implement" Bloom filters but provides only behavioral evidence. No QK circuit analysis showing anything resembling hashing into a bit array.
- **Action:** Global find-replace. Use "exhibit behavior consistent with" or "functionally approximate" instead of "implement." Keep "implement" only in Discussion as a hypothesis for future mechanistic work.

### 🟠 TIER 2: Major Revision Triggers (fix before submission to venue)

**6. Selectivity threshold mismatch: 3× vs 30× (Pass 8 #17)**
- Methods §3.3 says threshold is >3×. Contributions list says ">30× over baseline."
- **Action:** Clarify. If 3× is the threshold, say so consistently. Note that observed values happen to be 51×–146×, far above threshold.

**7. Duplicate-token heads not distinguished (Pass 2 #2, Pass 3 #8, Pass 7 #7)**
- Wang et al. (2022) already identified duplicate-token heads. Paper waves this away in one sentence.
- **Action:** Run the IOI duplicate-token head identification on the same model. Show overlap or lack thereof. If they're the same heads, own it and argue the Bloom filter framing adds something (capacity theory, independence analysis). If different, show it.

**8. Capacity experiment confound: unique tokens conflated with sequence length (Pass 5 #7)**
- Varying "5 to 200 unique tokens" changes both filter load AND sequence length/positional effects.
- **Action:** Add a control: hold sequence length constant at 200, vary unique-token proportion. If FP rate still tracks the Bloom formula, the confound is ruled out.

**9. r = 0.89 scaling on n=4 points, driven by one outlier (Pass 1 #4, Pass 8 #13)**
- Three models have 3–4 Bloom heads; GPT-2 Large has 27. That's not "scaling" — it's one data point.
- **Action:** Either (a) test more models (Pythia suite has 8 sizes — use them), or (b) drastically downgrade the scaling claim to an observation. Remove r=0.89 from abstract.

**10. Ablation lacks CIs, significance tests, and control specification (Pass 1 #9, Pass 8 #15-16)**
- Point estimates only. Control heads unspecified. Head count unmatched (4 Bloom vs 16 induction).
- **Action:** Report bootstrap CIs on ablation deltas. Specify control selection (random, layer-matched?). Match head count or normalize per-head.

**11. p-values of $< 10^{-134}$ are asymptotic artifacts (Pass 1 #2)**
- Mann-Whitney U on n≈238 cannot produce p-values this small exactly.
- **Action:** Use exact permutation p-values or cap at "< 10⁻²⁰" with a note explaining the floor.

**12. Ablation uses zero ablation — method unspecified (Pass 5 #9)**
- Zero vs. mean vs. resample ablation gives different results. Paper doesn't specify.
- **Action:** State the method. Ideally run mean ablation as a robustness check.

### 🟡 TIER 3: Strengthening Improvements (do before camera-ready)

**13. Add naturalistic validation (Pass 5 #8)**
- All results on 100 constructed sentences. Run on WikiText/OpenWebText with natural repetitions.

**14. Harold Bloom framing: cut to 1-2 sentences in intro, remove from conclusion (Pass 2 #4, Pass 4 #2,5, Pass 7 #6)**
- Charming but creates expectation of semantic depth the experiments don't deliver.

**15. R² = 0.99 on 5 data points with 3-parameter model (Pass 1 #5, Pass 6 #4)**
- Report adjusted R², run at finer granularity (every 10 tokens), clarify data point count.

**16. Synonym source unspecified (Pass 5 #3)**
- How were synonyms chosen? WordNet? LLM? Author judgment? Report it.

**17. Miss threshold 0.01 is near-chance and unjustified (Pass 5 #4)**
- Derive from null distribution or justify.

**18. Abstract rewrite (Pass 4 #1, Pass 7 #2,8)**
- Lead with plain English. Move numbers to body. Fix "2–4%" range (actual: 0.8–3.8%). Remove unvalidated application claims or hedge.

**19. Promote ablation to contributions list (Pass 7 #3)**
- It's the strongest causal evidence. Demote cross-model scaling (too weak at n=4).

**20. Conmy (2023) orphan bib entry; Biderman @article→@inproceedings (Pass 9 #2, #12)**
- Cite conmy2023 or remove. Fix biderman entry type.

**21. L1H11 non-monotonic FP rates in Table 4 (Pass 6 #9)**
- 8% → 35% → 22% → 26% → 33% — the dip at 50 contradicts Bloom filter theory. Discuss it.

**22. L5H5 control in Fig 3 unexplained (Pass 6 #5)**
- A non-Bloom head tracks the theoretical curve. Discuss or remove.

**23. Fig 7 layer range inconsistency: "2–7" vs "2–6" (Pass 6 #6)**
- Table 3 says 2–6, figure says 2–7. Check and align.

**24. Test multiple prior occurrences (Pass 5 #14)**
- If token appears 3+ times, does head attend to ALL priors? Strong Bloom filter evidence if yes.

**25. 16 induction heads seems high (Pass 8 #10)**
- Olsson et al. report 4–6. State your threshold or explain the discrepancy.

### Recommended Revision Order

1. Fix text-figure contradictions (#1) — 2 hours, raw data needed
2. Resolve L3H0 miss rate (#2) — 30 min
3. Resolve the L0H1/L0H5 "hash table" contradiction (#3) — 1 hour, decision needed
4. Add missing citations and revise framing (#4) — 2 hours
5. Global "implement" → "behave like" (#5) — 1 hour
6. Threshold and statistical fixes (#6, 11) — 1 hour
7. Ablation improvements (#10, 12) — 3 hours if re-running experiments
8. Duplicate-token head comparison (#7) — 4 hours, new experiment
9. Capacity confound control (#8) — 4 hours, new experiment
10. Everything else — ongoing

**Estimated total revision time: 2–3 focused days for Tier 1+2, plus 1–2 days for Tier 3.**

**Next pass suggestion:** After Tier 1 fixes are made, do a fresh read-through focusing on whether the revised framing is internally consistent.

---

## 2026-02-17 — Pass 11: Reproducibility & Code/Data Availability

**Focus:** Could an independent researcher reproduce these results from the paper alone? Are experimental details sufficient? Code/data concerns?

### Critical Issues

1. **No code availability statement.** NeurIPS requires (or strongly encourages) a reproducibility statement and code release. The paper has no mention of code, data, or supplementary materials beyond "supplementary materials" for Appendix B. Will you release:
   - The 100 sentence triplets? (Critical — results depend entirely on stimulus design)
   - The analysis code (TransformerLens scripts)?
   - The raw attention matrices / computed metrics?
   Without the stimulus set, no one can reproduce a single number in this paper.

2. **Stimulus generation procedure is underspecified.** §3.2 says "We construct 100 sentence triplets" but never explains *how*. Were they hand-written? Template-generated? LLM-generated and curated? The sentence structure ("The X did Y and the X did Z") appears formulaic — is every sentence this structure? Are there variations in length, complexity, position of the repeated word? A reviewer doing due diligence will want to know if the results hold beyond one syntactic template.

3. **"Content word" is undefined.** §3.2 says repeated tokens are "content words" but doesn't define this. Does it include verbs? Adjectives? Only nouns? Are function words excluded? This matters because attention patterns differ dramatically between content and function words. If all 100 stimuli use nouns, the results may not generalize to verb or adjective repetition.

4. **Random seed / determinism not addressed.** TransformerLens inference is deterministic (no sampling), so the core attention measurements should be reproducible. But the bootstrap CIs (10,000 resamples) and permutation tests (10,000 permutations) depend on random seeds. Were seeds fixed? If not, the exact CI bounds and p-values will differ on re-run. This is minor (the qualitative results won't change) but good practice to report.

5. **"Random earlier position" for baseline is vague.** §3.3 defines baseline attention as "mean attention from a unique token to a random earlier position." Random how? Uniformly sampled? One per sentence or multiple? If one random position per sentence, baseline estimates are noisy (explaining the wide CIs in Table 1). If multiple, how many? This procedural detail substantially affects the selectivity metric.

6. **Cross-model experiment: same 100 stimuli?** §4.5 tests GPT-2 medium, large, and Pythia-160M. Did all models receive the same 100 sentence triplets? Different tokenizers (GPT-2 vs Pythia) will tokenize differently — was this controlled? A word that's one token in GPT-2 might be two tokens in Pythia, changing what "repeated token" means. This is never discussed.

7. **Capacity experiment (§4.3): how are contexts constructed?** "Varying context size from 5 to 200 unique tokens" — but how? Random words? Sentences? If random words, there's no syntactic structure, which is very different from the sentence-based main experiment. If sentences of varying length, word frequency and syntax vary too. The construction procedure for these contexts is entirely omitted.

8. **No reproducibility checklist.** NeurIPS 2026 will almost certainly require the ML reproducibility checklist (it has since 2019). The paper should be prepared to answer: compute requirements, number of runs, hyperparameter sensitivity, statistical significance, etc. Most of these are easy to fill in but need to be documented.

### Moderate Issues

9. **TransformerLens version not specified.** TransformerLens has had breaking API changes between versions. Pin the version in methods or requirements.

10. **No compute budget reported.** How long do the experiments take? On what hardware? This is a lightweight paper (no training), so presumably minutes on a single GPU — but say so. Reviewers appreciate knowing the computational accessibility of the work.

11. **The Bloom filter head classification threshold (§3.3) needs a sensitivity analysis.** How do results change if the threshold is selectivity > 5× instead of > 3×? Or > 10×? If the bimodal distribution is as extreme as claimed (51× minimum vs. near-zero for non-Bloom), any reasonable threshold should give the same heads — but show this explicitly.

12. **Appendix B promises "complete selectivity, miss rate, and FP ratio for all 144 heads" in supplementary materials.** But no supplementary file is referenced or presumably exists yet. This data is essential for reproducibility — prepare it.

### Minor Issues

13. **The randomly-initialized control needs a seed.** "A randomly initialized model with identical architecture" — what random seed? One initialization or multiple? With one seed, you might get unlucky. Run 5 seeds and report mean ± SD of the number of "Bloom filter heads" detected (should be 0 ± 0 to be convincing).

14. **Bonferroni correction denominator.** §3.4 says α = 0.05/144 for GPT-2 small's 144 heads. But you're also testing 3 metrics per head (selectivity, miss rate, FP ratio). Shouldn't the correction be 0.05/(144 × 3) = 1.16 × 10⁻⁴? Or are the three metrics treated as a single composite test? Clarify.

15. **Phi coefficient binarization threshold (flagged in Pass 5 #15, still unresolved).** The computation of φ requires binary FP decisions. What attention threshold turns a continuous value into "false positive = yes/no"? This is never stated and is critical for the independence analysis.

### Summary

The paper's biggest reproducibility gap is the complete absence of stimulus data and code. The 100 sentence triplets ARE the experiment — without them, nothing is reproducible. Secondary gaps include underspecified procedures for baseline sampling, capacity experiment construction, and cross-model tokenization handling. Adding a 2-page supplementary with the full stimulus set, analysis code pointer, and procedural details would address most concerns. For a NeurIPS submission, prepare the reproducibility checklist now — it will force you to document the details that are currently missing.

**Next pass suggestion:** After Tier 1 fixes from Pass 10 are applied, do a clean read-through to check revised framing consistency.

---

## 2026-02-17 — Pass 12: Alternative Explanations & Threats to Validity

**Focus:** What simpler explanations could account for the findings? What would a skeptical Reviewer 2 argue?

### The Big Alternative: These Are Just Cosine-Similarity Detectors

The most damaging alternative explanation is this: attention heads compute QK dot products. Repeated tokens have *identical* embeddings (before positional encoding) and therefore naturally produce high QK scores. **Any head whose QK weights approximate an identity-like matrix will "detect" repeated tokens as a trivial consequence of the dot-product mechanism — no Bloom filter needed.**

This isn't a Bloom filter "implementing" membership testing. It's just: same vector → high dot product → high attention. The "zero false negatives" property follows trivially (identical vectors always produce high scores). The "false positives for synonyms" follow trivially (similar vectors produce moderately high scores). The "capacity degradation" follows from softmax normalization (more tokens → attention mass spreads → scores for any one token drop, looking like a "filling" Bloom filter).

**This alternative predicts every result in the paper without invoking Bloom filters at all.** To rule it out, you'd need to show:
1. The QK circuit is doing something more complex than approximate identity matching — e.g., hashing into multiple independent subspaces (which would look like actual hash functions).
2. The capacity curve specifically follows the Bloom filter formula and NOT a generic softmax-dilution curve. Currently: is (1 - e^{-kn/m})^k distinguishable from the softmax normalization curve 1/(1+e^{-αn}) or a simple saturating function? On 5 data points, probably not. Fit competing models (logistic, power law, softmax-dilution) and compare AIC/BIC. If the Bloom filter formula doesn't win, the capacity argument collapses.

**This is the single question a strong reviewer will ask, and the paper has no answer for it.** The entire "Bloom filter" interpretation could be replaced by "identity-matching heads with softmax dilution" — a much simpler explanation that invokes no novel data structure analogy.

### Positional Encoding Confound

Repeated tokens in the stimulus always appear at specific relative positions (first occurrence early, second later). GPT-2 uses absolute positional encodings, so the QK product includes a positional component. Heads could be learning to attend to *specific positional distances* rather than *token identity*. To rule this out:
- Test with the repeated token at varying positions (distances of 3, 10, 30, 80 tokens)
- Show selectivity is invariant to position gap (or characterize how it varies)
- The paper never varies position — all 100 stimuli presumably have similar structure and similar positional gaps

### The "4 Out of 144" Problem

Finding 4 heads with unusual properties out of 144 is a 2.8% hit rate. With 144 independent tests and a generous enough search criterion, you *expect* to find outliers. The permutation test (Pass 1) addresses this partially, but:
- The head identification criteria were chosen *after* looking at the data (post-hoc threshold selection, flagged in Pass 5 #10)
- The permutation test uses the criteria that were chosen to capture these specific heads
- A proper correction would pre-register the criteria on one model and validate on another

### Softmax Temperature as Alternative to "Hash Functions"

The "independent hash functions" interpretation (§4.4) rests on low φ between heads. But different heads have different effective temperatures (due to different QK weight norms), which means they'll hit saturation/FP at different inputs for purely parametric reasons — not because they're "hashing differently." Low φ is expected between any set of detectors with different sensitivity thresholds, no independence or hashing needed.

### Duplicate-Token Heads Rebranded (The Occam's Razor Attack)

This was flagged before but deserves restating as the *primary* Reviewer 2 argument: "This paper identifies duplicate-token heads (Wang et al. 2022), relabels them as 'Bloom filter heads,' fits a curve, and calls it a contribution. The Bloom filter analogy, while evocative, adds no explanatory or predictive power beyond what 'heads that detect repeated tokens' already captures. The capacity curve is predicted by softmax dilution. The false positive pattern is predicted by embedding similarity. What, exactly, is new?"

The paper's defense must be: the Bloom filter framework generates *quantitative predictions* (the specific capacity formula) that (a) a simpler model doesn't and (b) match the data. But if competing simple models (softmax dilution) fit equally well (see #1 above), this defense fails.

### Missing: What Happens During Training?

If Bloom filter heads are a "convergent solution" to membership testing, they should:
- Emerge at a consistent point during training (early? late?)
- Emerge across different random seeds
- Perhaps emerge even in non-language domains (vision transformers?)

None of this is tested. Without training dynamics, the "gradient descent inevitably discovers Bloom filters" narrative is just a snapshot observation, not evidence of convergence.

### Summary: The Paper's Argumentative Vulnerability

The paper's argument has this structure:
1. Some heads strongly attend to repeated tokens ✓ (well-demonstrated)
2. This looks like a Bloom filter ← (analogy, not mechanism)
3. The capacity curve fits the Bloom filter formula ← (but may not beat simpler models)
4. Therefore these heads "implement" Bloom filters ← (overclaimed)

A skeptical reviewer can replace step 2-4 with: "These are identity-matching heads. QK dot products naturally spike for identical tokens. Softmax normalization creates capacity-like effects. No novel data structure is needed." **Until the paper rules out this simpler explanation — ideally with a competing model comparison on the capacity curve — the Bloom filter interpretation is unfalsifiable ornamentation on a straightforward observation.**

Strongest recommendation from this pass: **Add a section (or subsection in Discussion) that explicitly states the cosine-similarity alternative, tests it with competing curve fits, and shows whether the Bloom filter formula adds predictive power.** If it does: the paper is much stronger. If it doesn't: better to know now than after posting.

**Next pass suggestion:** After Tier 1 fixes, fresh consistency read-through.

---

## 2026-02-17 — Pass 13: Simulated NeurIPS Review

**Focus:** What would actual NeurIPS reviewers write? Synthesize all prior passes into a realistic review to calibrate expectations.

### Simulated Review (Score: 4 — Borderline Reject)

**Summary:** The paper identifies attention heads in GPT-2 and Pythia that strongly attend to repeated tokens, and argues these heads implement Bloom filters. The evidence includes high selectivity for repeated tokens, a capacity curve that fits the Bloom filter formula, low inter-head correlation, and ablation showing functional specificity.

**Strengths:**
- The core observation is real and well-demonstrated: some early-layer heads are dramatically more selective for repeated tokens than others (50×–150× baseline)
- The capacity experiment connecting context size to false positive rates is creative
- Clean experimental design with matched sentence triplets
- Ablation showing repeat-specific impact is good causal evidence
- Well-written paper with a memorable framing

**Weaknesses:**

**W1 (Major): The "Bloom filter" interpretation is unfalsifiable without ruling out the simpler alternative.** Identical tokens produce high QK dot products by construction. The "zero false negatives" property is trivially predicted by dot-product similarity, as is "false positives for synonyms." The capacity curve could be softmax dilution rather than filter saturation. The paper never fits competing models (logistic, softmax-normalization) against the Bloom filter formula. Without this comparison, the Bloom filter analogy is descriptive, not explanatory. This is the central weakness.

**W2 (Major): Relationship to duplicate-token heads (Wang et al. 2022) is inadequately addressed.** The paper acknowledges these exist in one sentence but never tests overlap. If these are the same heads with a new name, the novelty claim is substantially weakened. The paper needs to either show they're different or argue convincingly that the Bloom filter framework adds something.

**W3 (Major): Multiple text-figure contradictions.** The phi coefficient values in §4.4 don't match Fig 4. The FP distribution percentages in §4.4 don't match Fig 6. L3H0's miss rate is 0% in the text and 0.4% in Table 1. These errors are serious — they suggest insufficient verification and undermine trust in all reported numbers.

**W4 (Moderate): The learned index structure literature (Kraska et al. 2018, Rae et al. 2019, Mitzenmacher 2018) is entirely absent.** There's an existing body of work on neural networks learning Bloom filters. The paper's framing that "gradient descent independently arrived at a fifty-year-old data structure" is misleading without this context.

**W5 (Moderate): The "implement" language overclaims.** No mechanistic evidence (QK circuit analysis) is provided to show these heads contain anything structurally resembling a Bloom filter. The evidence is behavioral. "Behave like" ≠ "implement."

**W6 (Minor): r=0.89 scaling claim on n=4 is statistically meaningless.** Three models have 3–4 heads; one has 27. This isn't scaling — it's one outlier.

**Questions for Authors:**
1. Can you fit a softmax-dilution model to the capacity curve data and compare AIC/BIC against the Bloom filter formula?
2. Which of the four identified heads overlap with Wang et al.'s duplicate-token heads in the IOI circuit?
3. The text says φ ranges from 0.02–0.38 but the figure shows different values. Which is correct?
4. L0H1 and L0H5 are called "approximate hash tables rather than Bloom filters" in §4.3. Why are they counted as Bloom filter heads in all other analyses?

**Recommendation:** The observation that some heads specialize in detecting repeated tokens is solid and worth reporting. But the Bloom filter interpretation — the paper's central claim and raison d'être — is insufficiently supported. The paper needs: (1) competing model fits on the capacity curve, (2) mechanistic evidence or honest downgrading of "implement" to "resemble," and (3) a direct comparison with duplicate-token heads. In its current form, the paper is a well-written observation dressed up as a deeper insight than the evidence supports.

---

### What This Means for Revision Strategy

This simulated review hits 4/10 — borderline reject at NeurIPS. The path to acceptance:

**Minimum viable revision (gets to 5-6):**
- Fix all text-figure contradictions (trust issue)
- Add competing model fits for the capacity curve (if Bloom filter wins → paper is strong; if it doesn't → pivot framing)
- Add Kraska/Rae/Mitzenmacher citations and revise "independently arrived at" framing
- Replace "implement" with "functionally approximate" throughout
- Run IOI duplicate-token head comparison

**Full revision (gets to 7+):**
- All of the above, plus:
- QK circuit analysis showing subspace structure in at least one Bloom head
- Naturalistic text validation (WikiText)
- Capacity experiment with constant sequence length
- Training dynamics (when do these heads emerge?)

The competing-model-fit experiment is the single highest-leverage addition. If the Bloom filter formula genuinely outperforms simpler models, the paper's central claim goes from "interesting analogy" to "quantitative prediction" — and that's the difference between a reject and an accept.

**Next pass suggestion:** After Tier 1 fixes, do the competing model fit experiment. This is the make-or-break analysis.

---

## 2026-02-17 — Pass 14: The Make-or-Break Experiment — Competing Model Fits

**Focus:** Designing the competing model analysis that will determine whether the Bloom filter interpretation holds or collapses. This is the single highest-leverage experiment for revision.

### The Core Question

The paper's central claim rests on the capacity curve (§4.3): L3H0's false positive rate tracks the Bloom filter formula $p \approx (1 - e^{-kn/m})^k$ with $R^2 = 0.99$. But Pass 12 identified a devastating alternative: softmax normalization over growing context produces a capacity-like curve for *any* identity-matching head, no Bloom filter needed. **If a simpler model fits equally well, the Bloom filter interpretation is ornamentation.**

### Competing Models to Fit

Run all five on the same data (L3H0 FP rate vs. unique token count $n$):

1. **Bloom filter** (current): $p = (1 - e^{-kn/m})^k$ — 2 params ($m$, $k$)
2. **Softmax dilution**: $p = 1 - 1/(1 + \beta n)$ — 1 param ($\beta$). Rationale: as $n$ grows, attention mass spreads, threshold-crossing probability increases monotonically. This is the null model.
3. **Logistic saturation**: $p = L/(1 + e^{-r(n - n_0)})$ — 3 params ($L$, $r$, $n_0$). Generic S-curve, very flexible.
4. **Power law**: $p = \alpha n^\gamma$ clipped to [0,1] — 2 params ($\alpha$, $\gamma$). Common in scaling analyses.
5. **Linear**: $p = a + bn$ — 2 params. Straw man, but establishes floor.

### Critical Requirements

- **Same data points for all fits.** Use the full granularity from Fig 3 (~10 points), not just the 5 in Table 4.
- **Compare with AIC and BIC**, not $R^2$. With 5–10 data points, $R^2$ is nearly meaningless. AIC/BIC penalize parameter count. If the Bloom filter formula (2 params) beats the logistic (3 params) on BIC, that's strong evidence. If the softmax dilution model (1 param) ties or beats it, the Bloom interpretation is dead.
- **Run on ALL four Bloom heads**, not just L3H0. L0H1 and L0H5 are already problematic (near-zero FP at all loads). If only 1 of 4 heads fits the Bloom formula better than alternatives, the claim is weak.
- **Bootstrap the model comparison.** Resample the data 1000 times, refit all models each time, compute AIC difference distribution. Report "Bloom filter wins X% of bootstrap samples."
- **Plot residuals.** Systematic residual patterns reveal model misspecification that $R^2$/AIC miss.

### What Each Outcome Means

| Result | Implication | Paper Action |
|---|---|---|
| Bloom formula wins AIC/BIC for L3H0 and 2+ other heads | Strong support — the specific functional form matters | Keep Bloom framing, strengthen claim |
| Bloom formula wins only for L3H0 | Weak support — one head fits, not a general pattern | Downgrade: "one head quantitatively matches; others show qualitative similarity" |
| Softmax dilution ties or wins | Bloom interpretation collapses | **Major pivot:** reframe as "repeated-token detection heads with capacity limitations consistent with softmax normalization." Drop the Bloom filter claim from the title. The head identification and ablation results still stand. |
| Logistic wins (3 params) | Ambiguous — more flexible model wins, as expected | Report but note the parameter penalty; if Bloom is close second, the simpler model is still preferable |

### The Uncomfortable Truth

There's a real possibility the softmax dilution model fits just as well. The Bloom filter formula and softmax dilution are both monotonically increasing saturating functions. On 5–10 data points, they may be indistinguishable. If so, the honest move is to say: "The capacity curve is consistent with both Bloom filter theory and softmax normalization. We cannot distinguish them with current data, but the Bloom filter framework generates additional testable predictions [list them]."

This is less sexy but more defensible — and reviewers respect honesty about the limits of an analogy far more than overclaiming.

### Bonus: The Killer Experiment (If Bloom Wins)

If the Bloom filter formula does win, there's one additional experiment that would make the paper very strong: **predict the optimal number of hash functions.** Bloom filter theory says optimal $k = (m/n) \ln 2$. If you can show that the number of Bloom heads in a model approximates this formula for the model's typical context length and head dimension, that's a quantitative prediction no alternative model makes. This would genuinely demonstrate that the Bloom filter framework has explanatory power beyond curve-fitting.

### Implementation Notes

- Use `scipy.optimize.curve_fit` for all models with same initial bounds
- Use `statsmodels` or manual AIC: $\text{AIC} = 2k + n \ln(\text{RSS}/n)$ where $k$ = param count
- For BIC: $\text{BIC} = k \ln(n) + n \ln(\text{RSS}/n)$
- First: resolve how many data points exist (5 from Table 4? ~10 from Fig 3?). More is better — if you can run at every 10-token interval from 5–200, that's 20 points, much more discriminating.
- Time estimate: ~4 hours including data regeneration at finer granularity

**This experiment should be run BEFORE any other revision work. Its outcome determines whether the paper keeps its title or gets reframed.**

**Next pass suggestion:** After this experiment, fresh read-through of revised manuscript.

---

## 2026-02-17 — Pass 15: Progress Audit — What's Fixed, What's Not

**Focus:** Cross-check the current main.tex against the Tier 1/2 priority fixes from Pass 10 to assess revision status.

### ✅ FIXED (verified in current manuscript)

1. **P-values capped.** Table 1 now reports "$\ll 10^{-20}$" with a note about asymptotic unreliability. ✓ (was Pass 1 #2 / Tier 2 #11)
2. **L3H0 miss rate — partially fixed.** §4.1 now says "Three have 0% miss rate and one (L3H0) has a miss rate of 0.4% (1/238 token-level observations)." The 238 count is explained in a footnote. ✓ However: the abstract still says "near-zero miss rates" which is defensible. BUT §4.3 still says "miss rates remain near zero across all capacity levels for all Bloom heads" — this needs a parenthetical "(L3H0's single miss at the base load level does not recur at higher loads)" or similar. **Mostly fixed; one lingering instance needs nuance.**
3. **Missing citations added.** Kraska, Rae, Mitzenmacher now cited in Related Work (§6). McDougall, Conmy, Gould, Quirke, Geva, Wu (retrieval heads) all present. ✓ (was Tier 1 #4)
4. **"Implement" language softened.** Abstract: "behavior is consistent with Bloom filters." Conclusion: "exhibit behavior consistent with approximate membership testing." The QK conjecture in §2.2 now explicitly says "This conjecture is not directly tested in the present work." ✓ (was Tier 1 #5)
5. **L0H1/L0H5 hash-table contradiction addressed.** §4.3 now says "We retain them in the Bloom filter category as they satisfy all behavioral criteria... while noting their FP profile more closely resembles a perfect hash table than a classical Bloom filter." ✓ This is a reasonable resolution. (was Tier 1 #3)
6. **Orphan Fig 5 now included.** Figure 5 is referenced in §4.4 with honest discussion: "the observed combined FP rate (0.17%) substantially exceeds the rate predicted by perfect independence ($3 \times 10^{-6}$), indicating that the heads are not fully independent." ✓ Good — the independence claim is now properly caveated. (was Pass 6 #3)
7. **Ablation promoted to contributions list.** Now contribution #5. ✓ (was Tier 3 #19)
8. **Ablation now includes mean ablation alongside zero ablation.** Table now shows both methods with CIs. Control is described as "layer-matched" with 10 random selections. ✓ (was Tier 2 #10, #12)
9. **Abstract range fixed.** Now says "0.8--3.8% across models." ✓ (was Tier 3 #18 / Pass 7 #8)
10. **Selectivity threshold clarified.** Contributions now say "selectivity $>3\times$ over baseline (observed values 51$\times$--146$\times$)." ✓ (was Tier 2 #6)
11. **Fig 4 phi values — text updated.** Text now says "Individual pairs range from $\phi = 0.08$ (L0H1 $\leftrightarrow$ L0H5) to $\phi = 0.18$ (L0H5 $\leftrightarrow$ L3H0)." Matches the figure. ✓ (was Tier 1 #1a)
12. **Fig 6 FP distribution — text updated.** Now reported as aggregate: "19.7% of probes trigger no heads (true negatives), 80.2% trigger between one and three heads, and only 0.2% trigger all four." The individual breakdown is noted as estimated. ✓ (was Tier 1 #1b)
13. **Related Work substantially expanded.** Learned data structures (Kraska/Rae/Mitzenmacher) get their own paragraph. Retrieval heads (Wu 2024) explicitly distinguished. Duplicate-token heads (Wang 2022) now get extended discussion. ✓ (was Tier 2 #7)
14. **r = 0.89 scaling claim removed.** No mention of scaling correlation anywhere in the current text. Cross-model section reports counts without fitting a trend. ✓ (was Tier 2 #9)
15. **Harold Bloom material — reduced in intro.** Intro now gives it ~2 sentences rather than a full paragraph. The conclusion still has a Harold Bloom paragraph, but it's shorter and more tightly connected to the false-positive finding. Acceptable compromise. (was Tier 3 #14)

### ❌ STILL UNFIXED (needs attention)

**A. Duplicate-token head overlap — never tested empirically.** (Tier 2 #7) The Related Work now *discusses* the distinction verbally but the paper still has not run the IOI duplicate-token head identification to show whether the heads overlap. This remains the biggest "aren't these just renamed duplicate-token heads?" vulnerability. The current text says the contribution is the quantitative Bloom filter framing + capacity analysis + independence analysis — which is a reasonable defense, but an empirical overlap test would close the gap.

**B. Capacity experiment confound — unique tokens still conflated with sequence length.** (Tier 2 #8) §4.3 still varies "context size from 5 to 200 unique tokens" with no constant-length control. This is the main methodological weakness remaining.

**C. Competing model fits not done.** (Pass 14) The make-or-break experiment — fitting softmax dilution, logistic, power law against the Bloom filter formula — has not been performed. This remains the single highest-leverage addition. Without it, the capacity curve evidence is descriptive, not discriminative.

**D. L1H11 non-monotonic FP rates never discussed.** (Pass 6 #9) Table 4 still shows the 35%→22% dip at 50 tokens. No mention in text. A reviewer will notice.

**E. L5H5 control in Fig 3 — still unexplained in text.** (Pass 6 #5) If this control head also tracks the theoretical curve, it weakens the specificity of the Bloom filter claim. Needs at minimum a sentence.

**F. Synonym source still unspecified.** (Pass 5 #3, Tier 3 #16) Methods still says nothing about how synonyms were chosen.

**G. No code/data availability statement.** (Pass 11 #1) The 100 sentence triplets, analysis code, and raw data are not mentioned. NeurIPS requires this.

**H. No reproducibility checklist prepared.** (Pass 11 #8)

**I. Naturalistic validation still absent.** (Pass 5 #8, Tier 3 #13) All results on constructed stimuli only.

**J. Appendix B "full results" referenced but presumably not yet prepared.** (Pass 11 #12)

### Assessment

The manuscript has improved substantially — roughly 15 of the original ~25 critical/major issues are fixed. The remaining gaps fall into three categories:

1. **Experiments not yet run:** competing model fits (C), constant-length capacity control (B), duplicate-token overlap test (A), naturalistic validation (I). These require code execution, not just text edits.
2. **Missing text:** L1H11 discussion (D), L5H5 explanation (E), synonym source (F), code availability (G), reproducibility checklist (H).
3. **Missing supplementary materials:** Appendix B data (J).

**Recommended next steps (in priority order):**
1. Run competing model fits (Pass 14 design). ~4 hours. Outcome determines framing.
2. Run duplicate-token head overlap test. ~2 hours. Closes the biggest novelty question.
3. Add missing text for D, E, F, G. ~1 hour. Pure writing.
4. Constant-length capacity control. ~3 hours. Closes main methods objection.
5. Prepare supplementary materials. ~2 hours.

**Estimated remaining revision time: 1.5–2 focused days for experiments + writing.**

**Next pass suggestion:** After competing model fits are run, re-evaluate the capacity argument and adjust framing accordingly.

---

## 2026-02-17 — Pass 16: Post-Fix Regression Check

**Focus:** Do the applied fixes introduce new problems? Are revised sections internally consistent? Fresh-eyes audit of changed text.

### New Issues Introduced by Fixes

1. **Ablation table now tells two contradictory stories — and the paper doesn't pick one.** The revised §4.6 honestly reports both zero and mean ablation, which is good. But the narrative is now muddled. Zero ablation says Bloom heads are repeat-specific (+14.6% interaction). Mean ablation says they're general-purpose (interaction = −3.7%). The paper concludes "behavioral specialization for membership testing coexists with broader computational roles" — but this is hand-waving. A reviewer will ask: which ablation method do you trust, and what's your actual claim about functional specificity? **Recommendation:** State clearly that mean ablation is the more principled method (you already note this), then make the mean-ablation result your primary finding. Relegate zero ablation to "for completeness" or a footnote. Currently both methods get equal weight, leaving the reader confused about the paper's own conclusion.

2. **The L0H1/L0H5 "hash table subtype" resolution creates a taxonomic mess.** §4.3 now says these heads "satisfy all behavioral criteria" for Bloom filter heads but "more closely resemble a perfect hash table." This is internally consistent but raises a new question the paper doesn't answer: if 2 of 4 heads are hash tables, not Bloom filters, then the "Bloom filter heads" category is actually *two* subcategories. The independence analysis (§4.4) treats all 4 as equivalent — but AND-combining a near-perfect detector (L0H1, FP ≈ 1%) with a capacity-limited one (L3H0, FP ≈ 25-93%) is not combining "independent hash functions of the same filter." It's combining fundamentally different mechanisms. **The independence/combination analysis needs to acknowledge this heterogeneity.** Currently it doesn't.

3. **Fig 5 inclusion exposes the 0.17% number as misleading.** The paper now says observed combined FP is 0.17% vs. independence-predicted 3×10⁻⁶ — a ~60× discrepancy. Good that this is now disclosed. But the abstract still says "combined false positive rates dropping... to 0.17%." A reader of just the abstract gets the impressive 0.17% without the caveat that this is 60× worse than independence predicts. **Either add "though exceeding the independence prediction by two orders of magnitude" to the abstract, or remove the 0.17% from the abstract entirely.** Currently the abstract cherry-picks the flattering number.

4. **The expanded Related Work now cites Wu et al. (2024) retrieval heads, but the distinction paragraph is thin.** The current text says: "retrieval heads perform content-addressed lookup and value copying... while Bloom filter heads perform membership testing without content retrieval." This is a *claimed* distinction, not a *demonstrated* one. Has anyone checked whether the four Bloom filter heads overlap with Wu et al.'s retrieval heads in GPT-2? If L0H1 is both a "Bloom filter head" and a "retrieval head," the taxonomy is muddier than presented. **At minimum, add a sentence acknowledging this hasn't been tested empirically.**

5. **The Kraska/Rae/Mitzenmacher paragraph in Related Work is well-written but doesn't feed back into the intro/abstract framing.** The Related Work now correctly notes that neural networks learning Bloom filters isn't new. But the abstract's final sentence still says "gradient descent converges on a fifty-year-old data structure, without explicit design" — which is less remarkable now that we know there's a whole literature on this. The framing hasn't been updated to reflect the new citations. **Revise the abstract's closing to something like: "...converges on a solution to the membership-testing problem that quantitatively matches a classical data structure — not through explicit optimization for this objective (as in the learned index literature), but as an emergent subroutine of language modeling."** This acknowledges the prior work while preserving what's genuinely novel.

6. **Contribution #5 (ablation) undersells the finding.** It currently reads: "Two ablation methods reveal that Bloom filter heads contribute to both repeated-token and general processing, indicating behavioral specialization for membership testing coexists with broader computational roles." This is the weakest possible framing of an ablation result — it basically says "the result is ambiguous." If you're promoting ablation to a contribution, make it contribute something definitive. Reframe: "Ablation reveals that Bloom filter heads causally affect repetition processing (zero ablation) and participate in broader contextual computation (mean ablation), consistent with superposition of multiple functions within individual heads."

7. **The QK conjecture disclaimer in §2.2 is good but creates an unfulfilled promise.** "This conjecture is not directly tested in the present work; we characterize these heads behaviorally rather than through circuit-level analysis of the QK weights." — this manages expectations well, but it's in the Background section, which is early. By the time readers hit the Results, they may forget this disclaimer and expect mechanistic evidence. **Repeat a version of this caveat at the start of Results or in the Bloom Filter Signature subsection header.** A single sentence: "All results below characterize behavioral signatures; mechanistic circuit analysis is deferred to future work."

### Consistency of Revised Numbers

8. **Abstract says "false positive rate tracks the theoretical Bloom filter capacity curve as context grows, with fitted parameters closely matching the head dimension."** §4.3 says fitted m = 59 vs d_head = 64 (92%). "Closely matching" is defensible but the abstract implies near-identity. Fine — but be prepared for a reviewer to ask "is 92% close enough?"

9. **The mean phi of 0.13 is now consistent between abstract and §4.4.** ✓ Verified.

10. **Table 1 p-values now say "$\ll 10^{-20}$" with a caption note about asymptotic unreliability.** ✓ Good. But the Cohen's d p-value ($5.8 \times 10^{-8}$) in §4.1 is from a different test (Mann-Whitney comparing 4 vs 140 heads) and doesn't have the same caveat. Per Pass 8 #8, this is near the exact minimum for this sample size. **Add a brief note or report as "p < 10⁻⁷"** to avoid the same asymptotic-artifact concern.

### What's Still Quietly Broken

11. **§4.3's last sentence says "miss rates remain near zero across all capacity levels for all Bloom heads."** Pass 15 flagged that this needs a parenthetical about L3H0's single miss. It still doesn't have one. Small fix — add "(L3H0's single miss at the base load does not recur at higher loads)" or similar.

12. **The paper still has no code/data availability section.** This was flagged in Pass 11 and Pass 15. For arXiv this is optional but strongly expected; for NeurIPS submission it's required. This is a 15-minute fix (write 3 sentences + prepare a GitHub repo) that should not be deferred further.

### Summary

The applied fixes are generally well done — the manuscript is substantially more honest and internally consistent than before. The main new risk is that the ablation section now reads as inconclusive rather than illuminating (issue #1), and the abstract hasn't been updated to reflect the more nuanced framing from the new Related Work and Fig 5 inclusion (issues #3, #5). The L0H1/L0H5 heterogeneity within the "Bloom filter heads" category (#2) is a latent problem that will surface during Q&A if not addressed.

**Top 3 quick fixes from this pass:**
1. Update abstract to reflect the independence caveat and learned-index literature (~15 min)
2. Pick mean ablation as primary, demote zero ablation (~30 min restructuring §4.6)
3. Add code/data availability statement (~15 min)

**Next pass suggestion:** Run the competing model fits (Pass 14). This remains the single most important experiment not yet done.

---

## 2026-02-17 — Pass 17: Hash Resolution Analysis Deep Dive (§4.5 / §5)

**Focus:** The similarity sweep and hash resolution profiles — the newest and potentially strongest section. Is it rigorous? Does it hold up?

### Strengths First

The hash resolution analysis (Section 5) is genuinely the paper's strongest new contribution. It goes beyond "these heads detect repeats" to show *how similarity is encoded* — and the multi-resolution finding (L0H5 ultra-precise vs L1H11 broad) is both novel and functionally meaningful. This section does more to justify the Bloom filter framing than the capacity curve, because it shows these heads aren't just binary detectors — they have graded, distance-dependent responses consistent with locality-sensitive hashing.

### Critical Issues

1. **The "locality-sensitive hashing" claim is again behavioral, not mechanistic.** The paper says the QK projection "functions as a locality-sensitive hash." But all QK projections are similarity computations — that's what dot products do. Showing that attention decays with cosine distance doesn't demonstrate LSH; it demonstrates that a similarity-based mechanism produces similarity-dependent outputs. This is tautological unless you show the QK projection is doing something *beyond* raw similarity — e.g., projecting into distinct subspaces that partition the embedding space differently per head (which would actually be hash-function-like). The multi-resolution property is suggestive but could also reflect different effective temperatures (QK weight norms) rather than different "hash functions."

2. **Cosine similarity is computed in INPUT embedding space, but QK operates in projected space.** The sweep measures probe-target similarity in GPT-2's raw embedding space (before any projection). But the attention heads compute $Q = W_Q x$ and $K = W_K x$, so the relevant similarity is in $W_Q W_K^\top$ space, not raw embedding space. If the QK projection rotates/scales the embedding space substantially, cosine distance in input space is a poor proxy for the space the head actually operates in. **The monotonic decay could be even sharper or different in the projected space.** This isn't necessarily a problem — it just means the current analysis is a lower bound on the relationship — but it should be stated.

3. **1,284 measurements sounds large but the design is dense in some conditions and sparse in others.** Breakdown: 100 exact repeats + 1,000 similarity-graded (100 targets × 10 levels) + 84 synonyms + 100 controls = 1,284. At each of the 10 cosine levels, there are 100 measurements — reasonable. But some levels may have poor coverage if the nearest real word at cosine 0.3 for many targets is the same word, or if finding a word at exactly cosine 0.1 is difficult (most words are near-orthogonal in high-d space, so the 0.0–0.3 range may be densely populated while 0.7–0.9 is sparse). **Report the actual distribution of achieved cosine similarities per bin** — are they tightly clustered at the target level or spread?

4. **The "bandwidth" characterization is qualitative.** The paper describes L0H5 as "ultra-precise" and L1H11 as "broad" based on where FP rates drop to noise floor. But there's no formal bandwidth metric — e.g., the cosine similarity at which FP rate crosses 50% (a standard psychometric threshold), or the slope of the sigmoid fit. Without a formal metric, "ultra-precise" vs "broad" is subjective. **Fit a sigmoid to each head's FP-vs-cosine curve and report the inflection point and slope as formal bandwidth parameters.**

5. **The Kirsch & Hua citations are apt but the connection is loose.** The paper correctly identifies Kirsch (2006) distance-sensitive Bloom filters and Hua (2012) locality-sensitive Bloom filters. But these frameworks use *explicit* LSH functions (e.g., random hyperplane projections) with known collision probability profiles. The paper doesn't show that the QK projections have the same collision probability structure. The citation is "this reminds us of X" rather than "this satisfies the formal definition of X." **Either show the QK projections satisfy the formal LSH property (for any two points, collision probability is a monotone function of distance), or soften the connection to "reminiscent of" rather than citing the papers as if they directly apply.**

6. **Template sentence structure may inflate similarity effects.** All 1,284 measurements use the same sentence frame ("The X was noted and the Y was confirmed" per Appendix A example). If the frame words are identical across conditions, the only varying element is the target-probe pair. This means the head is seeing identical context except for one word swap. In naturalistic text, the surrounding context differs too, which could modulate the similarity response. **The controlled design is a strength for isolation but a weakness for generalizability.** Acknowledge this.

7. **WordNet synonyms at cosine 0.4–0.5 is presented as explanatory, but the causal direction is unclear.** The paper says: "WordNet synonyms, which sit at a mean cosine similarity of roughly 0.4–0.5 in GPT-2's space, land squarely in the transition zone." This explains *why* synonyms produce variable FP rates across heads. But it also means the synonym FP results from §4.1 are entirely predicted by the cosine similarity profile — they add no independent information. **The synonym condition is redundant with the similarity sweep.** This isn't a flaw, but the paper should note it explicitly rather than presenting the synonym result (§4.1) and the similarity sweep (§5) as independent pieces of evidence.

8. **The "multi-scale locality-sensitive hash" parallel (Dong et al. 2019) is intriguing but untested.** The paper cites Dong et al. on multi-granularity LSH and claims the four heads form such a system. But multi-scale LSH systems are designed to provide coverage at different granularities by construction. The four heads were not designed — they emerged from training. Showing they form a multi-scale system requires showing they're *used* at different granularities downstream — e.g., the narrow head's output feeds into exact-coreference circuits while the broad head feeds into semantic-similarity circuits. Without downstream circuit analysis, "they have different bandwidths" ≠ "they form a multi-scale system." The latter implies coordinated function.

### Minor Issues

9. **Fig caption says "Shaded region highlights the 'active zone' where hash collisions occur."** What defines the boundaries of this zone? Visually it appears to be cosine 0.3–0.9, but the text doesn't define it formally. Either define it (e.g., "the range where at least one head has FP rate > 5%") or remove the shading.

10. **The 0.1 attention threshold for FP classification (from §4.4) is used again here.** Was this validated for the similarity sweep data? The threshold was presumably tuned on the original 100-sentence experiment. If the similarity sweep uses different sentence structures or context lengths, the same threshold may not be appropriate.

11. **No error bars on Figure 8 (similarity sweep).** Each data point is the mean over 100 targets. What's the variance? At cosine 0.5, if the mean FP rate is 15% but the 95% CI is [2%, 40%], the "active zone" boundary is much less precise than the clean curve suggests. **Add shaded CI bands.**

### Summary

The hash resolution analysis is the paper's most original and convincing section — it genuinely distinguishes these heads from generic repeat detectors by showing graded, distance-dependent responses with head-specific bandwidths. The main weaknesses are: (1) the LSH framing is behavioral analogy, not formal verification (same issue as the Bloom filter claim but less severe here), (2) similarity is measured in the wrong space (input embeddings vs QK-projected space), and (3) the multi-scale system claim requires downstream circuit evidence. The section would benefit from formal bandwidth metrics (sigmoid fits) and error bars on the sweep figure.

**This section should be the centerpiece of the revision.** If the competing model fits (Pass 14) fail to distinguish the Bloom filter formula from softmax dilution on the capacity curve, the resolution profiles are the paper's remaining strong argument for the Bloom filter / LSH interpretation over the trivial "dot-product similarity" alternative. Invest in making this section bulletproof.

**Next pass suggestion:** Run the competing model fits (Pass 14). Still the highest-priority experiment.

---

## 2026-02-17 — Pass 18: Notation & Mathematical Consistency

**Focus:** Are equations, variable definitions, and mathematical claims internally consistent and rigorous?

### Critical Issues

1. **Equation 1 uses $k$, $n$, $m$ — but the head analogy maps them inconsistently.** The Bloom filter formula $p \approx (1 - e^{-kn/m})^k$ has: $m$ = bit array size, $n$ = number of inserted elements, $k$ = number of hash functions. In §4.3, the fitted parameters are $m = 59$ and $k = 2.16$, with $n$ = number of unique tokens in context. But in §4.4, the four Bloom heads are described as "independent hash functions" — implying $k = 4$ (one per head). These are two different values of $k$: the within-head $k = 2.16$ from the capacity fit, and the between-head $k = 4$ from the independence analysis. The paper never reconciles these. If each head is one hash function ($k=1$ per head, $k=4$ combined), why does the single-head capacity fit yield $k = 2.16$? Does each head internally use ~2 hash functions, making the total system $k \approx 8$? This is never discussed and creates a mathematical inconsistency in the Bloom filter analogy.

2. **$k = 2.16$ "hash functions" is not physically interpretable.** A Bloom filter has an integer number of hash functions. A fitted $k = 2.16$ means either (a) the mapping to Bloom filter parameters is approximate (fine, but say so), or (b) the head is interpolating between $k=2$ and $k=3$ behavior (meaningless for a discrete data structure). The paper presents 2.16 as if it's a meaningful fitted parameter without addressing its non-integer nature. **At minimum, note that $k$ in the fit is a continuous approximation to a discrete parameter, and that rounding to $k = 2$ gives [report the fit quality].**

3. **The $d_k$ in Equation 2 vs $d_\text{head}$ in §4.3.** Equation 2 uses $d_k$ as the key dimension in the attention formula's $\sqrt{d_k}$ scaling. §4.3 compares fitted $m = 59$ to $d_\text{head} = 64$. In standard transformer notation, $d_k = d_\text{head}$ (they're the same thing — the per-head dimension). But the paper uses both notations without stating they're equal. A reader unfamiliar with transformers might not know $d_k = d_\text{head} = d_\text{model} / H$. **Add a sentence in §2.2 or §3.1 defining $d_k = d_\text{head} = 64$ for GPT-2 small.**

4. **Selectivity formula is a ratio of means, not mean of ratios.** §3.3 defines selectivity as $\bar{a}_\text{hit} / \bar{a}_\text{baseline}$. This is the ratio of two sample means. The bootstrap CIs in Table 1 presumably bootstrap this ratio. But the ratio of means ≠ the mean of ratios, and the former can be misleading when the denominator ($\bar{a}_\text{baseline}$) is near zero (which it is — 0.003 for L0H1). The bootstrap should be bootstrapping at the sentence level (resample sentences, compute both means, take ratio) rather than bootstrapping the ratio directly. **Specify the bootstrap unit (sentence-level? token-level?) and confirm the resampling scheme handles the near-zero denominator correctly.**

5. **FP ratio definition is ambiguous for multi-token contexts.** §3.3 defines FP ratio as $\bar{a}_\text{synonym} / \bar{a}_\text{hit}$. But in the capacity experiment (§4.3), "false positive rate" means something different: the fraction of non-member tokens exceeding an attention threshold. These are two different FP metrics — one is a continuous ratio, the other is a binary classification rate. The paper uses "false positive rate" for both without distinguishing them. **Use "FP ratio" consistently for the continuous metric (§4.1) and "FP rate" for the binary threshold metric (§4.3, §4.4, §5). Define both explicitly in §3.3.**

6. **The theoretical Bloom FP curve in Table 4 uses $m = 64$, but the fitted value is $m = 59$.** The "Theory ($m=64$)" column shows predictions using the head dimension as $m$, while the fit yields $m = 59$. This is fine as a comparison, but the caption should clarify: "Theory column uses $m = d_\text{head} = 64$ with optimal $k = (m/n)\ln 2$; the fitted curve (Fig 3) uses $m = 59$, $k = 2.16$." Currently the caption doesn't explain what parameters generate the Theory column values.

7. **"Fitted parameters closely matching the head dimension" (abstract) conflates two things.** The abstract implies $m \approx d_\text{head}$ is the match. But the fit has *two* parameters. The $k = 2.16$ doesn't "match" anything obvious about the architecture (there's no architectural constant near 2.16). So really, one parameter matches and one is free. The abstract should say "fitted filter size closely matching the head dimension" not "fitted parameters" (plural).

### Moderate Issues

8. **No formal definition of "attention to first occurrence."** The selectivity metric references "attention from a repeated token to its first occurrence." But in multi-head attention, "attention" is the softmax output $\alpha_{ij}$. Is this the raw softmax weight? The attention-weighted value contribution? The paper presumably means the softmax weight, but this is never stated. For Bloom filter behavior, the softmax weight is the right quantity (it measures the "query" result), but stating this explicitly prevents confusion.

9. **The "optimal $k$" discussion is missing from §4.3.** Bloom filter theory says the optimal number of hash functions is $k^* = (m/n) \ln 2$. For $m = 59$ and, say, $n = 20$ (a typical context), $k^* = 2.05$ — remarkably close to the fitted $k = 2.16$. This is a strong quantitative prediction that the paper completely misses! **If the fitted $k$ matches the theoretical optimum for typical context lengths, that's much better evidence for the Bloom filter interpretation than the capacity curve alone.** Add this analysis.

10. **Equation 2 doesn't show layer/head subscripts on Q, K, V inputs.** It shows $Q_{\ell,h}$ and $K_{\ell,h}$ as outputs of the projection, but the input $x$ is implicit. In standard notation, $Q_{\ell,h} = W^Q_{\ell,h} x_\ell$ where $x_\ell$ is the residual stream at layer $\ell$. This matters because for layer-0 heads, $x_0$ is just the embedding + positional encoding, while for layer-3 heads, $x_3$ has been processed by 3 layers. The "similarity in embedding space" measured in §5 (input embeddings) is only directly relevant to layer-0 heads. **Note this in the hash resolution section.**

11. **$\phi$ coefficient is defined but the formula is never given.** §4.4 reports phi coefficients but never defines $\phi = (n_{11}n_{00} - n_{10}n_{01}) / \sqrt{n_{1\cdot}n_{0\cdot}n_{\cdot 1}n_{\cdot 0}}$ or even states it's the Pearson correlation for binary variables. For self-containedness, add a brief definition or citation.

### Minor Issues

12. **"$R^2 = 0.99$" — ordinary or adjusted?** With 2 fitted parameters on ~5-10 data points, adjusted $R^2$ could be substantially lower. Specify which $R^2$ is reported.

13. **Table 1's CI format "[105, 201]" uses comma separation.** Convention in statistics papers varies between [105, 201] and (105, 201). Be consistent with NeurIPS conventions (brackets for closed intervals is fine).

14. **The Mann-Whitney $U$ test is non-parametric — Cohen's $d$ is parametric.** Reporting both is fine and common, but note that $d$ assumes normality. With a bimodal distribution (4 Bloom heads vs 140 others), the normal assumption is violated. Cohen's $d$ still has interpretive value as a standardized mean difference, but the normality caveat should be mentioned.

15. **"$p = 5.8 \times 10^{-8}$" for the Mann-Whitney comparing 4 vs 140 heads.** Per Pass 8, this is near the combinatorial floor $\binom{144}{4}^{-1} \approx 5.7 \times 10^{-8}$. This means the observed test statistic is literally the most extreme possible value — all 4 Bloom heads rank above all 140 non-Bloom heads. That's worth stating explicitly: "the four Bloom filter heads occupy the four highest ranks for selectivity among all 144 heads ($p = 5.8 \times 10^{-8}$, exact Mann-Whitney $U$)." This is more informative and more honest than just reporting the p-value.

### High-Value Addition (from issue #9)

The optimal-$k$ analysis could be a significant addition to the paper. If you compute $k^* = (m/n) \ln 2$ for $m = 59$ across a range of typical context lengths (say $n = 20$–50 for the stimuli used), you get $k^* \approx 0.8$–2.0. The fitted $k = 2.16$ is in this range. More compellingly, if the typical context in GPT-2's training data has ~20 unique tokens per sentence, $k^* = 2.05$ almost exactly matches the fit. **This is the kind of quantitative prediction that distinguishes the Bloom filter interpretation from "it's just a saturating curve" — no competing model predicts the specific value of $k$.** Flag this for the next revision pass.

### Summary

The mathematical framework is mostly sound but has two significant gaps: (1) the $k$ parameter inconsistency between the within-head fit ($k = 2.16$) and the between-head interpretation ($k = 4$ heads as hash functions), and (2) two different FP metrics used interchangeably under one name. The biggest missed opportunity is the optimal-$k$ analysis (#9), which could provide the strongest single piece of evidence for the Bloom filter interpretation. The notation could be tightened in several places ($d_k$ vs $d_\text{head}$, bootstrap scheme, $\phi$ definition) but none of these are fatal.

**Next pass suggestion:** Run the competing model fits (Pass 14) and the optimal-$k$ analysis (this pass, #9). These two experiments together determine whether the Bloom filter interpretation survives quantitative scrutiny.

---

## 2026-02-18 — Pass 19: Verification of New Experimental Content

**Focus:** The manuscript has been substantially revised since Passes 1–18. Multiple experiments were added: competing model fits, naturalistic validation, capacity confound control, duplicate-token overlap, optimal-k analysis, and hash resolution profiles. Are these new additions rigorous and internally consistent? Do they actually resolve the issues they were designed to address?

### Competing Model Analysis (§4.3, Table — model comparison)

1. **The competing model analysis is the paper's strongest new addition — and it delivers.** ΔAIC > 11 over the logistic (3 params) is decisive by any standard (Burnham & Anderson's rule of thumb: ΔAIC > 10 = "essentially no support" for the weaker model). The Bloom filter formula beating a more flexible model with *fewer* parameters is exactly the evidence Pass 14 said was make-or-break. ✓

2. **The softmax dilution model gets crushed (ΔAIC = 23.5).** This conclusively rules out the "it's just attention spreading" alternative from Pass 12. The 1-parameter model simply can't capture the curve shape. This is the single most important result for defending the Bloom filter interpretation. ✓

3. **However: the table reports AIC but not BIC.** BIC penalizes parameters more heavily with small samples. With ~10 data points, BIC would be more conservative. If BIC still favors Bloom, report it. If it doesn't, the result is less robust than AIC alone suggests. **Add BIC column to the model comparison table.**

4. **How many data points in the fit?** The text says "varying context size from 5 to 200 unique tokens" but the table shows 5 rows (5, 20, 50, 100, 200). Pass 15 flagged that Fig 3 shows ~10 points. **If AIC is computed on 5 points with a 2-parameter model, the effective degrees of freedom are 3 — barely enough to discriminate models.** Confirm the point count. If 5, acknowledge the limitation. If 10+, state it clearly.

5. **Adjusted R² reported — good.** The table now shows R²_adj = 0.990 for Bloom, 0.965 for logistic. This addresses Pass 1 #5. ✓

### Optimal-k Analysis (§4.3)

6. **This is excellent — exactly the kind of quantitative prediction no competing model makes.** Fitted k = 2.16 vs theoretical optimal k* = (59/20)ln2 = 2.05. The 5% agreement is striking and unique to the Bloom filter framework. No alternative explanation (softmax dilution, generic saturation) predicts anything about the value of an internal parameter. This was flagged as a "high-value addition" in Pass 18 #9 and it delivers. ✓

7. **The choice of n = 20 "typical context" is somewhat arbitrary.** k* depends on n. At n = 50, k* = (59/50)ln2 = 0.82 — quite different from 2.16. The claim works for n ≈ 20 but not for all context lengths. **State the range of n for which k is near-optimal (roughly n = 15–25), and note this corresponds to typical unique-token counts in short sentences. If your stimuli average ~20 unique tokens, say so explicitly — it makes the prediction testable rather than post-hoc.**

### Naturalistic Validation (§4.5, Table — WikiText)

8. **This was one of the most-requested additions (Pass 5 #8, Tier 3 #13) and it's well-executed.** 761 passages, 35,998 repeat pairs — large sample. Selectivity 15–54× with <1% miss rate on real text. The rank ordering of heads is preserved. This substantially strengthens generalizability. ✓

9. **Selectivity drops from 51–146× (constructed) to 15–54× (naturalistic).** The paper explains this as function-word dilution, which is reasonable. But there's another explanation: in natural text, "repeated" tokens often appear in different syntactic roles (e.g., "the" as determiner for different nouns), and the contextual embeddings differ even if the token is identical. For layer-0 heads (L0H1, L0H5), this doesn't matter (they see raw embeddings). For L3H0 (layer 3), the residual stream has been processed by 3 layers, so "the" in different contexts has different QK representations. **The selectivity reduction for L3H0 (from 51× to 15.4×) is steeper than for L0H5 (from 74× to 53.8×). This is consistent with the layer-depth explanation — deeper heads see more context-modulated representations. Worth noting as further evidence that the QK mechanism, not just token identity, drives these heads.**

10. **Control comparison: "Control mean 0.6×" — what are the control heads?** Same layer-matched non-Bloom heads from the ablation? Different set? Specify. Also: 0.6× means control heads attend *less* to repeated tokens than to random positions — why? This is anti-selectivity. Is it an artifact of softmax normalization (repeated tokens' attention is spread across their first occurrence, leaving less for other positions)? This deserves a sentence of explanation rather than just reporting the number.

### Capacity Confound Control (§4.3, paragraph on controlling for sequence length)

11. **This directly addresses Pass 5 #7 — the most critical methodological weakness.** Fixed sequence length at 200, varying unique proportion. L3H0 still fits Bloom formula with R² = 0.98. ✓ The confound is ruled out.

12. **Fitted parameters differ substantially: m = 10.4, k = 0.81 (control) vs m = 59, k = 2.16 (original).** The paper notes "lower fitted m reflects a different operating regime (many repeated tokens competing for attention)" — but this is hand-waving. If the Bloom filter parameters change 6× depending on experimental conditions, what does "the head has m = 59 bits" actually mean? Is it a property of the head or a property of the stimulus? **This raises a legitimate concern: the fitted parameters may be curve-fitting artifacts rather than reflecting a fixed internal structure.** A real Bloom filter has fixed m and k regardless of how you test it. The fact that m and k shift with experimental design suggests the Bloom filter formula is fitting the *behavior* well but the parameters may not map to physical quantities in the head. Acknowledge this more honestly — currently the "different operating regime" explanation sounds dismissive.

### Duplicate-Token Head Overlap (§6, Related Work)

13. **This is the most honest and effective resolution of the biggest prior weakness.** The paper now explicitly states: all four Bloom filter heads ARE the top four duplicate-token heads from Wang et al. The contribution is reframed as quantitative characterization + Bloom filter theory + capacity analysis + resolution profiles, not discovery of new heads. This is exactly what Passes 2, 3, 7, and 13 demanded. ✓

14. **The generalization index comparison (0.64 vs 0.42) is a nice touch.** Showing Bloom filter heads respond to *any* repeated token, not just names in the IOI task, is genuine added value. ✓

15. **One vulnerability remains: the paper still calls them "Bloom filter heads" rather than "duplicate-token heads exhibiting Bloom filter behavior."** A reviewer could argue the renaming is misleading if they're literally the same heads. The current framing ("our contribution is the characterization, not the identification") is defensible but delicate. **Consider a single sentence early in the paper (§1 or §2.3) explicitly stating: "The heads we study are identical to the duplicate-token heads of Wang et al. (2022); our contribution is the quantitative demonstration that their behavior matches Bloom filter theory."** Preempt the objection rather than waiting for Related Work.

### Hash Resolution Analysis (§4.5 — now §5 in the structure)

16. **The cosine similarity proxy caveat is now included.** Footnote acknowledges that input-space cosine is a proxy for QK-projected space, and that the relationship may be tighter in the true operating space. ✓ (addresses Pass 17 #2)

17. **The bandwidth descriptions are still qualitative ("ultra-precise," "broad") without formal metrics.** Pass 17 #4 suggested fitting sigmoids and reporting inflection points. This hasn't been done. The descriptions are informative but a reviewer wanting quantitative rigor will ask for numbers. **Low priority but worth adding: the cosine threshold at which FP rate crosses 10% for each head.**

18. **No error bars on the similarity sweep figure.** Pass 17 #11 flagged this. Each point is a mean over 100 targets — the variance matters. Add CI bands or at minimum report the range. **This is higher priority than it seems: if the FP rate at cosine 0.5 is 15% ± 20%, the "transition zone" is much less precisely located than the clean curve suggests.**

### Revised Ablation (§4.6)

19. **Mean ablation as primary method, zero ablation for completeness — good structure.** ✓ (addresses Pass 16 #1)

20. **The mean ablation result is genuinely interesting but uncomfortable for the paper's narrative.** Interaction = −3.7% means Bloom heads affect novel tokens *more* than repeated tokens under the more principled method. The paper handles this honestly ("behavioral specialization coexists with broader computational roles") but a hostile reviewer will read this as: "these heads aren't specialized for membership testing at all — they're general-purpose heads that happen to produce high attention for repeated tokens." **The defense is the behavioral evidence (selectivity, capacity curves) is overwhelming regardless of what ablation shows. Ablation tests causal necessity, not behavioral function. A head can behave as a Bloom filter (producing membership signals) while also contributing to other computations (superposition). Make this distinction crisper.**

21. **The individual head ablation finding — L3H0 has near-zero ablation impact despite being the best Bloom filter match — is actually troubling.** If the head that most closely matches Bloom filter theory is the most dispensable, it suggests the Bloom filter behavior may be an epiphenomenon (a byproduct of the head's actual function, which is redundant with other heads). **The paper's explanation (redundancy with other Bloom heads) is plausible but untestable without ablating all four simultaneously.** If ablating all four together produces a larger effect than the sum of individual ablations, that's evidence for redundancy. If not, they may genuinely be unimportant. Was this tested? If not, flag as future work.

### Conclusion & Harold Bloom

22. **The revised conclusion paragraph connecting false positive misprision to Harold Bloom's misprision is genuinely clever and well-written.** The structural parallel — "the poet misreads what is there; the filter mis-recognizes what is not" — is sharp and specific. Unlike the earlier version (which was vague literary hand-waving), this connects to a concrete finding (the distance-sensitive FP profile). The parallel works because both involve proximity-dependent errors at the boundary of recognition. ✓ This is now an asset rather than a liability.

23. **The final sentence is strong:** "Without blueprints, without explicit data structure objectives, and apparently without choice." Good closer. ✓

### Summary

The manuscript has improved dramatically. The competing model analysis and optimal-k prediction are the two strongest additions — together they elevate the paper from "interesting analogy" to "quantitative prediction that beats alternatives." The naturalistic validation and duplicate-token overlap acknowledgment close the two biggest prior vulnerabilities. 

Remaining concerns (in priority order):
1. **Capacity control parameters (m shifts from 59 to 10.4)** — needs more honest discussion about what fitted parameters mean (issue #12)
2. **No error bars on similarity sweep** — could undermine the hash resolution section's precision claims (issue #18)
3. **L3H0's zero ablation impact** — the head that best matches Bloom filter theory is the most dispensable; needs clearer framing (issue #21)
4. **Data point count for AIC comparison** — if only 5 points, the model discrimination is weak despite large ΔAIC (issue #4)
5. **BIC column missing from model comparison** — easy addition that strengthens the claim (issue #3)

**Overall assessment:** The paper is now in solid shape for arXiv posting. Tier 1 issues are resolved. Most Tier 2 issues are resolved. The simulated review score from Pass 13 (4/10) would likely rise to 6–7/10 with these changes — competitive for NeurIPS. The competing model analysis and optimal-k prediction are the difference-makers.

**Next pass suggestion:** Final pre-submission checklist — LaTeX compilation, figure resolution, supplementary materials, GitHub repo contents.

