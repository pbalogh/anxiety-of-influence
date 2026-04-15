# Literature Map — The Anxiety of Influence

How each reference supports the claim that transformer attention heads implement Bloom filters.

---

## The Two Blooms

### The Namesakes
- **Burton H. Bloom 1970** — Space/time trade-offs in hash coding with allowable errors. The original Bloom filter paper. A probabilistic data structure: hash item through k functions, set k bits. Membership test: check if all k bits are set. False positives possible (bits set by other items), false negatives impossible.
- **Harold Bloom 1973** — *The Anxiety of Influence: A Theory of Poetry.* Poets misread their precursors through six "revisionary ratios." The title pun: attention heads experience "anxiety" about whether a token was truly seen before (the false positive problem).

**Our claim:** Attention heads implement learned Bloom filters — probabilistic membership tests over the context window. The QK dot product is a learned hash; the attention pattern is a soft membership query; false positives produce characteristic errors.

---

## Attention Head Mechanics

### Circuit-Level Understanding
- **Vaswani et al 2017** — Attention is All You Need. The architecture. QKV attention is the substrate we claim implements Bloom filtering.
- **Elhage et al 2021** — Mathematical framework for transformer circuits. Residual stream + attention + MLP decomposition. The language for describing what each head does.
- **Olsson et al 2022** — Induction heads. The best-characterized attention head type: [A][B]...[A] → predict [B]. We argue induction heads are a SPECIAL CASE of Bloom filter heads — they test membership of the bigram [A][current] in the context.
- **Wang et al 2022** — IOI circuit. Indirect object identification as a circuit. Shows attention heads have specific, interpretable roles. Our Bloom filter interpretation provides a unifying explanation for multiple head types.
- **Conmy et al 2023** — Automated circuit discovery (ACDC). Methodology for finding circuits; complements our per-head Bloom filter analysis.

### Head Specialization & Pruning
- **Voita et al 2019** — Specialized heads do the heavy lifting. Some heads matter, most can be pruned. In Bloom filter terms: some hash functions are more discriminative than others.
- **Michel et al 2019** — Are sixteen heads really better than one? Head redundancy. In Bloom filter terms: redundant hash functions.
- **Clark et al 2019** — What does BERT look at? Attention pattern analysis. The empirical ground truth for what attention heads attend to.

### Specific Head Types
- **McDougall et al 2023** — Copy suppression heads. Heads that SUPPRESS tokens that have appeared. This is the complement of a Bloom filter: instead of "was this seen?" it's "suppress because seen." Anti-Bloom.
- **Wu et al 2024** — Retrieval heads for long-context factuality. Heads that retrieve specific facts from context — a Bloom filter with a value payload (Bloom filter → counting Bloom filter → retrieval).
- **Gould et al 2023** — Successor heads. Heads that predict the next item in a sequence (days of week, months). A Bloom filter over sequential patterns.
- **Quirke et al 2023** — Understanding addition. Arithmetic heads. Shows the diversity of head specialization — some heads compute rather than retrieve.

### Factual Recall
- **Geva et al 2023** — Dissecting factual recall. Subject→last-token→MLP pipeline. The attention part is the Bloom filter lookup; the MLP part is the value retrieval.

## Learned Data Structures (the CS bridge)

- **Kraska et al 2018** — The case for learned index structures. LANDMARK paper: neural nets can replace traditional data structures (B-trees, hash maps, Bloom filters) and outperform them because they learn the data distribution. We extend this to show transformers have ALREADY learned these structures — attention heads ARE learned Bloom filters, not by design but by gradient descent.
- **Rae et al 2019** — Meta-learning neural Bloom filters. Explicitly trained neural Bloom filters. The supervised version of what attention heads learn implicitly.
- **Mitzenmacher 2018** — Optimizing learned Bloom filters by sandwiching. Hybrid learned+classical Bloom filters. Methodological reference for our false positive analysis.

### Hashing Connections
- **Indyk & Motwani 1998** — Locality-sensitive hashing (LSH). Approximate nearest neighbors via hash families. QK attention IS a form of LSH — similar queries and keys hash to similar attention weights.
- **Dong et al 2019** — Multi-resolution LSH. Learning hash functions at multiple scales. Multi-head attention = multi-resolution hashing.
- **Kirsch & Mitzenmacher 2006** — Distance-sensitive Bloom filters. Bloom filters that consider distance, not just membership. Soft attention is exactly this — graded membership based on QK similarity.
- **Hua et al 2012** — Locality-sensitive Bloom filters. Combines LSH with Bloom filters. The theoretical object that best describes what attention heads implement.

## Model & Infrastructure
- **Radford et al 2019** — GPT-2. The model we study.
- **Biderman et al 2023** — Pythia suite. Cross-model validation (the architecture-dependence question).
- **Nanda 2022** — TransformerLens. Hook-based analysis framework.

---

## The False Positive → Hallucination Connection

The paper's most provocative claim: Bloom filter false positives in attention (token X activates a head pattern meant for token Y) contribute to hallucination. The head "remembers" seeing something it didn't, and this phantom activation propagates through the residual stream.

This connects forward to:
- → **Discrete Charm**: MLP consensus detects when attention has produced unreliable routing
- → **Darkness Visible**: N2123 exception handler fires on attention-produced anomalies
- → **Garden-Path**: The one-stage parser finding means false positives are resolved immediately, not through reanalysis
- → **Dendritic Diffusion**: Supersaturation (probe confidence) is high when attention is reliable, low when Bloom filter uncertainty is high

---

*25 references. arXiv: 2602.17526.*
