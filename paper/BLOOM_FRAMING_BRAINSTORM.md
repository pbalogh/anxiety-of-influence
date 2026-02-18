# Harold Bloom Framing — Iterative Brainstorm

## The Problem
The current paragraph (intro ¶2) is too on-the-nose: "Literary theorist Harold Bloom said X. Transformers do X too. Neat!" It reads as ornamental rather than illuminating.

## What We Want
The parallel should feel *earned* — like the Bloom filter connection to Harold Bloom reveals something about both, not just name-drops one to dress up the other.

## Current Version (v0)
> Three years later, literary theorist Harold Bloom published *The Anxiety of Influence* (1973), arguing that every literary work is haunted by its predecessors—that the act of creation is fundamentally an act of grappling with what came before. The parallel to transformer language models is striking: at every token position, the model must determine what has appeared in its context and how that context constrains the present. The question "have I seen this before?" is not peripheral to language processing—it is foundational.

### What's wrong: "The parallel is striking" is the author telling you to be impressed. Show, don't tell.

---

## Alternative Framings

### v1: The Shared Problem (functional, understated)
> The question "have I seen this before?" sits at an unlikely intersection. For Burton Bloom, it was an engineering problem: how to test set membership in sublinear space. For Harold Bloom, it was the central anxiety of literary creation: the impossibility of writing without echoing what came before. For a transformer processing language, we argue it is both—a computational primitive that the architecture must solve at every position, and that some attention heads solve in a way Burton Bloom would have recognized immediately.

*Pro: Makes the connection feel structural rather than cosmetic. Con: Maybe too cute about the two Blooms.*

### v2: Lead with the question, reveal the connection late
> At every position in a sequence, a language model faces a deceptively simple question: has this token appeared before in my context? The answer determines whether to copy, to predict, or to treat the current moment as genuinely new. This is not a peripheral computation—it is the hinge on which in-context learning, repetition avoidance, and coreference all turn. We show that some attention heads are dedicated to exactly this question, implementing it through a mechanism that is, in a precise technical sense, a Bloom filter.

*Then in the conclusion:* > The title of this paper alludes to a different Bloom—Harold, whose *Anxiety of Influence* (1973) argued that every act of creation is shadowed by what came before. The parallel is not merely nominal: the heads we identify are literally anxious about influence, devoting their representational capacity to tracking which tokens have already exerted their presence in the context window.

*Pro: The literary connection lands harder because you've earned it with 8 pages of evidence. Con: Loses the title payoff in the intro.*

### v3: The epigraph approach
Don't explain the connection at all. Just open with an epigraph:

> *"Every poem is a misinterpretation of a parent poem."* —Harold Bloom, *The Anxiety of Influence* (1973)

Then write the intro as pure CS. Let the reader figure it out or not. The title does the work.

*Pro: Maximum elegance. Trusts the reader. Con: Some reviewers won't get it and will think the title is just whimsical.*

### v4: Invert it — start from the transformer's perspective
> A transformer generating text is, at every token, engaged in an act Harold Bloom might have recognized: the determination of what has come before, and what—if anything—is new. Bloom's literary theory held that creation is inseparable from this reckoning with prior influence. We find that transformers agree, at least architecturally: a subset of attention heads in early layers are dedicated to nothing but this question, implementing approximate membership testing over the context window using a mechanism formally equivalent to the (other) Bloom filter.

*Pro: "(other) Bloom filter" is genuinely funny and self-aware about the conceit. Con: Might be too casual for NeurIPS.*

### v5: Frame it as convergence
> It is a minor curiosity that the data structure and the literary theory share a surname. It is a more substantive curiosity that they share a core concern: the problem of determining what is original and what is repetition. Burton Bloom's 1970 filter solves this as an engineering problem; Harold Bloom's 1973 *Anxiety of Influence* frames it as the central problem of literary creation. We show that transformer attention heads converge on Burton's solution to Harold's problem.

*Pro: "Burton's solution to Harold's problem" is a great line. Con: "minor curiosity / more substantive curiosity" structure is a bit academic-cute.*

---

## Evaluation Criteria
- Does it make the reader smarter, or just amused?
- Would it survive a skeptical reviewer who thinks literary references in ML papers are pretentious?
- Does it set up the paper's actual argument (membership testing is a dedicated computation)?
- Is the title payoff clear without being labored?

### v6: Hybrid of v1 structure + v5 payoff
> The question "have I seen this before?" sits at an unlikely intersection of computer science and literary theory. For Burton Bloom, it was an engineering problem: how to test set membership in sublinear space, solved by the probabilistic filter that bears his name (Bloom, 1970). For Harold Bloom, it was the central anxiety of creative work: every text is haunted by its predecessors, and the act of writing is inseparable from the reckoning with what has already been written (Bloom, 1973). We show that transformer attention heads converge on Burton's solution to Harold's problem—dedicating a subset of early-layer heads to nothing but determining which tokens have already appeared in context, using a mechanism that is, in a precise technical sense, a Bloom filter.

### v7: Detection as precondition for meaning (Peter's insight about repetition altering meaning)
> The question "have I seen this before?" is more consequential than it appears. In language, repetition is not merely a fact to be catalogued but a semantic event: the second occurrence of a word establishes coreference, modulates surprisal, and reshapes the expectations that govern everything downstream. This is the core insight that connects Burton Bloom's engineering and Harold Bloom's literary theory—not simply that both concern repetition, but that both recognize detection as the *precondition* for meaning. Burton Bloom's 1970 filter provides the mechanism for approximate membership testing in sublinear space. Harold Bloom's 1973 *Anxiety of Influence* argues that no text can mean anything without first reckoning with what came before it. We show that transformer attention heads converge on the former to accomplish the latter—dedicating early-layer heads to detecting which tokens have already appeared in context, as the necessary first step before the model can determine what that repetition *means*.

*Key move: The two Blooms aren't a coincidence being noted — they're two perspectives on the same insight (detection is precondition for meaning).*

## Status: OPEN — revisit periodically

---

## 2026-02-17: Clinamen Detection — Hallucination Prevention Framing

### The Insight (from Peter)
Bloom's "misprision" enables innovation through a "swerve" (clinamen) away from the source material. Our hallucination detection application is essentially **clinamen detection**: watching for moments where the model's recognition of context has swerved from accuracy.

### Why This Works
- A Bloom filter FP is a clinamen: a token arrives, the head fires as if recognizing a predecessor, but the token has *swerved* — it's "physician" not "doctor." The head detects the influence but not the swerve.
- The errors aren't random — they cluster around synonyms and semantically related tokens. The swerve is always *from something real*, a near-miss that carries the ghost of actual context.
- "Hallucination prevention via clinamen detection" reframes error detection as detecting where recognition has deviated from fidelity.

### The Inversion (important for accuracy)
Harold Bloom and Burton Bloom describe **inverse operations**:
- **Harold Bloom (misprision):** The familiar predecessor is misread → distorted into something new (familiar → unfamiliar). The poet knows Milton but swerves away.
- **Burton Bloom (filter FP):** The new token is mis-recognized → treated as something already seen (unfamiliar → familiar). The head encounters "physician" and registers "doctor."
Both involve imperfect recognition at the old/new boundary, but they swerve in opposite directions. The poet misreads what's there; the filter mis-recognizes what isn't.

### Where to Use This
- **NOT in the paper** — too much domain-specific jargon (clinamen, misprision as technical terms) for an ML venue
- **Blog post / talk / Twitter thread** — "hallucination prevention via clinamen detection" is a killer line for public-facing discussion
- **The inversion** is worth noting in any literary-audience version of this work
- The conclusion paragraph now captures the structural parallel without the Lucretius terminology
