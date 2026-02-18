# The Anxiety of Influence: Bloom Filters in Transformer Attention Heads

Paper, code, and data for our mechanistic interpretability study showing that transformer attention heads independently converge on Bloom filter data structures for token membership testing.

## Key Findings

- **4 attention heads** in GPT-2 Small exhibit Bloom filter behavior: zero miss rate, characteristic false positive patterns, capacity curves matching theoretical predictions (R²=0.99)
- **Novel functional category**: Bloom filter heads (L0-L3) are completely separate from induction heads (L5-L11) — zero overlap
- **Multi-model validation**: Bloom filter heads found in GPT-2 Small/Medium/Large and Pythia-160M
- **Distance-sensitive**: Similarity sweep (1,284 measurements) reveals these are distance-sensitive Bloom filters (Kirsch & Mitzenmacher 2006)
- **Competing model analysis**: Bloom filter model decisively outperforms logistic, softmax dilution, power law, and linear alternatives (ΔAIC > 11)

## Structure

```
paper/          LaTeX source, figures, bibliography
code/           All experiment scripts (Python + TransformerLens)
results/        JSON result files from all experiments
data/           Stimulus data (target words, sentence frames)
```

## Building the Paper

```bash
cd paper
tectonic main.tex
```

## Running Experiments

Requires Python 3.10+, PyTorch, TransformerLens:

```bash
pip install transformer-lens torch numpy scipy matplotlib
cd code
python experiment1_bloom_signature.py
```

## Citation

```bibtex
@article{balogh2026anxiety,
  title={The Anxiety of Influence: Bloom Filters in Transformer Attention Heads},
  author={Balogh, Peter},
  year={2026}
}
```
