"""
Figure: Attention as a function of embedding similarity for Bloom filter heads.
Shows the "hash resolution" of each head — how sharply attention decays
as probe tokens become less similar to the target.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.size'] = 11

# Load results
data = json.load(open('../results/similarity_sweep_full.json'))
agg = data['aggregated_by_head_and_level']

bloom_heads = ['L0H5', 'L0H1', 'L3H0', 'L1H11']  # ordered by precision
labels = {
    'L0H5': 'L0H5 (ultra-precise)',
    'L0H1': 'L0H1 (precise)',
    'L3H0': 'L3H0 (standard)',
    'L1H11': 'L1H11 (broad)',
}
colors = {
    'L0H5': '#2166ac',
    'L0H1': '#4393c3',
    'L3H0': '#f4a582',
    'L1H11': '#d6604d',
}

sim_bins = ['0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1', '0.0']
sim_values = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# LEFT PANEL: Normalized attention (% of exact-repeat)
for h in bloom_heads:
    exact = agg[h]['exact']['mean_attention']
    values = [agg[h][b]['mean_attention'] / exact for b in sim_bins]
    # 95% CI error bars: ±1.96 * std / sqrt(n) , normalized by exact
    ci = [1.96 * agg[h][b]['std_attention'] / np.sqrt(agg[h][b]['n_samples']) / exact for b in sim_bins]
    ax1.errorbar(sim_values, values, yerr=ci, fmt='o-', color=colors[h], label=labels[h], 
                 linewidth=2, markersize=6, capsize=3, capthick=1)

ax1.set_xlabel('Cosine similarity to target token', fontsize=12)
ax1.set_ylabel('Attention (% of exact-repeat)', fontsize=12)
ax1.set_title('(a) Hash Resolution Profiles', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.set_xlim(-0.05, 0.95)
ax1.set_ylim(-0.02, 0.60)
ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax1.invert_xaxis()
ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax1.grid(True, alpha=0.2)

# RIGHT PANEL: FP rate at each similarity level
for h in bloom_heads:
    fp_rates = [agg[h][b]['fp_rate'] for b in sim_bins]
    # 95% CI for proportions: ±1.96 * sqrt(p*(1-p)/n)
    fp_ci = [1.96 * np.sqrt(agg[h][b]['fp_rate'] * (1 - agg[h][b]['fp_rate']) / agg[h][b]['n_samples']) for b in sim_bins]
    ax2.errorbar(sim_values, fp_rates, yerr=fp_ci, fmt='s-', color=colors[h], label=labels[h],
                 linewidth=2, markersize=6, capsize=3, capthick=1)

ax2.set_xlabel('Cosine similarity to target token', fontsize=12)
ax2.set_ylabel('False positive rate', fontsize=12)
ax2.set_title('(b) False Positive Rate by Similarity', fontsize=13, fontweight='bold')
ax2.legend(loc='upper left', fontsize=10)
ax2.set_xlim(-0.05, 0.95)
ax2.set_ylim(-0.02, 1.0)
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax2.invert_xaxis()
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax2.grid(True, alpha=0.2)

# Add annotation about the "hash bandwidth" region
for ax in [ax1, ax2]:
    ax.axvspan(0.4, 0.95, alpha=0.05, color='red', zorder=0)
    
plt.tight_layout()
plt.savefig('../figures/fig_similarity_sweep.pdf', dpi=300, bbox_inches='tight')
plt.savefig('../figures/fig_similarity_sweep.png', dpi=150, bbox_inches='tight')
print('Saved fig_similarity_sweep.pdf and .png')

# Also print a compact summary table
print('\n\nCOMPACT SUMMARY:')
print(f'{"Head":<8} {"50% attn at":<15} {"FP drops to 0 at":<18} {"Synonym FP%":<12}')
print('-' * 55)
for h in bloom_heads:
    exact = agg[h]['exact']['mean_attention']
    # Find where attention drops below 50% of exact
    half_point = "< 0.9"
    for i, b in enumerate(sim_bins):
        val = agg[h][b]['mean_attention'] / exact
        if val < 0.5:
            if i > 0:
                half_point = f"~{sim_values[i-1]:.1f}"
            break
    # Find where FP rate drops to 0
    zero_fp = "> 0.9"
    for i, b in enumerate(sim_bins):
        fp = agg[h][b]['fp_rate']
        if fp == 0:
            zero_fp = f"~{sim_values[i]:.1f}"
            break
    syn_fp = agg[h]['synonym']['fp_rate'] * 100
    print(f'{h:<8} {half_point:<15} {zero_fp:<18} {syn_fp:.0f}%')
