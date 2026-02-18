"""
EXP-1: Competing Model Fits for Bloom Filter Capacity Curve
Compare 5 models against the observed FP rates for each head.
"""
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import comb
import warnings
warnings.filterwarnings('ignore')

# Load data
with open('../results/experiment3_capacity.json') as f:
    data = json.load(f)

n_unique = np.array(data['capacity_levels'], dtype=float)

# --- Model definitions ---

def bloom_filter(n, m, k):
    """Classic Bloom filter FP rate: (1 - e^(-kn/m))^k"""
    return (1 - np.exp(-k * n / m)) ** k

def softmax_dilution(n, alpha):
    """1-param: FP = 1 - 1/(1 + alpha*n). As n grows, attention dilutes."""
    return 1 - 1 / (1 + alpha * n)

def logistic(n, L, k_log, n0):
    """3-param logistic: FP = L / (1 + exp(-k*(n-n0)))"""
    return L / (1 + np.exp(-k_log * (n - n0)))

def power_law(n, a, b):
    """2-param power law: FP = a * n^b, clipped to [0,1]"""
    return np.clip(a * n**b, 0, 1)

def linear_model(n, a, b):
    """2-param linear: FP = a*n + b, clipped to [0,1]"""
    return np.clip(a * n + b, 0, 1)

# --- AIC/BIC ---

def compute_aic_bic(y_obs, y_pred, k_params):
    """Compute AIC and BIC from residuals."""
    n_pts = len(y_obs)
    residuals = y_obs - y_pred
    ss_res = np.sum(residuals**2)
    # Use RSS-based likelihood (assumes Gaussian errors)
    if ss_res == 0:
        ss_res = 1e-15
    log_likelihood = -n_pts/2 * np.log(2*np.pi*ss_res/n_pts) - n_pts/2
    aic = 2*k_params - 2*log_likelihood
    bic = k_params*np.log(n_pts) - 2*log_likelihood
    # Also compute R^2
    ss_tot = np.sum((y_obs - np.mean(y_obs))**2)
    r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0
    # Adjusted R^2
    if n_pts > k_params + 1:
        r2_adj = 1 - (1-r2)*(n_pts-1)/(n_pts-k_params-1)
    else:
        r2_adj = r2
    return {
        'aic': aic, 'bic': bic, 'r2': r2, 'r2_adj': r2_adj,
        'ss_res': ss_res, 'n_params': k_params
    }

# --- Fit each model to each head ---

results = {}

for head_name, hdata in data['bloom_heads'].items():
    fp = np.array(hdata['fp_rate'], dtype=float)
    fits = {}
    
    # 1. Bloom filter (2 params: m, k)
    try:
        popt, _ = curve_fit(bloom_filter, n_unique, fp, p0=[60, 2], 
                           bounds=([1, 0.1], [500, 20]), maxfev=10000)
        y_pred = bloom_filter(n_unique, *popt)
        stats = compute_aic_bic(fp, y_pred, 2)
        fits['bloom'] = {'params': {'m': popt[0], 'k': popt[1]}, **stats}
    except Exception as e:
        fits['bloom'] = {'error': str(e)}
    
    # 2. Softmax dilution (1 param: alpha)
    try:
        popt, _ = curve_fit(softmax_dilution, n_unique, fp, p0=[0.01],
                           bounds=([0], [1]), maxfev=10000)
        y_pred = softmax_dilution(n_unique, *popt)
        stats = compute_aic_bic(fp, y_pred, 1)
        fits['softmax_dilution'] = {'params': {'alpha': popt[0]}, **stats}
    except Exception as e:
        fits['softmax_dilution'] = {'error': str(e)}
    
    # 3. Logistic (3 params: L, k, n0)
    try:
        popt, _ = curve_fit(logistic, n_unique, fp, p0=[1, 0.05, 50],
                           bounds=([0, 0, -100], [1.5, 1, 300]), maxfev=10000)
        y_pred = logistic(n_unique, *popt)
        stats = compute_aic_bic(fp, y_pred, 3)
        fits['logistic'] = {'params': {'L': popt[0], 'k': popt[1], 'n0': popt[2]}, **stats}
    except Exception as e:
        fits['logistic'] = {'error': str(e)}
    
    # 4. Power law (2 params: a, b)
    try:
        popt, _ = curve_fit(power_law, n_unique, fp, p0=[0.001, 1],
                           bounds=([0, 0], [10, 5]), maxfev=10000)
        y_pred = power_law(n_unique, *popt)
        stats = compute_aic_bic(fp, y_pred, 2)
        fits['power_law'] = {'params': {'a': popt[0], 'b': popt[1]}, **stats}
    except Exception as e:
        fits['power_law'] = {'error': str(e)}
    
    # 5. Linear (2 params: a, b)
    try:
        popt, _ = curve_fit(linear_model, n_unique, fp, p0=[0.005, 0],
                           bounds=([-0.1, -1], [0.1, 1]), maxfev=10000)
        y_pred = linear_model(n_unique, *popt)
        stats = compute_aic_bic(fp, y_pred, 2)
        fits['linear'] = {'params': {'a': popt[0], 'b': popt[1]}, **stats}
    except Exception as e:
        fits['linear'] = {'error': str(e)}
    
    results[head_name] = fits

# --- Optimal k analysis (EXP-2) ---
optimal_k_analysis = {}
for head_name, fits in results.items():
    if 'bloom' in fits and 'params' in fits['bloom']:
        m_fit = fits['bloom']['params']['m']
        k_fit = fits['bloom']['params']['k']
        # For typical context lengths
        for n_typical in [10, 20, 50, 100]:
            k_optimal = (m_fit / n_typical) * np.log(2)
            optimal_k_analysis.setdefault(head_name, []).append({
                'n': n_typical, 'k_fitted': k_fit, 'k_optimal': round(k_optimal, 3),
                'ratio': round(k_fit / k_optimal, 3) if k_optimal > 0 else None
            })

# --- Print results ---
print("=" * 80)
print("COMPETING MODEL FITS — RESULTS")
print("=" * 80)

for head_name, fits in results.items():
    fp = data['bloom_heads'][head_name]['fp_rate']
    print(f"\n{'='*60}")
    print(f"HEAD: {head_name}")
    print(f"FP rates: {fp}")
    print(f"{'='*60}")
    
    # Rank by AIC
    valid = {k: v for k, v in fits.items() if 'aic' in v}
    ranked = sorted(valid.items(), key=lambda x: x[1]['aic'])
    
    print(f"\n{'Model':<22} {'AIC':>8} {'BIC':>8} {'R²':>7} {'R²adj':>7} {'Params':>6} {'ΔAIC':>7}")
    print("-" * 70)
    best_aic = ranked[0][1]['aic'] if ranked else 0
    for name, stats in ranked:
        delta = stats['aic'] - best_aic
        winner = " ◀ BEST" if delta == 0 else ""
        print(f"{name:<22} {stats['aic']:>8.2f} {stats['bic']:>8.2f} {stats['r2']:>7.4f} {stats['r2_adj']:>7.4f} {stats['n_params']:>6} {delta:>7.2f}{winner}")
    
    # Print params for top model
    if ranked:
        best_name, best_stats = ranked[0]
        print(f"\nBest model: {best_name}")
        print(f"Parameters: {fits[best_name].get('params', {})}")

# Print optimal k analysis
print(f"\n{'='*80}")
print("OPTIMAL k ANALYSIS")
print("="*80)
for head_name, analyses in optimal_k_analysis.items():
    print(f"\n{head_name} (fitted k = {analyses[0]['k_fitted']:.3f}):")
    for a in analyses:
        print(f"  n={a['n']:>3}: k* = {a['k_optimal']:.3f}, ratio k/k* = {a['ratio']}")

# Save full results
output = {
    'model_fits': {},
    'optimal_k': optimal_k_analysis,
    'capacity_levels': data['capacity_levels']
}
for head, fits in results.items():
    output['model_fits'][head] = {}
    for model, stats in fits.items():
        if 'params' in stats:
            # Convert numpy to float
            params = {k: float(v) for k, v in stats['params'].items()}
            output['model_fits'][head][model] = {
                'params': params,
                'aic': float(stats['aic']),
                'bic': float(stats['bic']),
                'r2': float(stats['r2']),
                'r2_adj': float(stats['r2_adj']),
                'n_params': stats['n_params']
            }

with open('../results/competing_model_fits.json', 'w') as f:
    json.dump(output, f, indent=2)
print("\n\nResults saved to ../results/competing_model_fits.json")
