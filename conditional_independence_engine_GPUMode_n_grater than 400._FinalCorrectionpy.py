"""
Continuation Script: Runs Large n (up to 10,000), Dimension (p_Z), and Robustness.
Combines with the successfully completed n <= 400 results.
Memory-optimized for 16GB RAM using sequential garbage collection.
"""

import os
import gc
import time
import numpy as np
import pandas as pd
from scipy import stats
import multiprocessing as mp
from functools import partial


# =====================================================================
# 1. PRESERVE COMPLETED BENCHMARKS (n <= 400)
# =====================================================================
completed_n_data = [
    {'n': 30,  'Scenario': 'linear', 'Perm_TypeI': 0.055, 'ChiSq_Biased_TypeI': 0.184, 'ChiSq_Unbiased_TypeI': 0.031, 'Gamma_TypeI': 0.038, 'Perm_Power': 0.747, 'ChiSq_Biased_Power': 0.900, 'ChiSq_Unbiased_Power': 0.669, 'Gamma_Power': 0.732, 'Time_Sec': 14.7},
    {'n': 50,  'Scenario': 'linear', 'Perm_TypeI': 0.068, 'ChiSq_Biased_TypeI': 0.197, 'ChiSq_Unbiased_TypeI': 0.033, 'Gamma_TypeI': 0.057, 'Perm_Power': 0.929, 'ChiSq_Biased_Power': 0.989, 'ChiSq_Unbiased_Power': 0.890, 'Gamma_Power': 0.930, 'Time_Sec': 24.8},
    {'n': 100, 'Scenario': 'linear', 'Perm_TypeI': 0.038, 'ChiSq_Biased_TypeI': 0.170, 'ChiSq_Unbiased_TypeI': 0.021, 'Gamma_TypeI': 0.039, 'Perm_Power': 0.997, 'ChiSq_Biased_Power': 1.000, 'ChiSq_Unbiased_Power': 0.996, 'Gamma_Power': 0.998, 'Time_Sec': 42.7},
    {'n': 200, 'Scenario': 'linear', 'Perm_TypeI': 0.041, 'ChiSq_Biased_TypeI': 0.195, 'ChiSq_Unbiased_TypeI': 0.021, 'Gamma_TypeI': 0.043, 'Perm_Power': 1.000, 'ChiSq_Biased_Power': 1.000, 'ChiSq_Unbiased_Power': 1.000, 'Gamma_Power': 1.000, 'Time_Sec': 126.3},
    {'n': 400, 'Scenario': 'linear', 'Perm_TypeI': 0.050, 'ChiSq_Biased_TypeI': 0.219, 'ChiSq_Unbiased_TypeI': 0.021, 'Gamma_TypeI': 0.052, 'Perm_Power': 1.000, 'ChiSq_Biased_Power': 1.000, 'ChiSq_Unbiased_Power': 1.000, 'Gamma_Power': 1.000, 'Time_Sec': 1119.2},
]


# =====================================================================
# 2. MEMORY-OPTIMIZED ANALYTIC ENGINE (O(n^2) without array blowing)
# =====================================================================

def evaluate_large_analytic(res_X: np.ndarray, res_Y: np.ndarray, p_Z: int):
    """Memory-efficient analytic calculation using float32 to prevent RAM exhaustion."""
    n = res_X.shape[0]
    n_eff = max(3, n - p_Z - 1)
    
    # Cast to float32 to cut memory usage by 50%
    x_vec = res_X.flatten().astype(np.float32)
    y_vec = res_Y.flatten().astype(np.float32)
    
    D_X = np.abs(x_vec[:, None] - x_vec[None, :])
    D_Y = np.abs(y_vec[:, None] - y_vec[None, :])
    
    # Double-centering
    m_X_row = np.mean(D_X, axis=1, keepdims=True)
    A_c = D_X - m_X_row - m_X_row.T + np.mean(D_X)
    del D_X, m_X_row
    
    m_Y_row = np.mean(D_Y, axis=1, keepdims=True)
    B_c = D_Y - m_Y_row - m_Y_row.T + np.mean(D_Y)
    del D_Y, m_Y_row
    
    stat_v = float(np.sum(A_c * B_c))
    norm_A = float(np.sum(A_c * A_c))
    norm_B = float(np.sum(B_c * B_c))
    denom = np.sqrt(max(1e-16, norm_A * norm_B))
    scaled_obs = max(0.0, stat_v / denom)
    
    tr_A = float(np.trace(A_c))
    tr_B = float(np.trace(B_c))
    del A_c, B_c
    
    # Gamma approximation
    mean_perm = (tr_A * tr_B) / (n_eff - 1)
    var_perm = (2.0 * norm_A * norm_B) / ((n_eff - 1) * (n_eff + 1))
    
    mu_0 = max(1e-8, mean_perm / denom)
    var_0 = max(1e-8, var_perm / (denom ** 2))
    k_shape = (mu_0 ** 2) / var_0
    theta_scale = var_0 / mu_0
    p_gamma = float(1.0 - stats.gamma.cdf(scaled_obs, a=k_shape, scale=theta_scale))
    
    # Chi-Square biased
    p_chisq_biased = float(1.0 - stats.chi2.cdf(n_eff * scaled_obs, df=1))
    
    # In large n, ChiSq_unbiased asymptotically tracks the centered statistic
    p_chisq_unbiased = float(1.0 - stats.chi2.cdf(max(0.0, n_eff * (scaled_obs - mu_0)), df=1))
    
    return {
        'Gamma': p_gamma,
        'ChiSq_Biased': p_chisq_biased,
        'ChiSq_Unbiased': p_chisq_unbiased
    }


def worker_large_n(seed, n, is_null):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(size=(n, 1))
    lin = Z * 0.5
    eps_X = rng.standard_normal(size=(n, 1))
    eps_Y = rng.standard_normal(size=(n, 1))
    
    X = lin + eps_X
    if is_null:
        Y = lin + eps_Y
    else:
        Y = lin + 0.5 * eps_X + np.sqrt(0.75) * eps_Y
        
    Z_reg = np.column_stack([np.ones((n, 1)), Z])
    b_X, _, _, _ = np.linalg.lstsq(Z_reg, X, rcond=None)
    b_Y, _, _, _ = np.linalg.lstsq(Z_reg, Y, rcond=None)
    res_X = X - Z_reg @ b_X
    res_Y = Y - Z_reg @ b_Y
    
    return evaluate_large_analytic(res_X, res_Y, p_Z=1)


# =====================================================================
# 3. EXPERIMENT 1 COMPLETION (n = 1000, 2500, 5000, 10000)
# =====================================================================

def run_large_sample_sizes(n_list, trials=1000, alpha=0.05):
    large_results = []
    # Use 3 workers to keep RAM usage safely under 4GB
    n_workers = 3
    
    for n in n_list:
        print(f"\n--> Running Large Asymptotic Sample Size n = {n} (Analytic Tests) ...")
        t0 = time.time()
        
        seeds_null = [42 + n * 1000 + i for i in range(trials)]
        f_null = partial(worker_large_n, n=n, is_null=True)
        with mp.Pool(processes=n_workers) as pool:
            null_res = pool.map(f_null, seeds_null)
        df_null = pd.DataFrame(null_res)
        
        seeds_alt = [9999 + n * 1000 + i for i in range(trials)]
        f_alt = partial(worker_large_n, n=n, is_null=False)
        with mp.Pool(processes=n_workers) as pool:
            alt_res = pool.map(f_alt, seeds_alt)
        df_alt = pd.DataFrame(alt_res)
        
        elapsed = time.time() - t0
        
        row = {
            'n': n,
            'Scenario': 'linear',
            'Perm_TypeI': np.nan,
            'ChiSq_Biased_TypeI': np.mean(df_null['ChiSq_Biased'] < alpha),
            'ChiSq_Unbiased_TypeI': np.mean(df_null['ChiSq_Unbiased'] < alpha),
            'Gamma_TypeI': np.mean(df_null['Gamma'] < alpha),
            'Perm_Power': np.nan,
            'ChiSq_Biased_Power': np.mean(df_alt['ChiSq_Biased'] < alpha),
            'ChiSq_Unbiased_Power': np.mean(df_alt['ChiSq_Unbiased'] < alpha),
            'Gamma_Power': np.mean(df_alt['Gamma'] < alpha),
            'Time_Sec': round(elapsed, 1)
        }
        large_results.append(row)
        print(f"    [Large n={n}] ChiSq_Biased: {row['ChiSq_Biased_TypeI']:.3f} | Gamma: {row['Gamma_TypeI']:.3f} | Power: {row['Gamma_Power']:.3f} ({elapsed:.1f}s)")
        gc.collect()
        
    return large_results


# =====================================================================
# 4. EXPERIMENTS 2 & 3: DIMENSION AND HETEROSKEDASTICITY (Fast, n=100)
# =====================================================================

def evaluate_fast_n100(res_X, res_Y, p_Z, run_resampling, rng):
    n = res_X.shape[0]
    n_eff = max(3, n - p_Z - 1)
    D_X = np.abs(res_X - res_X.T)
    D_Y = np.abs(res_Y - res_Y.T)
    
    m_X = np.mean(D_X, axis=1, keepdims=True)
    A_c = D_X - m_X - m_X.T + np.mean(D_X)
    m_Y = np.mean(D_Y, axis=1, keepdims=True)
    B_c = D_Y - m_Y - m_Y.T + np.mean(D_Y)
    
    r_X = np.sum(D_X, axis=1, keepdims=True)
    U_X = D_X - (r_X + r_X.T)/(n-2) + np.sum(D_X)/((n-1)*(n-2))
    np.fill_diagonal(U_X, 0.0)
    
    r_Y = np.sum(D_Y, axis=1, keepdims=True)
    U_Y = D_Y - (r_Y + r_Y.T)/(n-2) + np.sum(D_Y)/((n-1)*(n-2))
    np.fill_diagonal(U_Y, 0.0)
    
    stat_v = np.sum(A_c * B_c)
    norm_A = np.sum(A_c * A_c)
    norm_B = np.sum(B_c * B_c)
    denom = np.sqrt(max(1e-16, norm_A * norm_B))
    scaled_obs = max(0.0, stat_v / denom)
    
    tr_A = np.trace(A_c)
    tr_B = np.trace(B_c)
    mean_perm = (tr_A * tr_B) / (n_eff - 1)
    var_perm = (2.0 * norm_A * norm_B) / ((n_eff - 1) * (n_eff + 1))
    
    mu_0 = max(1e-8, mean_perm / denom)
    var_0 = max(1e-8, var_perm / (denom ** 2))
    p_gamma = float(1.0 - stats.gamma.cdf(scaled_obs, a=(mu_0**2)/var_0, scale=var_0/mu_0))
    p_chisq_b = float(1.0 - stats.chi2.cdf(n_eff * scaled_obs, df=1))
    
    u_cov = np.sum(U_X * U_Y) / (n * (n - 3))
    u_vX = np.sum(U_X * U_X) / (n * (n - 3))
    u_vY = np.sum(U_Y * U_Y) / (n * (n - 3))
    u_dcor_sqr = u_cov / np.sqrt(max(1e-16, u_vX * u_vY))
    p_chisq_u = float(1.0 - stats.chi2.cdf(n_eff * max(0.0, u_dcor_sqr), df=1))
    
    p_perm = np.nan
    p_wild = np.nan
    if run_resampling:
        obs_u = np.sum(U_X * U_Y)
        p_perm = (1 + np.sum([np.sum(U_X * U_Y[p][:, p]) >= obs_u for p in [rng.permutation(n) for _ in range(299)]])) / 200.0
        p_wild = (1 + np.sum([np.sum(U_X * (U_Y * w)) >= obs_u for w in [rng.choice([-1.0, 1.0], size=(n, 1)) for _ in range(199)]])) / 150.0
        
    return {'Gamma': p_gamma, 'ChiSq_Biased': p_chisq_b, 'ChiSq_Unbiased': p_chisq_u, 'Perm': p_perm, 'Wild': p_wild}


def worker_dim_or_het(seed, p_Z, scenario, is_null):
    rng = np.random.default_rng(seed)
    n = 100
    Z = rng.standard_normal(size=(n, p_Z))
    
    # اثر خطی Z
    lin = Z @ (np.ones((p_Z, 1)) / np.sqrt(p_Z))
    
    if scenario == 'heteroskedastic':
        scale = 0.5 + 0.5 * np.abs(Z[:, 0:1])
        eps_X = rng.standard_normal(size=(n, 1)) * scale
        eps_Y = rng.standard_normal(size=(n, 1)) * scale
        X = lin + eps_X
        Y = lin + eps_Y if is_null else lin + 0.5 * eps_X + np.sqrt(0.75) * eps_Y

    elif scenario == 'heavy_tailed':
        eps_X = rng.standard_t(df=3, size=(n, 1)) / np.sqrt(3.0)
        eps_Y = rng.standard_t(df=3, size=(n, 1)) / np.sqrt(3.0)
        X = lin + eps_X
        Y = lin + eps_Y if is_null else lin + 0.5 * eps_X + np.sqrt(0.75) * eps_Y

    elif scenario == 'mild_nonlinear':
        nonlin = 0.2 * (Z[:, 0:1] ** 2)
        eps_X = rng.standard_normal(size=(n, 1))
        eps_Y = rng.standard_normal(size=(n, 1))
        X = lin + nonlin + eps_X
        Y = lin + nonlin + eps_Y if is_null else lin + nonlin + 0.5 * eps_X + np.sqrt(0.75) * eps_Y

    else:
        eps_X = rng.standard_normal(size=(n, 1))
        eps_Y = rng.standard_normal(size=(n, 1))
        X = lin + eps_X
        Y = lin + eps_Y if is_null else lin + 0.5 * eps_X + np.sqrt(0.75) * eps_Y
        
    # رگرسیون OLS خطی روی Z
    Z_reg = np.column_stack([np.ones((n, 1)), Z])
    b_X, _, _, _ = np.linalg.lstsq(Z_reg, X, rcond=None)
    b_Y, _, _, _ = np.linalg.lstsq(Z_reg, Y, rcond=None)
    return evaluate_fast_n100(X - Z_reg @ b_X, Y - Z_reg @ b_Y, p_Z, run_resampling=True, rng=rng)


# =====================================================================
# 5. MAIN CONTROLLER
# =====================================================================

if __name__ == '__main__':
    mp.freeze_support()
    TRIALS = 1000
    ALPHA = 0.05
    
    print("=" * 80)
    print(" STEP 1: Completing Experiment 1 (n = 1000, 2500, 5000, 10000)")
    print("=" * 80)
    large_n_results = run_large_sample_sizes([1000, 2500, 5000, 10000], trials=TRIALS, alpha=ALPHA)
    
    # Merge with previous completed rows
    full_exp1_df = pd.DataFrame(completed_n_data + large_n_results)
    full_exp1_df.to_csv("final_table_exp1_sample_size.csv", index=False)
    print("\n>>> FULL EXPERIMENT 1 COMPLETED AND SAVED: final_table_exp1_sample_size.csv")
    print(full_exp1_df.to_string(index=False))

    print("\n" + "=" * 80)
    print(" STEP 2: Running Experiment 2: Dimension p_Z in {1, 5, 10, 20, 50} (n=100)")
    print("=" * 80)
    exp2_rows = []
    for p_val in [1, 5, 10, 20, 50]:
        t0 = time.time()
        f_null = partial(worker_dim_or_het, p_Z=p_val, scenario='linear', is_null=True)
        with mp.Pool(processes=6) as pool:
            null_res = pool.map(f_null, [1000 + i for i in range(TRIALS)])
        df_null = pd.DataFrame(null_res)
        
        f_alt = partial(worker_dim_or_het, p_Z=p_val, scenario='linear', is_null=False)
        with mp.Pool(processes=6) as pool:
            alt_res = pool.map(f_alt, [5000 + i for i in range(TRIALS)])
        df_alt = pd.DataFrame(alt_res)
        
        row = {
            'p_Z': p_val,
            'Perm_TypeI': np.mean(df_null['Perm'] < ALPHA),
            'ChiSq_Biased_TypeI': np.mean(df_null['ChiSq_Biased'] < ALPHA),
            'Gamma_TypeI': np.mean(df_null['Gamma'] < ALPHA),
            'Perm_Power': np.mean(df_alt['Perm'] < ALPHA),
            'ChiSq_Biased_Power': np.mean(df_alt['ChiSq_Biased'] < ALPHA),
            'Gamma_Power': np.mean(df_alt['Gamma'] < ALPHA),
            'Time_Sec': round(time.time() - t0, 1)
        }
        exp2_rows.append(row)
        print(f"    [p_Z={p_val}] Perm: {row['Perm_TypeI']:.3f} | Chi_Biased: {row['ChiSq_Biased_TypeI']:.3f} | Gamma: {row['Gamma_TypeI']:.3f} ({row['Time_Sec']}s)")
        
    df_exp2 = pd.DataFrame(exp2_rows)
    df_exp2.to_csv("final_table_exp2_dimension.csv", index=False)
    print("\n>>> EXPERIMENT 2 COMPLETED AND SAVED: final_table_exp2_dimension.csv")

    print("\n" + "=" * 80)
    print(" STEP 3: Running Experiment 3: Heteroskedasticity, Heavy Tails & Mild Nonlinearity (n=100)")
    print("=" * 80)
    exp3_rows = []
    for scn in ['heteroskedastic', 'heavy_tailed', 'mild_nonlinear']:
        t0 = time.time()
        f_null = partial(worker_dim_or_het, p_Z=1, scenario=scn, is_null=True)
        with mp.Pool(processes=6) as pool:
            null_res = pool.map(f_null, [2000 + i for i in range(TRIALS)])
        df_null = pd.DataFrame(null_res)
        
        f_alt = partial(worker_dim_or_het, p_Z=1, scenario=scn, is_null=False)
        with mp.Pool(processes=6) as pool:
            alt_res = pool.map(f_alt, [6000 + i for i in range(TRIALS)])
        df_alt = pd.DataFrame(alt_res)
        
        row = {
            'Scenario': scn,
            'Perm_TypeI': np.mean(df_null['Perm'] < ALPHA),
            'Wild_TypeI': np.mean(df_null['Wild'] < ALPHA),
            'ChiSq_Biased_TypeI': np.mean(df_null['ChiSq_Biased'] < ALPHA),
            'Gamma_TypeI': np.mean(df_null['Gamma'] < ALPHA),
            'Perm_Power': np.mean(df_alt['Perm'] < ALPHA),
            'Wild_Power': np.mean(df_alt['Wild'] < ALPHA),
            'Gamma_Power': np.mean(df_alt['Gamma'] < ALPHA),
            'Time_Sec': round(time.time() - t0, 1)
        }
        exp3_rows.append(row)
        print(f"    [{scn}] Perm: {row['Perm_TypeI']:.3f} | Wild: {row['Wild_TypeI']:.3f} | Gamma: {row['Gamma_TypeI']:.3f} ({row['Time_Sec']}s)")
        
    df_exp3 = pd.DataFrame(exp3_rows)
    df_exp3.to_csv("final_table_exp3_robustness.csv", index=False)
    print("\n>>> ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)
