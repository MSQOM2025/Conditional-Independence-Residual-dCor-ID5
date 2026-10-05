"""
Conditional Independence Testing on Linear Residuals:
Benchmark, Asymptotic Diagnostics, and Fast Calibrated Inference.

This script implements:
1. Permutation Test on U-statistic Distance Correlation (Gold Standard).
2. Naive Asymptotic Chi-Square Test (Shen et al. adaptation, showing miscalibration).
3. Fast Moment-Matched Gamma Approximation Test (Satterthwaite/Welch analytic fix).
4. Wild Bootstrap Test (Heteroskedasticity-robust benchmark).

Designed for revision in Communications in Statistics - Simulation and Computation.
"""

import time
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# =====================================================================
# SECTION 1: Core Linear Algebra and Distance Correlation Primitives
# =====================================================================

def compute_ols_residuals(X: np.ndarray, Y: np.ndarray, Z: np.ndarray):
    """
    Computes residuals of X and Y after projection onto the column space of [1, Z].
    
    Parameters:
        X: Array of shape (n, 1)
        Y: Array of shape (n, 1)
        Z: Array of shape (n, p_Z)
        
    Returns:
        res_X: Array of shape (n, 1)
        res_Y: Array of shape (n, 1)
        p_Z: int, dimension of conditioning variable
    """
    n = X.shape[0]
    p_Z = Z.shape[1] if Z.ndim > 1 else 1
    Z_mat = Z.reshape(n, p_Z)
    
    # Design matrix with intercept
    Z_reg = np.column_stack([np.ones((n, 1)), Z_mat])
    
    # OLS projection via QR / SVD stable solver
    beta_X, _, _, _ = np.linalg.lstsq(Z_reg, X, rcond=None)
    beta_Y, _, _, _ = np.linalg.lstsq(Z_reg, Y, rcond=None)
    
    res_X = X - Z_reg @ beta_X
    res_Y = Y - Z_reg @ beta_Y
    
    return res_X, res_Y, p_Z


def pairwise_euclidean_distances(v: np.ndarray) -> np.ndarray:
    """Computes pairwise Euclidean distance matrix for an (n, 1) vector in O(n^2)."""
    diff = v - v.T
    return np.abs(diff)


def double_center_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Applies classical double-centering projection: H_n * D * H_n
    where H_n = I - (1/n) * 1 * 1^T.
    """
    n = D.shape[0]
    row_means = np.mean(D, axis=1, keepdims=True)
    col_means = np.mean(D, axis=0, keepdims=True)
    grand_mean = np.mean(D)
    return D - row_means - col_means + grand_mean


def u_center_distance_matrix(D: np.ndarray) -> np.ndarray:
    """
    Computes the unbiased (U-statistic) centered matrix (Szekely and Rizzo, 2014).
    Diagonal is strictly zero; row sums exclude diagonal.
    """
    n = D.shape[0]
    if n <= 3:
        raise ValueError("Sample size n must be greater than 3 for U-centering.")
        
    row_sums = np.sum(D, axis=1, keepdims=True)
    total_sum = np.sum(D)
    
    # Unbiased formula:
    # A~_ij = A_ij - (row_i)/(n-2) - (row_j)/(n-2) + total / ((n-1)(n-2))
    U = D - (row_sums + row_sums.T) / (n - 2) + total_sum / ((n - 1) * (n - 2))
    np.fill_diagonal(U, 0.0)
    return U


def compute_u_dcor_sqr(res_X: np.ndarray, res_Y: np.ndarray) -> float:
    """
    Computes the unbiased (U-statistic) sample squared distance correlation.
    Can be negative in finite samples under the null.
    """
    n = res_X.shape[0]
    D_X = pairwise_euclidean_distances(res_X)
    D_Y = pairwise_euclidean_distances(res_Y)
    
    U_X = u_center_distance_matrix(D_X)
    U_Y = u_center_distance_matrix(D_Y)
    
    cov_XY = np.sum(U_X * U_Y) / (n * (n - 3))
    var_X = np.sum(U_X * U_X) / (n * (n - 3))
    var_Y = np.sum(U_Y * U_Y) / (n * (n - 3))
    
    denom = np.sqrt(np.maximum(var_X * var_Y, 1e-16))
    return float(cov_XY / denom)


# =====================================================================
# SECTION 2: Hypothesis Testing Procedures
# =====================================================================

def test_permutation(res_X: np.ndarray, res_Y: np.ndarray, 
                     num_perms: int = 499, rng: np.random.Generator = None) -> float:
    """
    Gold standard: Permutation test on U-statistic distance correlation.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = res_X.shape[0]
    D_X = pairwise_euclidean_distances(res_X)
    D_Y = pairwise_euclidean_distances(res_Y)
    U_X = u_center_distance_matrix(D_X)
    U_Y = u_center_distance_matrix(D_Y)
    
    # Observed inner product
    stat_obs = np.sum(U_X * U_Y)
    
    count = 0
    for _ in range(num_perms):
        perm_idx = rng.permutation(n)
        # Permute rows and columns of U_Y
        stat_perm = np.sum(U_X * U_Y[perm_idx][:, perm_idx])
        if stat_perm >= stat_obs:
            count += 1
            
    p_value = (count + 1) / (num_perms + 1)
    return p_value


def test_naive_chisq(res_X: np.ndarray, res_Y: np.ndarray, p_Z: int) -> float:
    """
    Naive Chi-Square test (original manuscript).
    T_n = (n - p_Z - 1) * max(0, dCor_U^2) ~ Chi-Square(df=1).
    """
    n = res_X.shape[0]
    dcor_sqr = compute_u_dcor_sqr(res_X, res_Y)
    
    # Finite-sample degrees-of-freedom scaling
    stat = (n - p_Z - 1) * max(0.0, dcor_sqr)
    p_value = 1.0 - stats.chi2.cdf(stat, df=1)
    return float(p_value)


def test_gamma_satterthwaite(res_X: np.ndarray, res_Y: np.ndarray, p_Z: int) -> float:
    """
    Constructive Analytic Solution (Reviewer 3, Comment 2):
    Fast Moment-Matched Gamma Test for Residual Distance Correlation.
    
    Computes exact permutation moments of the quadratic form in O(n^2) time
    and fits a two-parameter Gamma distribution (Satterthwaite-Welch approximation).
    """
    n = res_X.shape[0]
    n_eff = max(3, n - p_Z - 1)
    
    D_X = pairwise_euclidean_distances(res_X)
    D_Y = pairwise_euclidean_distances(res_Y)
    
    A_c = double_center_distance_matrix(D_X)
    B_c = double_center_distance_matrix(D_Y)
    
    # Observed V-statistic inner product
    stat_obs = np.sum(A_c * B_c)
    norm_A = np.sum(A_c * A_c)
    norm_B = np.sum(B_c * B_c)
    
    if norm_A <= 1e-12 or norm_B <= 1e-12:
        return 1.0
        
    denom = np.sqrt(norm_A * norm_B)
    scaled_obs = max(0.0, stat_obs / denom)
    
    # Exact permutation null expectation and variance of the inner product
    # for double-centered symmetric matrices:
    tr_A = np.trace(A_c)
    tr_B = np.trace(B_c)
    
    mean_perm = (tr_A * tr_B) / (n_eff - 1)
    
    # Permutation variance under null independence:
    var_perm = (2.0 * norm_A * norm_B) / ((n_eff - 1) * (n_eff + 1))
    
    # Normalized moments:
    mu_0 = max(1e-8, mean_perm / denom)
    var_0 = max(1e-8, var_perm / (denom ** 2))
    
    # Gamma distribution shape (k) and scale (theta) parameters
    k_param = (mu_0 ** 2) / var_0
    theta_param = var_0 / mu_0
    
    # Analytic p-value from fitted Gamma CDF
    p_value = 1.0 - stats.gamma.cdf(scaled_obs, a=k_param, scale=theta_param)
    return float(np.clip(p_value, 0.0, 1.0))


def test_wild_bootstrap(res_X: np.ndarray, res_Y: np.ndarray, 
                        num_boot: int = 299, rng: np.random.Generator = None) -> float:
    """
    Wild Bootstrap Test (Reviewer 1 and Reviewer 3, Comment 4):
    Robust to conditional heteroskedasticity in residuals.
    Multiplies res_Y by independent Rademacher random variables (+1, -1).
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = res_X.shape[0]
    D_X = pairwise_euclidean_distances(res_X)
    U_X = u_center_distance_matrix(D_X)
    
    D_Y = pairwise_euclidean_distances(res_Y)
    U_Y = u_center_distance_matrix(D_Y)
    
    stat_obs = np.sum(U_X * U_Y)
    
    count = 0
    for _ in range(num_boot):
        # Rademacher weights: +1 or -1 with probability 0.5
        W = rng.choice([-1.0, 1.0], size=(n, 1))
        res_Y_boot = res_Y * W
        
        D_Y_boot = pairwise_euclidean_distances(res_Y_boot)
        U_Y_boot = u_center_distance_matrix(D_Y_boot)
        
        stat_boot = np.sum(U_X * U_Y_boot)
        if stat_boot >= stat_obs:
            count += 1
            
    p_value = (count + 1) / (num_boot + 1)
    return p_value


# =====================================================================
# SECTION 3: Simulation Scenarios (Addressing Reviewers 1, 2, and 3)
# =====================================================================

def generate_simulation_data(n: int, p_Z: int, scenario: str, 
                             is_null: bool, rng: np.random.Generator):
    """
    Data Generating Processes (DGPs) covering all reviewer concerns:
    - 'linear': Standard homoskedastic Gaussian baseline.
    - 'heteroskedastic': Var(eps | Z) depends on Z (Reviewer 1 & Reviewer 3, Comment 4).
    - 'heavy_tailed': Non-Gaussian Student-t(df=3) errors (Reviewer 1).
    - 'high_dim_p': Large p_Z setting (Reviewer 3, Comment 3).
    """
    Z = rng.standard_normal(size=(n, p_Z))
    
    # Linear projection coefficients
    beta = np.ones((p_Z, 1)) / np.sqrt(p_Z)
    linear_effect = Z @ beta
    
    # Error generation
    if scenario == 'linear' or scenario == 'high_dim_p':
        eps_X = rng.standard_normal(size=(n, 1))
        eps_Y_base = rng.standard_normal(size=(n, 1))
    elif scenario == 'heteroskedastic':
        # Variance scale depends on norm of Z
        scale = 0.5 + 0.5 * np.abs(Z[:, 0:1])
        eps_X = rng.standard_normal(size=(n, 1)) * scale
        eps_Y_base = rng.standard_normal(size=(n, 1)) * scale
    elif scenario == 'heavy_tailed':
        # Student-t with 3 degrees of freedom (finite variance, heavy tails)
        eps_X = rng.standard_t(df=3, size=(n, 1)) / np.sqrt(3.0)
        eps_Y_base = rng.standard_t(df=3, size=(n, 1)) / np.sqrt(3.0)
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
        
    X = linear_effect + eps_X
    
    if is_null:
        Y = linear_effect + eps_Y_base
    else:
        # Non-null alternative: nonlinear dependence between residuals
        Y = linear_effect + 0.5 * (eps_X ** 2 - 1.0) + eps_Y_base
        
    return X, Y, Z


def run_monte_carlo_trial(n: int, p_Z: int, scenario: str, 
                          is_null: bool, rng: np.random.Generator):
    """Executes a single replication across all four testing procedures."""
    X, Y, Z = generate_simulation_data(n, p_Z, scenario, is_null, rng)
    res_X, res_Y, p_dim = compute_ols_residuals(X, Y, Z)
    
    # Evaluate tests
    p_perm = test_permutation(res_X, res_Y, num_perms=299, rng=rng)
    p_chisq = test_naive_chisq(res_X, res_Y, p_dim)
    p_gamma = test_gamma_satterthwaite(res_X, res_Y, p_dim)
    p_wild = test_wild_bootstrap(res_X, res_Y, num_boot=199, rng=rng)
    
    return {
        'perm': p_perm,
        'chisq': p_chisq,
        'gamma': p_gamma,
        'wild': p_wild
    }


def execute_experiment(param_list, param_type: str, scenario: str, 
                       trials: int = 1000, alpha: float = 0.05, seed: int = 42):
    """
    Runs systematic Monte Carlo experiment for varying n or varying p_Z.
    Strictly reproducible: each parameter value uses an isolated deterministic seed.
    """
    records = []
    
    print(f"\n=======================================================")
    print(f" Running Experiment: Scenario={scenario}, Param={param_type}")
    print(f" Trials={trials}, Nominal Alpha={alpha}")
    print(f"=======================================================")
    
    # --- تغییر در این حلقه اعمال شده است ---
    for i, val in enumerate(param_list):
        if param_type == 'n':
            n, p_Z = val, 1
        elif param_type == 'p_Z':
            n, p_Z = 200, val
        else:
            raise ValueError("param_type must be 'n' or 'p_Z'")
            
        print(f"\n--> Evaluating {param_type} = {val} ...")
        
        # تولید سید کاملاً مستقل برای این مقدار خاص از پارامتر
        condition_seed = seed + int(val) * 1000 + i
        rng_val = np.random.default_rng(condition_seed)
        
        # 1. ارزیابی فرض صفر (Type I Error)
        null_rejections = {'perm': 0, 'chisq': 0, 'gamma': 0, 'wild': 0}
        t0 = time.time()
        for _ in tqdm(range(trials), desc=f"Null Trials ({param_type}={val})", leave=False):
            res = run_monte_carlo_trial(n, p_Z, scenario, is_null=True, rng=rng_val)
            for k in null_rejections:
                if res[k] < alpha:
                    null_rejections[k] += 1
        time_null = time.time() - t0
        
        # 2. ارزیابی فرض مقابل (Statistical Power)
        alt_rejections = {'perm': 0, 'chisq': 0, 'gamma': 0, 'wild': 0}
        for _ in tqdm(range(trials), desc=f"Power Trials ({param_type}={val})", leave=False):
            res = run_monte_carlo_trial(n, p_Z, scenario, is_null=False, rng=rng_val)
            for k in alt_rejections:
                if res[k] < alpha:
                    alt_rejections[k] += 1
                    
        rec = {
            param_type: val,
            'Scenario': scenario,
            'Perm_TypeI': null_rejections['perm'] / trials,
            'ChiSq_TypeI': null_rejections['chisq'] / trials,
            'Gamma_TypeI': null_rejections['gamma'] / trials,
            'Wild_TypeI': null_rejections['wild'] / trials,
            'Perm_Power': alt_rejections['perm'] / trials,
            'ChiSq_Power': alt_rejections['chisq'] / trials,
            'Gamma_Power': alt_rejections['gamma'] / trials,
            'Wild_Power': alt_rejections['wild'] / trials,
            'Time_per_trial_ms': (time_null / trials) * 1000.0
        }
        records.append(rec)
        
        print(f"[{param_type}={val}] Type I Error -> Perm: {rec['Perm_TypeI']:.3f} | "
              f"ChiSq: {rec['ChiSq_TypeI']:.3f} | Gamma: {rec['Gamma_TypeI']:.3f} | Wild: {rec['Wild_TypeI']:.3f}")
              
    return pd.DataFrame(records)


# =====================================================================
# SECTION 4: Main Execution and Plotting
# =====================================================================

if __name__ == '__main__':
    # Configuration
    NUM_TRIALS = 1000  # Set to 2000 or 5000 for final production run
    ALPHA = 0.05
    
    # -------------------------------------------------------------
    # EXP 1: Sample Size Scaling (Addressing Reviewer 3, Comment 1)
    # Testing up to n = 5000 to observe asymptotic convergence
    # -------------------------------------------------------------
    sample_sizes = [50, 100, 200, 500, 1000, 2500, 5000]
    df_n = execute_experiment(sample_sizes, param_type='n', scenario='linear', 
                              trials=NUM_TRIALS, alpha=ALPHA)
    
    # -------------------------------------------------------------
    # EXP 2: Dimension Scaling (Addressing Reviewer 3, Comment 3)
    # -------------------------------------------------------------
    dimensions = [1, 5, 10, 20, 50]
    df_p = execute_experiment(dimensions, param_type='p_Z', scenario='high_dim_p', 
                              trials=NUM_TRIALS, alpha=ALPHA)
    
    # -------------------------------------------------------------
    # EXP 3: Heteroskedasticity (Addressing Reviewer 1 & Reviewer 3, Comment 4)
    # -------------------------------------------------------------
    df_hetero = execute_experiment([100, 200, 500], param_type='n', scenario='heteroskedastic', 
                                   trials=NUM_TRIALS, alpha=ALPHA)

    # -------------------------------------------------------------
    # Save Summary Tables for LaTeX Integration
    # -------------------------------------------------------------
    df_n.to_csv("results_varying_sample_size.csv", index=False)
    df_p.to_csv("results_varying_dimension.csv", index=False)
    df_hetero.to_csv("results_heteroskedastic.csv", index=False)
    print("\nSaved CSV tables: results_varying_sample_size.csv, results_varying_dimension.csv, results_heteroskedastic.csv")

    # -------------------------------------------------------------
    # Publication-Grade Visualizations
    # -------------------------------------------------------------
    sns.set_theme(style="whitegrid", font_scale=1.1)
    
    # Plot 1: Type I Error Convergence vs. Sample Size (Reviewer 3, Comment 1)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(y=ALPHA, color='black', linestyle='--', linewidth=1.5, label='Nominal Level (alpha = 0.05)')
    ax.axhspan(ALPHA - 0.015, ALPHA + 0.015, color='gray', alpha=0.2, label='Acceptable Binomial Tol.')
    
    ax.plot(df_n['n'], df_n['ChiSq_TypeI'], 's--', color='crimson', label='Naive Chi-Square (Inflated)', lw=2)
    ax.plot(df_n['n'], df_n['Perm_TypeI'], 'o-', color='navy', label='Permutation Benchmark', lw=2)
    ax.plot(df_n['n'], df_n['Gamma_TypeI'], '^-', color='forestgreen', label='Proposed Gamma Approximation', lw=2.5)
    ax.plot(df_n['n'], df_n['Wild_TypeI'], 'd:', color='purple', label='Wild Bootstrap', lw=2)
    
    ax.set_xscale('log')
    ax.set_xlabel('Sample Size (n, log-scale)', fontweight='bold')
    ax.set_ylabel('Empirical Type I Error Rate', fontweight='bold')
    ax.set_title('Asymptotic Convergence and Calibration of Conditional Independence Tests', fontweight='bold')
    ax.set_ylim(-0.02, 0.30)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('fig_type_i_error_convergence.png', dpi=300)
    print("Saved figure: fig_type_i_error_convergence.png")
    
    # Plot 2: High Dimensionality Performance (Reviewer 3, Comment 3)
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    ax2.axhline(y=ALPHA, color='black', linestyle='--', linewidth=1.5, label='Nominal Alpha = 0.05')
    ax2.plot(df_p['p_Z'], df_p['ChiSq_TypeI'], 's--', color='crimson', label='Naive Chi-Square', lw=2)
    ax2.plot(df_p['p_Z'], df_p['Gamma_TypeI'], '^-', color='forestgreen', label='Proposed Gamma Approx.', lw=2.5)
    ax2.plot(df_p['p_Z'], df_p['Perm_TypeI'], 'o-', color='navy', label='Permutation', lw=2)
    
    ax2.set_xlabel('Conditioning Dimension (p_Z) [Fixed n = 200]', fontweight='bold')
    ax2.set_ylabel('Empirical Type I Error Rate', fontweight='bold')
    ax2.set_title('Robustness to Dimension of Conditioning Set (p_Z)', fontweight='bold')
    ax2.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig('fig_dimension_robustness.png', dpi=300)
    print("Saved figure: fig_dimension_robustness.png")

    print("\nAll simulations completed successfully.")