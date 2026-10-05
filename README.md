# Conditional-Independence-Residual-dCor-ID5
Fast Moment-Matched Gamma Test for Residual Distance Correlation
This repository provides the official implementation, replication scripts, and simulation engines for the paper:

> Residual Distance Correlation for Conditional Independence: Limitations of Chi-Square Approximations and a Fast Gamma Alternative

 
 Overview

Permutation-based conditional independence tests using residual distance correlation are statistically robust but computationally prohibitive ($O(B \cdot n^2)$). Naive analytic shortcuts—such as adapting a one-degree-of-freedom Chi-Square ($\chi_1^2$) envelope to linear residuals—suffer from severe empirical Type I error inflation ($\approx 21\%$), a structural failure that persists even at $n = 10{,}000$.

This repository implements:
1. **The Fast Moment-Matched Gamma Test (Algorithm 1):** A closed-form, non-iterative test based on leading conditional permutation moments and a Satterthwaite-type effective degrees-of-freedom calibration ($n_{\mathrm{eff}} = n - p_Z - 1$). Operates in $O(n^2)$ time and accurately maintains nominal 5% size control across all sample sizes.
2. Wild Bootstrap Benchmark (Algorithm 2): A Rademacher-multiplier residual bootstrap tailored for conditional heteroskedasticity.
3. Simulation Engines: Replicating all Monte Carlo experiments across sample sizes up to $n = 10{,}000$, conditioning dimensions up to $p_Z = 50$, and structural model departures (heteroskedasticity, heavy tails, and mild nonlinear confounding).
4. Real Data Pipeline: Validation on the single-cell flow cytometry protein signaling dataset of Sachs et al. (2005) ($n = 853$).



 Repository Structure

├── data/
│   └── sachs_2005.csv                  # Observational flow cytometry dataset (n = 853)
├── src/
│   ├── tests.py                        # Implementation of Gamma, Chi-Square, Permutation, and Wild Bootstrap
│   ├── simulation_engine.py            # Multiprocessing Monte Carlo experiment runner
│   └── real_data_analysis.py           # Analysis script for Sachs et al. (2005) network paths
├── results/
│   ├── final_table_exp1_sample_size.csv  # Experiment 1 results (n = 30 to 10,000)
│   ├── final_table_exp2_dimension.csv    # Experiment 2 results (p_Z = 1 to 50)
│   ├── final_table_exp3_robustness.csv   # Experiment 3 results (Hetero, t3, Nonlinear)
│   └── sachs_results.csv                 # Real data application test statistics and p-values
├── figures/
│   ├── make_plots.py                   # Plotting script for Figure 1 and Figure 2
│   ├── fig_sim_exp1_convergence_n.png  # Figure 1: Convergence trajectory up to n=10,000
│   └── fig_sim_exp2_dimension_p.png    # Figure 2: Dimensional sensitivity
├── requirements.txt                    # Python package dependencies
├── LICENSE                             # MIT License
└── README.md                           # Documentation

Prerequisites and Installation

The code requires Python 3.9+ and standard scientific libraries. Clone the
repository and install the dependencies:

git clone https://github.com/MohsenSalehi-Stat/Conditional-Independence-Residual-dCor.git
cd Conditional-Independence-Residual-dCor
pip install -r requirements.txt

Dependencies (requirements.txt)

numpy>=1.22.0
scipy>=1.8.0
pandas>=1.4.0
matplotlib>=3.5.0
scikit-learn>=1.0.0

Usage and Replication

1. Quick Example: Running the Fast Gamma Test

import numpy as np
from src.tests import fast_gamma_test

# Generate synthetic data: X _|_ Y | Z
n, p_Z = 500, 2
rng = np.random.default_rng(42)
Z = rng.standard_normal(size=(n, p_Z))
X = Z @ [0.5, -0.3] + rng.standard_normal(n)
Y = Z @ [0.4,  0.6] + rng.standard_normal(n)

# Run proposed Gamma test (closed-form, no permutations required)
result = fast_gamma_test(X, Y, Z)
print(f"Statistic (S_obs): {result['stat']:.4f}")
print(f"p-value:          {result['p_value']:.4f}")

2. Reproducing Synthetic Simulations (Section 3)

To execute the complete simulation study across all experiments:

python src/simulation_engine.py

  - Experiment 1 (Sample size scaling): Compares empirical Type I error and
    power for n \in \{30, 50, 100, 200, 400, 1000, 2500, 5000, 10000\} with
    M = 1{,}000 independent trials.
  - Experiment 2 (High-dimensional sensitivity): Varies
    p_Z \in \{1, 5, 10, 20, 50\} for fixed n = 100.
  - Experiment 3 (Structural model departures): Benchmarks under conditional
    heteroskedasticity
    (\mathrm{Var}(\varepsilon \mid Z) \propto (0.5 + 0.5|Z|)^2), heavy-tailed
    Student-t_3 errors, and mild quadratic confounding (Z + 0.2 Z^2).

Output CSV files are stored in results/.

3. Generating Figures 1 and 2

To generate the publication-ready figures from the simulation results:

python figures/make_plots.py

4. Reproducing the Real Data Application (Section 4)

To run the positive control (\texttt{Raf} \perp \texttt{Erk} \mid \texttt{Mek})
and primary causal hypothesis
(\texttt{Raf} \perp \texttt{Mek} \mid \texttt{PKC}) on the flow cytometry
dataset:

python src/real_data_analysis.py

Methodological Summary

| Method                    | Conditioning Approach                  | Time Complexity  | Memory Complexity | Calibration / Size Control                    |
| :------------------------ | :------------------------------------- | :--------------- | :---------------- | :-------------------------------------------- |
| **Permutation Test**      | OLS Residuals                          | $O(B \cdot n^2)$ | $O(n^2)$          | Non-asymptotic benchmark (accurate size)      |
| **Naive $\chi_1^2$ Test** | OLS Residuals + $\chi_1^2$             | $O(n^2)$         | $O(n^2)$          | Substantially inflated ($\approx 21\%$ error) |
| **Proposed Gamma Test**   | OLS Residuals + Moment Matching        | $O(n^2)$         | $O(n^2)$          | Closed-form calibrated ($\approx 5\%$ size)   |
| **Wild Bootstrap Test**   | OLS Residuals + Rademacher Multipliers | $O(B \cdot n^2)$ | $O(n^2)$          | Robust under conditional heteroskedasticity   |
