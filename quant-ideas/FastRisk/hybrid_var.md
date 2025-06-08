## 1. Executive Summary

Value-at-Risk (VaR) is a cornerstone of financial risk management, but its calculation can be computationally intensive. While proxy models (e.g., machine-learning models) offer significant speedups, they sacrifice accuracy. A hybrid approach—using a fast proxy to identify a small subset of “critical” scenarios for full re-evaluation—presents an optimal solution.

This document outlines a robust methodology to answer the central question in any hybrid model:

> **What is the minimum percentage of scenarios (k) that must be fully re-evaluated to ensure, with a specified level of confidence, that the hybrid VaR equals the VaR from a full re-evaluation?**

We have developed a novel analytical formula that accounts for realistic, proportional proxy errors. This formula allows risk managers to move from a heuristic choice of k to a principled, risk-quantified decision. The model’s validity has been confirmed through large-scale Monte Carlo simulations and its successful application to a real-world QuantLib-based VaR engine, demonstrating a near-perfect match between theory and practice.

---

## 2. Problem Statement & The Hybrid Approach

The computational burden of full revaluation for large portfolios under thousands of market scenarios is a significant challenge. The primary approaches to VaR calculation are:

1. **Full Revaluation**  
   - **Pros:** Highest accuracy.  
   - **Cons:** Extremely slow (prices every instrument under every scenario).

2. **Proxy-Only Model**  
   - **Pros:** Very fast.  
   - **Cons:** Inaccurate; calculated VaR may differ from the true VaR.

3. **Hybrid Model** (two-step process)  
   1. **Proxy Pass:**  
      - Run the fast proxy on all _N_ scenarios → approximate P&L for each.  
   2. **Targeted Revaluation:**  
      - Identify the top _k_% of most extreme losses (by proxy).  
      - Fully re-evaluate only those _k·N_ scenarios.  
      - Compute VaR from this “hybrid” P&L vector.

> **Key challenge:** Choosing _k_. Too small → risk miscalculating VaR. Too large → lose computational advantage.  
> **Solution:** A formal methodology to choose _k_ for a given confidence level.

---

## 3. Analytical Model Development

### 3.1 Key Assumptions

- **Normality**: True portfolio values \(Y\) across scenarios are normal. Proxy errors are also normal.  
- **Heteroscedastic (Proportional) Error**: Proxy error standard deviation \(\sigma_e\) scales with the magnitude of \(Y\).  
  \[
    \sigma_e(Y) = \lambda \, Y
  \]
  - Example: \(\lambda = 0.03\) ⇒ typical error ≈ 3% of true value.

### 3.2 Model Derivation

1. **Cutoff Definition**  
   Let \(x_c\) be the proxy-loss cutoff. We’ll re-evaluate all scenarios with proxy loss \(\ge x_c\).

2. **Critical Scenario (“Weakest Link”)**  
   The scenario exactly at the cutoff (\(X = x_c\)) is the riskiest among the excluded set. Ensuring it cannot exceed the true VaR threshold guarantees safety for all excluded scenarios.

3. **Conditional Distribution**  
   We need the distribution of \(Y\) given \(X = x_c\):
   \[
   P(Y \mid X = x_c)
   \]
   Proxy error pulls the expected true value toward the mean, with uncertainty \(\sigma_{e}\).

4. **Confidence Bound**  
   For confidence level \(\mathrm{conf}\), require:
   \[
     Y_q = \mathbb{E}[\,Y \mid X = x_c\,] \;+\; z_{\mathrm{conf}}\;\sqrt{\mathrm{Var}[\,Y \mid X = x_c\,]}
   \]
   where \(Y_q\) is the true VaR loss (the \(q\)-th percentile), and \(z_{\mathrm{conf}} = \Phi^{-1}(\mathrm{conf})\).

5. **Proportional-Error Approximation**  
   Around the VaR quantile, \(\sigma_{e,q} = \lambda\,Y_q\).

6. **Solve for \(x_c\)**  
   Substituting the conditional-mean and variance formulas and \(\sigma_{e,q}\), we derive:
   \[
     z_{\text{cutoff}}
     = \frac{x_c - \mu}{\sigma}
     = z_q\Bigl(1 + (\lambda\,z_q)^2\Bigr) - \lambda\,z_{\mathrm{conf}}
   \]
   where
   \[
     z_q = \Phi^{-1}(q),\quad
     z_{\mathrm{conf}} = \Phi^{-1}(\mathrm{conf}).
   \]

7. **Compute \(k\)**  
   The fraction to re-evaluate is the upper tail beyond \(z_{\text{cutoff}}\):
   \[
     k \;=\; 1 - \Phi\bigl(z_{\text{cutoff}}\bigr).
   \]

This provides a direct link between risk tolerance (\(\mathrm{conf}\)), proxy quality (\(\lambda\)), and the operational parameter (\(k\)).

### 3.3 Hybrid Model Mechanics in Practice

1. **Initial Ranking**  
   - Run proxy on all \(N\) scenarios → approximate losses.  
   - Sort by loss, preserve original indices (argsort).

2. **Identify Critical Set**  
   - Select top \(k \times N\) indices.

3. **Targeted Revaluation**  
   - Re-price only the selected scenarios with the full engine.

4. **Compute Final VaR**  
   - Take the \(q\)-th percentile from the accurately re-priced set of size \(kN\).

---

## 4. Model Validation

### 4.1 Monte Carlo Simulation

- **Setup**: \(q=0.99\), \(\lambda=0.03\), \(k=0.02\).  
- **Procedure**:  
  1. Generate a matrix (20 000 trials × 5 000 paths) of true values \(Y \sim N(0,1)\).  
  2. Generate proportional errors \(\varepsilon \sim N(0, (\lambda Y)^2)\).  
  3. Form proxies \(X = Y + \varepsilon\).  
  4. For each trial, identify top 2% by \(X\).  
  5. Check if any excluded path’s true loss exceeds the true 99th percentile.  
- **Result**: Empirical success rate matched the theoretical confidence (~99.997%) within a few hundredths of a percent.

### 4.2 Empirical Application with QuantLib Engine

- **Test**: 100 independent portfolios, 1% VaR target.  
- **Proxy quality**: \(\lambda < 1\%\) (from TFF fitting).  
- **Hybrid \(k\)**: 2%.  
- **Outcome**:  
  - **Hybrid VaR vs. Full VaR**: Perfect match in 100% of trials (within 0.01 absolute tolerance).  
  - **TFF-only Proxy Error**: Varied between –0.06% and +0.07% of base, average 0.01%.  
  - **Hybrid Error**: Always 0.00%.

| Metric                                    | Min   | Max   | Avg    |
|-------------------------------------------|-------|-------|--------|
| Full VaR – TFF-only Proxy VaR ( % of base) | –0.06 | +0.07 | +0.01 |
| Full VaR – Hybrid VaR ( % of base)         |  0.00 |  0.00 |  0.00 |

---

## 5. Conclusion

This methodology provides a scientifically rigorous framework for optimizing hybrid VaR calculations. By deriving and validating an analytical formula, we replace heuristic choices of \(k\) with a quantifiable, risk-based approach. Practitioners can now:

- **Set explicit confidence levels** (\(\mathrm{conf}\)).  
- **Quantify proxy model quality** (\(\lambda\)).  
- **Compute the minimal re-evaluation fraction** (\(k\)).  

Validated by simulation and real-world implementation, this approach delivers significant computational savings without compromising accuracy.