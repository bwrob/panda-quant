import numpy as np
from scipy.stats import norm
import time

def calculate_confidence_from_k(k, q, lambda_multiplier):
    """
    Calculates the theoretical confidence level for a fixed re-evaluation percentage 'k'.
    This is the inverse of the original problem.

    Args:
        k (float): The fixed percentage of top paths to re-evaluate (e.g., 0.02).
        q (float): The target quantile (e.g., 0.99).
        lambda_multiplier (float): The proportional error multiplier (e.g., 0.03).

    Returns:
        float: The theoretical confidence level.
    """
    z_q = norm.ppf(q)
    
    # We can find the z_cutoff directly from k
    z_cutoff = norm.ppf(1 - k)

    # Rearrange the core formula to solve for z_conf
    numerator = np.sqrt(1 + (lambda_multiplier * z_q)**2) - (z_cutoff / z_q)
    denominator = lambda_multiplier
    z_conf = numerator / denominator
    
    theoretical_confidence = norm.cdf(z_conf)
    
    return theoretical_confidence

def run_simulation_vectorized(N, num_trials, q, fixed_k, lambda_multiplier):
    """
    Runs a fully vectorized Monte Carlo simulation to test a fixed k.

    Args:
        N (int): The number of paths in each simulation trial.
        num_trials (int): The number of simulation trials to run.
        q (float): The target quantile.
        fixed_k (float): The fixed percentage of paths to re-evaluate.
        lambda_multiplier (float): The proportional error multiplier.

    Returns:
        float: The empirical success rate from the simulation.
    """
    print(f"Running {num_trials:,} vectorized trials with a fixed k of {fixed_k:.4%}...")
    
    num_to_reval = int(N * fixed_k)

    # 1. Generate all random numbers for all trials in one go. Shape: (num_trials, N)
    true_values = np.random.randn(num_trials, N)
    
    # 2. Generate corresponding proportional errors for all trials
    error_std_devs = lambda_multiplier * np.abs(true_values)
    errors = np.random.normal(0, error_std_devs)
    
    # 3. Create proxy values for all trials
    proxy_values = true_values + errors

    # 4. Find the true quantile for each trial (row). Result shape: (num_trials, 1)
    true_quantile_values = np.quantile(true_values, q, axis=1, keepdims=True)

    # 5. Get indices that would sort each row of proxy values from high to low.
    proxy_sorted_indices = np.argsort(proxy_values, axis=1)[:, ::-1]

    # 6. Identify the rejected paths for each trial. Shape: (num_trials, N - num_to_reval)
    rejected_indices = proxy_sorted_indices[:, num_to_reval:]

    # 7. Use the rejected indices to gather the true values of those paths.
    true_values_of_rejected_paths = np.take_along_axis(true_values, rejected_indices, axis=1)
    
    # 8. A failure occurs if ANY rejected path's true value is > the trial's true quantile.
    # The result is a boolean array of shape (num_trials,).
    failed_trials = np.any(true_values_of_rejected_paths > true_quantile_values, axis=1)

    # 9. Calculate the success rate by counting how many trials did NOT fail.
    success_count = num_trials - np.sum(failed_trials)
    
    return success_count / num_trials

def main():
    """ Main function to run the analysis. """
    # --- Parameters from our discussion ---
    TARGET_QUANTILE = 0.99
    PROPORTIONAL_ERROR = 0.03  # 3%
    FIXED_K = 0.02             # We are now fixing k to 2%

    # --- Simulation Parameters ---
    NUM_PATHS = 5000
    NUM_TRIALS = 20000

    # --- Step 1: Calculate the theoretical confidence for our fixed k ---
    theoretical_conf = calculate_confidence_from_k(FIXED_K, TARGET_QUANTILE, PROPORTIONAL_ERROR)
    
    print("--- Theoretical Calculation (Inverse) ---")
    print(f"Target Quantile (q): {TARGET_QUANTILE}")
    print(f"Proportional Error (λ): {PROPORTIONAL_ERROR}")
    print(f"Fixed Re-evaluation Percentage (k): {FIXED_K:.4%}")
    print("-" * 30)
    print(f"Implied Theoretical Confidence: {theoretical_conf:.6%}")
    print("-" * 30)
    print("\n--- Empirical Simulation (Vectorized) ---")

    # --- Step 2: Run vectorized simulation and time it ---
    start_time = time.time()
    empirical_success_rate = run_simulation_vectorized(
        N=NUM_PATHS,
        num_trials=NUM_TRIALS,
        q=TARGET_QUANTILE,
        fixed_k=FIXED_K,
        lambda_multiplier=PROPORTIONAL_ERROR
    )
    end_time = time.time()
    vectorized_time = end_time - start_time
    
    print("\n--- Results ---")
    print(f"Time for vectorized simulation: {vectorized_time:.2f} seconds")
    print(f"Implied Theoretical Confidence:                  {theoretical_conf:.6%}")
    print(f"Empirical Success Rate from {NUM_TRIALS:,} trials: {empirical_success_rate:.6%}")
    print("-" * 30)
    if abs(empirical_success_rate - theoretical_conf) < 0.01:
        print("Conclusion: The empirical result closely matches the theoretical formula. Success!")
    else:
        print("Conclusion: The empirical result differs from the theoretical target.")

if __name__ == "__main__":
    main()
