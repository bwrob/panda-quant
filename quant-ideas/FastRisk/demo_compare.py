import time
import numpy as np
from datetime import date
import os
import QuantLib as ql # For version printing

from scenario_generator import SimpleRandomScenarioGenerator
from demo_calibration import generate_instrument_definitions # Using this for consistent instrument set
from workflow_manager import (
    InstrumentProcessor,
    PortfolioBuilder,
    PortfolioAnalytics,
    generate_portfolio_specs_for_serialization
)
# product_definitions, custom_importer, pricers are used internally by workflow_manager

def run_comparison_demo(
    num_instruments_to_generate: int = 12,
    num_var_scenarios: int = 1000, # Scenarios for VaR calculation
    n_tff_domain_scenarios: int = 2000, # Scenario pool for TFF calibration
    n_tff_fitting_samples: int = 50, # Samples for TFF sample_and_fit
    num_tff_workers: int = None, # For parallel TFF calibration, None for os.cpu_count()
    tff_batch_size: int = 4, # Batch size for TFF calibration processing
    random_seed: int = 42
):
    """
    Compares Full Pricer and TFF Pricer paths for the same portfolio.
    Measures VaR and processing times for both approaches.
    """
    print(f"--- Comparison Demo: Full vs. TFF Pricing ---")
    print(f"QuantLib version: {ql.__version__}")
    print(f"Parameters: num_instruments={num_instruments_to_generate}, num_var_scenarios={num_var_scenarios}, "
          f"n_tff_domain_scenarios={n_tff_domain_scenarios}, n_tff_fitting_samples={n_tff_fitting_samples}, "
          f"num_tff_workers={num_tff_workers or os.cpu_count()}, tff_batch_size={tff_batch_size}, random_seed={random_seed}")

    # --- 1. Global Setup ---
    np.random.seed(random_seed)
    val_date = date(2025, 5, 18)
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"
    default_g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
    default_bs_rfr = 0.025
    default_bs_div = 0.01

    # Base maps for scenario generation (consistent with demo_calibration)
    base_rates_map = {f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
    all_underlying_symbols = list(set([f"STOCK_{i%10}" for i in range(num_instruments_to_generate)]))
    base_s0_map_gen = {}
    base_vol_map_gen = {}
    base_other_map_gen = {}
    for sym in all_underlying_symbols:
        base_s0_map_gen[f"{DEMO_CURRENCY}_{sym}_S0"] = 90 + np.random.rand() * 20
        base_vol_map_gen[f"{DEMO_CURRENCY}_{sym}_VOL"] = 0.20 + np.random.rand() * 0.1
        base_vol_map_gen[f"{DEMO_CURRENCY}_{sym}_EQVOL"] = 0.20 + np.random.rand() * 0.1 # For convertibles
        base_other_map_gen[f"{DEMO_CURRENCY}_{sym}_DIVYIELD"] = 0.01 + np.random.rand() * 0.01
        base_other_map_gen[f"{DEMO_CURRENCY}_{sym}_CS"] = 0.01 + np.random.rand() * 0.01
    merged_s0_map_gen = {**base_s0_map_gen, **base_other_map_gen}

    # Scenario generator for VaR (used by PortfolioAnalytics and for Full Path processing if needed)
    scen_gen_var = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=merged_s0_map_gen,
        base_vol_map=base_vol_map_gen, random_seed=random_seed
    )
    var_scenarios, var_factor_names = scen_gen_var.generate_scenarios(num_var_scenarios)
    print(f"Generated {num_var_scenarios} scenarios for VaR calculation.")

    # Scenario generator for TFF calibration domain
    scen_gen_tff_domain = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=merged_s0_map_gen,
        base_vol_map=base_vol_map_gen, random_seed=random_seed + 1 # Slightly different seed for diversity if desired
    )
    tff_domain_scenarios, tff_domain_factor_names = scen_gen_tff_domain.generate_scenarios(n_tff_domain_scenarios)
    print(f"Generated {n_tff_domain_scenarios} scenarios for TFF calibration domain.")

    # --- 2. Instrument Definitions ---
    instrument_definitions = generate_instrument_definitions(num_instruments_to_generate, val_date)
    print(f"Created {len(instrument_definitions)} common instrument definitions.")

    holdings_data = [
        {"client_id": "CompareClient", "instrument_id": d["instrument_id"], "num_holdings": 10000}
        for d in instrument_definitions
    ]

    # --- Path 1: Full Valuation ---
    print("\n--- Path 1: Full Valuation ---")
    time_full_path_start = time.time()

    full_instrument_definitions = [dict(d, pricing_preference="FULL") for d in instrument_definitions]
    
    iproc_full = InstrumentProcessor(
        scenario_generator=scen_gen_var, # Scenario gen for base values if iproc uses it
        global_valuation_date=val_date,
        default_numeric_rate_tenors=tenors,
        default_g2_params=default_g2_params,
        default_bs_risk_free_rate=default_bs_rfr,
        default_bs_dividend_yield=default_bs_div,
        parallel_workers_tff=False, # TFF calibration is OFF
        n_scenarios_for_tff_domain=0 # Ensure TFF calibration is skipped
    )
    
    print("Processing instruments for full pricers...")
    # Pass var_scenarios, though they aren't used for pricing by iproc.process_instruments itself
    # This step primarily populates the registry with full_pricer_instance
    full_model_registry = iproc_full.process_instruments(
        full_instrument_definitions, var_scenarios, var_factor_names, batch_size=tff_batch_size
    )
    
    full_portfolio_specs = generate_portfolio_specs_for_serialization(
        holdings_data, full_model_registry, full_instrument_definitions
    )
    
    builder_full = PortfolioBuilder(full_model_registry)
    portfolio_full_dict = builder_full.build_portfolios_from_specs(
        full_portfolio_specs, val_date, default_g2_params, default_bs_rfr, default_bs_div
    )
    
    analytics_full = PortfolioAnalytics(
        client_portfolios=portfolio_full_dict,
        global_market_scenarios=var_scenarios,
        global_factor_names=var_factor_names,
        numeric_rate_tenors=tenors,
        scenario_generator_for_base_values=scen_gen_var
    )
    print("Calculating VaR for FULL portfolio...")
    var_full_results = analytics_full.run_var_analysis(var_percentiles=[1.0, 5.0])
    time_full_path_end = time.time()
    total_time_full_path = time_full_path_end - time_full_path_start
    
    print(f"Full valuation VaR (CompareClient): {var_full_results.get('CompareClient', {})}")
    print(f"Total time for full valuation path: {total_time_full_path:.2f}s")
    if instrument_definitions:
        print(f"Average time per instrument (full path): {total_time_full_path / len(instrument_definitions):.4f}s")

    # --- Path 2: TFF Valuation ---
    print("\n--- Path 2: TFF Valuation ---")
    
    # TFF Calibration
    time_tff_calib_start = time.time()
    tff_instrument_definitions = [dict(d, pricing_preference="TFF") for d in instrument_definitions]

    iproc_tff = InstrumentProcessor(
        scenario_generator=scen_gen_tff_domain, # Used for TFF domain sampling
        global_valuation_date=val_date,
        default_numeric_rate_tenors=tenors,
        default_g2_params=default_g2_params,
        default_bs_risk_free_rate=default_bs_rfr,
        default_bs_dividend_yield=default_bs_div,
        parallel_workers_tff=num_tff_workers,
        n_scenarios_for_tff_domain=n_tff_fitting_samples
    )
    print("Calibrating TFF models...")
    tff_model_registry = iproc_tff.process_instruments(
        tff_instrument_definitions,
        tff_domain_scenarios, # Scenarios for TFF calibration domain
        tff_domain_factor_names,
        batch_size=tff_batch_size
    )
    time_tff_calib_end = time.time()
    total_time_tff_calib = time_tff_calib_end - time_tff_calib_start
    print(f"Total time for TFF calibration: {total_time_tff_calib:.2f}s")
    if instrument_definitions:
        print(f"Average TFF calibration time per instrument: {total_time_tff_calib / len(instrument_definitions):.4f}s")

    # TFF Pricing (Inference)
    time_tff_inference_start = time.time()
    tff_portfolio_specs = generate_portfolio_specs_for_serialization(
        holdings_data, tff_model_registry, tff_instrument_definitions
    )
    
    builder_tff = PortfolioBuilder(tff_model_registry)
    portfolio_tff_dict = builder_tff.build_portfolios_from_specs(
        tff_portfolio_specs, val_date, default_g2_params, default_bs_rfr, default_bs_div
    )
    
    analytics_tff = PortfolioAnalytics(
        client_portfolios=portfolio_tff_dict,
        global_market_scenarios=var_scenarios, # Use same VaR scenarios for comparison
        global_factor_names=var_factor_names,
        numeric_rate_tenors=tenors,
        scenario_generator_for_base_values=scen_gen_var
    )
    print("Calculating VaR for TFF portfolio...")
    var_tff_results = analytics_tff.run_var_analysis(var_percentiles=[1.0, 5.0])
    time_tff_inference_end = time.time()
    total_time_tff_inference = time_tff_inference_end - time_tff_calib_start # This should be end - start of inference
    total_time_tff_inference = time_tff_inference_end - time_tff_inference_start


    print(f"TFF valuation VaR (CompareClient): {var_tff_results.get('CompareClient', {})}")
    print(f"Total time for TFF inference (portfolio pricing): {total_time_tff_inference:.2f}s")
    
    total_time_tff_path = total_time_tff_calib + total_time_tff_inference
    print(f"Total time for TFF path (calibration + inference): {total_time_tff_path:.2f}s")
    if instrument_definitions:
         print(f"Average time per instrument (TFF path, incl. calib): {total_time_tff_path / len(instrument_definitions):.4f}s")


    # --- 5. Comparison Summary ---
    print("\n--- Comparison Summary ---")
    print(f"{'Metric':<35} | {'Full Pricer Path':<25} | {'TFF Path':<25}")
    print("-" * 90)
    print(f"{'Total Time (s)':<35} | {total_time_full_path:<25.2f} | {total_time_tff_path:<25.2f}")
    if instrument_definitions:
        avg_full = total_time_full_path / len(instrument_definitions)
        avg_tff_total = total_time_tff_path / len(instrument_definitions)
        print(f"{'Avg Time per Instrument (s)':<35} | {avg_full:<25.4f} | {avg_tff_total:<25.4f}")
    print(f"{'  TFF Calibration Time (s)':<35} | {'N/A':<25} | {total_time_tff_calib:<25.2f}")
    print(f"{'  TFF Inference Time (s)':<35} | {'N/A':<25} | {total_time_tff_inference:<25.2f}")
    
    var_full_client = var_full_results.get('CompareClient', {})
    var_tff_client = var_tff_results.get('CompareClient', {})
    
    # Get the base value for comparison
    base_value_full = var_full_client.get('base_value', 0.0)
    base_value_tff = var_tff_client.get('base_value', 0.0)
    
    print(f"{'Base Value':<35} | {base_value_full:<25.2f} | {base_value_tff:<25.2f}")
    
    # Get the nested 'var_values' dictionaries
    var_full_values = var_full_client.get('var_values', {})
    var_tff_values = var_tff_client.get('var_values', {})
    
    for p_level in [1.0, 5.0]: # These are the percentile levels for loss (e.g., 1.0 means 99% VaR)
        actual_key_in_results = f"var_{100-p_level:.0f}pct"
        
        val_f = var_full_values.get(actual_key_in_results, "N/A")
        val_t = var_tff_values.get(actual_key_in_results, "N/A")
        
        val_f_str = f"{val_f:.2f}" if isinstance(val_f, (int, float)) else str(val_f)
        val_t_str = f"{val_t:.2f}" if isinstance(val_t, (int, float)) else str(val_t)
        print(f"{f'VaR {100-p_level:.0f}% (Loss at {p_level:.0f}p)':<35} | {val_f_str:<25} | {val_t_str:<25}")

    # --- Per-Instrument TFF Fitting Details ---
    print("\n--- Per-Instrument TFF Fitting Results (Sample) ---")
    header = f"{'Instrument ID':<40} | {'Fit Time (s)':<12} | {'RMSE':<10} | {'TFF Factors':<60} | {'Fixed Params':<50}"
    print(header)
    print("-" * len(header))
    
    if 'tff_model_registry' in locals() and tff_model_registry:
        # Print details for a sample of instruments, e.g., first 5 or specific types
        count = 0
        max_to_print = 10 # Limit the number of printed entries for brevity
        for instrument_id, model_info in tff_model_registry.items():
            if count >= max_to_print:
                print("... (output truncated for brevity)")
                break

            engine_type = model_info.get('pricing_method')
            if engine_type == 'TFF':
                fit_time = model_info.get('tff_fit_time_seconds', 'N/A')
                rmse = model_info.get('tff_rmse', 'N/A')
                raw_factors = model_info.get('tff_raw_input_names', [])
                fixed_params = model_info.get('tff_fixed_pricer_params', {})

                fit_time_str = f"{fit_time:.4f}" if isinstance(fit_time, (int, float)) else str(fit_time)
                rmse_str = f"{rmse:.4f}" if isinstance(rmse, (int, float)) else str(rmse)
                
                # Truncate long lists/dicts for display
                raw_factors_str = str(raw_factors)
                if len(raw_factors_str) > 58:
                    raw_factors_str = raw_factors_str[:55] + "..."
                
                fixed_params_str = str(fixed_params)
                if len(fixed_params_str) > 48:
                    fixed_params_str = fixed_params_str[:45] + "..."

                print(f"{instrument_id:<40} | {fit_time_str:<12} | {rmse_str:<10} | {raw_factors_str:<60} | {fixed_params_str:<50}")
                count += 1
            elif model_info.get('error_tff_calibration'):
                error_msg = str(model_info.get('error_tff_calibration'))
                if len(error_msg) > 100: error_msg = error_msg[:97] + "..."
                print(f"{instrument_id:<40} | {'CALIB ERROR':<12} | {'N/A':<10} | {error_msg:<113}")
                count +=1

    else:
        print("TFF Model Registry not found or is empty.")

    print("\n--- End of Comparison Demo ---")

if __name__ == "__main__":
    run_comparison_demo(
        num_instruments_to_generate=3, # Keep low for quick demo runs
        num_var_scenarios=2000,
        n_tff_domain_scenarios=100,
        n_tff_fitting_samples=50,
        num_tff_workers=os.cpu_count(), # Use available cores for TFF
        tff_batch_size=4,
        random_seed=42
    )