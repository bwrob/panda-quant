import time
import numpy as np
from datetime import date
import os
import QuantLib as ql


from scenario_generator import SimpleRandomScenarioGenerator
from demo_calibration import generate_instrument_definitions
from workflow_manager import (
    InstrumentProcessor,
    PortfolioBuilder,
    generate_portfolio_specs_for_serialization,
    generate_price_strips
)

def run_hybrid_var_demo(
    num_instruments_to_generate: int = 12,
    num_var_scenarios: int = 1000,
    n_tff_domain_scenarios: int = 2000,
    n_tff_fitting_samples: int = 50,
    hybrid_critical_percentile: float = 0.2,
    num_tff_workers: int = None,
    tff_batch_size: int = 4,
    random_seed: int = 42,
    # NEW: Individual method flags
    run_full_reval: bool = True,
    run_tff_only: bool = True,
    run_hybrid: bool = True
):
    """
    Compares Full, TFF-Only, and Hybrid (TFF + targeted Full Reval) VaR.
    Each method can be enabled/disabled independently.
    """
    print(f"--- Hybrid VaR Comparison Demo ---")
    print(f"QuantLib version: {ql.__version__}")
    print(f"Parameters: num_instruments={num_instruments_to_generate}, num_var_scenarios={num_var_scenarios}, "
          f"n_tff_domain_scenarios={n_tff_domain_scenarios}, n_tff_fitting_samples={n_tff_fitting_samples}, "
          f"hybrid_critical_percentile={hybrid_critical_percentile}%, "
          f"num_tff_workers={num_tff_workers or os.cpu_count()}, tff_batch_size={tff_batch_size}, "
          f"random_seed={random_seed}")
    print(f"Methods enabled: Full={run_full_reval}, TFF={run_tff_only}, Hybrid={run_hybrid}")

    # Validation: At least one method must be enabled
    if not any([run_full_reval, run_tff_only, run_hybrid]):
        raise ValueError("At least one method must be enabled (run_full_reval, run_tff_only, or run_hybrid)")

    # --- 1. Global Setup ---
    np.random.seed(random_seed)
    val_date = date(2025, 5, 18)
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float)
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"
    default_g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
    default_bs_rfr = 0.025
    default_bs_div = 0.01

    base_rates_map = {f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t * 0.001 for t in tenors}
    all_underlying_symbols = list(set([f"STOCK_{i%10}" for i in range(num_instruments_to_generate)]))
    base_s0_map_gen, base_vol_map_gen, base_other_map_gen = {}, {}, {}
    for sym in all_underlying_symbols:
        base_s0_map_gen[f"{DEMO_CURRENCY}_{sym}_S0"] = round(90 + np.random.rand() * 20)
        base_vol_map_gen[f"{DEMO_CURRENCY}_{sym}_VOL"] = round(0.20 + np.random.rand() * 0.1, 2)
        base_vol_map_gen[f"{DEMO_CURRENCY}_{sym}_EQVOL"] = round(0.20 + np.random.rand() * 0.1, 2)
        base_other_map_gen[f"{DEMO_CURRENCY}_{sym}_DIVYIELD"] = round(0.01 + np.random.rand() * 0.01, 2)
        base_other_map_gen[f"{DEMO_CURRENCY}_{sym}_CS"] = round(0.01 + np.random.rand() * 0.01, 2)
    merged_s0_map_gen = {**base_s0_map_gen, **base_other_map_gen}

    scen_gen_var = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map, base_s0_map=merged_s0_map_gen,
        base_vol_map=base_vol_map_gen, random_seed=random_seed
    )
    var_scenarios, var_factor_names = scen_gen_var.generate_scenarios(num_var_scenarios)

    # Only generate TFF domain scenarios if TFF or Hybrid methods are enabled
    if run_tff_only or run_hybrid:
        scen_gen_tff_domain = SimpleRandomScenarioGenerator(
            base_rates_map=base_rates_map, base_s0_map=merged_s0_map_gen,
            base_vol_map=base_vol_map_gen, random_seed=random_seed + 1
        )
        tff_domain_scenarios, tff_domain_factor_names = scen_gen_tff_domain.generate_scenarios(n_tff_domain_scenarios)

    instrument_definitions = generate_instrument_definitions(num_instruments_to_generate, val_date)
    holdings_data = [{"client_id": "HybridClient", "instrument_id": d["instrument_id"], "num_holdings": 1000}
                     for d in instrument_definitions]
    

    # Initialize result variables
    var_1pct_full = None
    base_value_full = None
    total_time_full_path = None
    var_1pct_tff = None
    base_value_tff = None
    total_time_tff_path = None
    total_time_tff_calib = None
    total_time_tff_inference = None
    var_1pct_hybrid = None
    total_time_hybrid_path = None
    num_critical_scenarios = 0

    # Variables needed for cross-method dependencies
    portfolio_obj_full = None
    full_instrument_defs = None
    iproc_full = None
    losses_tff = None
    N = num_var_scenarios

    # --- Path 1: Full Revaluation (Baseline) ---
    if run_full_reval:
        print("\n--- Path 1: Full Revaluation ---")
        time_full_path_start = time.time()
        full_instrument_defs = [dict(d, pricing_preference="FULL") for d in instrument_definitions]
        # For convertibles, populate the correct vol, dividend yield, and credit spread from base maps
        for d in full_instrument_defs:
            if d["product_type"] == "ConvertibleBond":
                # These should be in pricer_params, not params
                if "pricer_params" not in d:
                    d["pricer_params"] = {}
                
                d["pricer_params"]["equity_volatility"] = base_vol_map_gen[f"{DEMO_CURRENCY}_{d['params']['underlying_symbol']}_EQVOL"]
                d["pricer_params"]["dividend_yield"] = base_other_map_gen[f"{DEMO_CURRENCY}_{d['params']['underlying_symbol']}_DIVYIELD"]
                d["pricer_params"]["credit_spread"] = base_other_map_gen[f"{DEMO_CURRENCY}_{d['params']['underlying_symbol']}_CS"]
                
                # Also ensure S0 is set correctly for convertibles
                d["pricer_params"]["s0_val"] = base_s0_map_gen[f"{DEMO_CURRENCY}_{d['params']['underlying_symbol']}_S0"]
        
        iproc_full = InstrumentProcessor(
            scen_gen_var, val_date, tenors, default_g2_params, default_bs_rfr, default_bs_div, False, 0
        )
        full_model_registry = iproc_full.process_instruments(full_instrument_defs, var_scenarios, var_factor_names, tff_batch_size)
        full_portfolio_specs = generate_portfolio_specs_for_serialization(holdings_data, full_model_registry, full_instrument_defs)
        builder_full = PortfolioBuilder(full_model_registry)
        portfolio_full_dict = builder_full.build_portfolios_from_specs(full_portfolio_specs, val_date, default_g2_params, default_bs_rfr, default_bs_div)
        
        portfolio_obj_full = portfolio_full_dict["HybridClient"]
        
        batch_size = max(1, num_instruments_to_generate // 100)  # Ensure reasonable batch size
        strips = generate_price_strips(
            instrument_specs=full_instrument_defs,
            global_market_scenarios=var_scenarios,
            global_factor_names=var_factor_names,
            iproc=iproc_full,
            num_workers=os.cpu_count(),
            batch_size=batch_size
        )
        
        portfolio_values_full_strips = portfolio_obj_full.price_portfolio_from_strips(strips)
        base_value_full = portfolio_values_full_strips[0]

        losses_full = base_value_full - portfolio_values_full_strips
        sorted_losses_full = np.sort(losses_full)
        N = len(sorted_losses_full)
        idx1 = max(0, int(np.ceil(0.01 * N)) - 1)
        var_1pct_full = -sorted_losses_full[idx1]

        time_full_path_end = time.time()
        total_time_full_path = time_full_path_end - time_full_path_start
        print(f"Full 1% VaR: {var_1pct_full:,.2f}. Base Value: {base_value_full:,.2f}")
        print(f"Time for Full Path: {total_time_full_path:.2f}s")
    else:
        print("\n--- Path 1: Full Revaluation SKIPPED ---")

    # --- Path 2: TFF-Only Revaluation ---
    if run_tff_only:
        print("\n--- Path 2: TFF-Only Revaluation ---")
        time_tff_calib_start = time.time()

        tff_instrument_defs = [dict(d, pricing_preference="TFF") for d in instrument_definitions]
        
        iproc_tff = InstrumentProcessor(
            scen_gen_tff_domain, val_date, tenors, default_g2_params, default_bs_rfr, default_bs_div,
            num_tff_workers, n_tff_fitting_samples
        )
        tff_model_registry = iproc_tff.process_instruments(tff_instrument_defs, tff_domain_scenarios, tff_domain_factor_names, tff_batch_size)
        time_tff_calib_end = time.time()
        total_time_tff_calib = time_tff_calib_end - time_tff_calib_start
        
        time_tff_inference_start = time.time()
        tff_portfolio_specs = generate_portfolio_specs_for_serialization(
            holdings_data, tff_model_registry, tff_instrument_defs
        )
        
        builder_tff = PortfolioBuilder(tff_model_registry)
        portfolio_tff_dict = builder_tff.build_portfolios_from_specs(
            tff_portfolio_specs, val_date, default_g2_params, default_bs_rfr, default_bs_div
        )

        portfolio_obj_tff = portfolio_tff_dict["HybridClient"]
        portfolio_values_tff = portfolio_obj_tff.price_portfolio(var_scenarios, var_factor_names, tenors)
        base_value_tff = portfolio_values_tff[0]
        
        losses_tff = base_value_tff - portfolio_values_tff
        sorted_losses_tff = np.sort(losses_tff)
        N = len(sorted_losses_tff)
        
        idx1_tff = max(0, int(np.ceil(0.01 * N)) - 1)
        var_1pct_tff = -sorted_losses_tff[idx1_tff]
      
        time_tff_inference_end = time.time()
        total_time_tff_inference = time_tff_inference_end - time_tff_inference_start
        total_time_tff_path = total_time_tff_calib + total_time_tff_inference
        print(f"TFF 1% VaR: {var_1pct_tff:,.2f}. Base Value: {base_value_tff:,.2f}")
        print(f"Time for TFF Path: {total_time_tff_path:.2f}s (Calib: {total_time_tff_calib:.2f}s, Inference: {total_time_tff_inference:.2f}s)")
    else:
        print("\n--- Path 2: TFF-Only Revaluation SKIPPED ---")

    # --- Path 3: Hybrid TFF + Full Revaluation ---
    if run_hybrid:
        if not run_tff_only:
            raise ValueError("Hybrid method requires TFF method to be enabled (run_tff_only=True)")
        if not run_full_reval:
            print("\n--- Path 3: Hybrid (TFF + Targeted Full Reval) ---")
            print("Warning: Full revaluation not run. Setting up minimal infrastructure for hybrid...")
            # Set up minimal full infrastructure for hybrid
            full_instrument_defs = [dict(d, pricing_preference="FULL") for d in instrument_definitions]
            iproc_full = InstrumentProcessor(
                scen_gen_var, val_date, tenors, default_g2_params, default_bs_rfr, default_bs_div, False, 0
            )
            
            # NEW: Calculate full base value for proper hybrid comparison
            print("Calculating full base value for hybrid comparison...")
            base_scenario = var_scenarios[0:1]  # First scenario is base scenario
            batch_size = max(1, num_instruments_to_generate // 100)  # Ensure reasonable batch size
            base_strips_full = generate_price_strips(
                instrument_specs=full_instrument_defs,
                global_market_scenarios=base_scenario,
                global_factor_names=var_factor_names,
                iproc=iproc_full,
                num_workers=os.cpu_count(),
                batch_size=batch_size
            )
            base_value_full = sum(
                holding["num_holdings"] * base_strips_full[holding["instrument_id"]][0]
                for holding in holdings_data
            )
            print(f"Full base value for hybrid: {base_value_full:,.2f} (vs TFF base: {base_value_tff:,.2f})")
        else:
            print("\n--- Path 3: Hybrid (TFF + Targeted Full Reval) ---")
        
        time_hybrid_full_reval_start = time.time()
        sorted_losses_tff = np.sort(losses_tff)
        sorted_indices_tff = np.argsort(losses_tff)
        
        idx_threshold = max(0, int(np.ceil(hybrid_critical_percentile * N)) - 1)
        
        critical_idx = sorted_indices_tff[:idx_threshold]
        critical_scenarios_data = var_scenarios[critical_idx]
        
        batch_size = max(1, num_instruments_to_generate // 100)  # Ensure reasonable batch size
        portfolio_values_full_critical = generate_price_strips(
            instrument_specs=full_instrument_defs,
            global_market_scenarios=critical_scenarios_data,
            global_factor_names=var_factor_names,
            iproc=iproc_full,
            num_workers=os.cpu_count(),
            batch_size=batch_size
        )

        portfolio_values_full_critical = np.array([
            sum(holding["num_holdings"] * portfolio_values_full_critical[holding["instrument_id"]][i]
                for holding in holdings_data)
            for i in range(len(critical_scenarios_data))
        ])
        
        # Use consistent base value for loss calculation
        losses_full_critical = base_value_full - portfolio_values_full_critical

        sorted_losses_full_critical = np.sort(losses_full_critical)
        num_critical_scenarios = len(sorted_losses_full_critical)
        
        print(f"Number of critical scenarios identified: {num_critical_scenarios} / {num_var_scenarios}")
        if num_critical_scenarios == 0:
            print("No critical scenarios identified. Skipping Hybrid VaR calculation.")
            var_1pct_hybrid = None
        else:
            # NEW: Adjusted VaR calculation for hybrid method
            # Since we only have critical scenarios, we need to adjust the percentile calculation
            if run_full_reval:
                # Standard calculation when we have full comparison
                m_1 = int(np.ceil(0.01 * N)) - 1
                var_1pct_hybrid = -sorted_losses_full_critical[m_1]
            else:
                # Adjusted calculation when using TFF+Hybrid only
                # Use the worst loss from critical scenarios as approximation
                critical_percentile_adj = 0.01 / hybrid_critical_percentile  # Adjust for subset
                m_1_adj = max(0, min(num_critical_scenarios - 1, int(np.ceil(critical_percentile_adj * num_critical_scenarios)) - 1))
                var_1pct_hybrid = -sorted_losses_full_critical[m_1_adj]
                print(f"Adjusted hybrid VaR calculation: using index {m_1_adj} of {num_critical_scenarios} critical scenarios")
        
        time_hybrid_full_reval_end = time.time()
        time_hybrid_full_reval_critical = time_hybrid_full_reval_end - time_hybrid_full_reval_start    

        total_time_hybrid_path = (
            total_time_tff_calib + total_time_tff_inference + time_hybrid_full_reval_critical
        )
        print(f"Hybrid 1% VaR: {var_1pct_hybrid:,.2f}. Base Value (Full Reval): {base_value_full:,.2f}")
        print(f"Time for Hybrid Path: {total_time_hybrid_path:.2f}s "
              f"(TFF Calib: {total_time_tff_calib:.2f}s, TFF Full Pass: {total_time_tff_inference:.2f}s, Full Reval Critical: {time_hybrid_full_reval_critical:.2f}s)")
    else:
        print("\n--- Path 3: Hybrid (TFF + Targeted Full Reval) SKIPPED ---")

    # --- Comparison Summary ---
    print("\n--- VaR Comparison Summary ---")
    print(f"{'Metric':<30} | {'Full Path':<20} | {'TFF-Only Path':<20} | {'Hybrid Path':<20}")
    print("-" * 95)
    
    # Helper function to format values
    def format_value(value, is_time=False, is_currency=False):
        if value is None:
            return "SKIPPED"
        if is_currency:
            return f"{value:,.2f}"  # Add thousands separators for currency
        elif is_time:
            return f"{value:.2f}"
        else:
            return f"{value:,.2f}"  # Add thousands separators for other numeric values
    
    print(f"{'Total Time (s)':<30} | {format_value(total_time_full_path, True):<20} | {format_value(total_time_tff_path, True):<20} | {format_value(total_time_hybrid_path, True):<20}")
    
    if run_tff_only:
        tff_calib_str = format_value(total_time_tff_calib, True)
        hybrid_calib_str = format_value(total_time_tff_calib, True) if run_hybrid else "SKIPPED"
    else:
        tff_calib_str = "SKIPPED"
        hybrid_calib_str = "SKIPPED"
    
    print(f"{'  TFF Calib Time (s)':<30} | {'N/A':<20} | {tff_calib_str:<20} | {hybrid_calib_str:<20}")
    
    if run_tff_only and run_hybrid:
        hybrid_inference_str = format_value(total_time_tff_inference + time_hybrid_full_reval_critical, True)
    elif run_hybrid:
        hybrid_inference_str = "PARTIAL"
    else:
        hybrid_inference_str = "SKIPPED"
    
    print(f"{'  Pricing/Inference Time (s)':<30} | {format_value(total_time_full_path, True):<20} | {format_value(total_time_tff_inference, True):<20} | {hybrid_inference_str:<20}")
    print(f"{'Base Value':<30} | {format_value(base_value_full, is_currency=True):<20} | {format_value(base_value_tff, is_currency=True):<20} | {format_value(base_value_full, is_currency=True):<20}")
    print(f"{'VaR 1% Loss':<30} | {format_value(var_1pct_full, is_currency=True):<20} | {format_value(var_1pct_tff, is_currency=True):<20} | {format_value(var_1pct_hybrid, is_currency=True):<20}")
    
    # Calculate speedup gains where possible
    if run_full_reval and run_tff_only:
        tff_speedup = total_time_full_path / total_time_tff_path
    else:
        tff_speedup = None
    
    if run_full_reval and run_hybrid:
        hybrid_speedup = total_time_full_path / total_time_hybrid_path
    else:
        hybrid_speedup = None
    
    print(f"{'Speedup Gain x':<30} | {'N/A':<20} | {format_value(tff_speedup):<20} | {format_value(hybrid_speedup):<20}")
    
    if run_hybrid:
        print(f"Number of Critical Scenarios Revalued: {num_critical_scenarios:,} / {num_var_scenarios:,}")
    
    
    # Display table of difference between TFF and Full pricers from model registry
    if run_full_reval and 'full_model_registry' in locals():
        print("\n--- TFF vs Full Pricer Comparison ---")
        header = f"{'Instrument ID':<25} | {'Product Type':<15} | {'TFF Fit Time':<12} | {'TFF RMSE':<10} | {'TFF Base':<15} | {'Full Base':<15} | {'Base Diff %':<12} | {'Status':<10}"
        print(header)
        print("-" * len(header))
        
        count = 0
        max_to_display = 15  # Limit display for readability
        
        for instrument_id in sorted(tff_model_registry.keys()):
            if count >= max_to_display:
                print("... (showing first 15 instruments, use full registry for complete analysis)")
                break
            
            tff_model = tff_model_registry.get(instrument_id, {})
            full_model = full_model_registry.get(instrument_id, {})
            
            # Get basic info
            product_type = next((d['product_type'] for d in instrument_definitions if d['instrument_id'] == instrument_id), 'Unknown')
            
            # TFF metrics
            tff_fit_time = tff_model.get('tff_fit_time_seconds', 'N/A')
            tff_rmse = tff_model.get('tff_rmse', 'N/A')
            tff_error = tff_model.get('error_tff_calibration')
            
            # Get base scenario values for comparison
            base_scenario = var_scenarios[0:1]  # First scenario
            tff_base_value = 'N/A'
            full_base_value = 'N/A'
            base_diff_pct = 'N/A'
            status = 'OK'
            
            try:
                # Get TFF base value
                tff_instance = tff_model.get('tff_instance')
                if tff_instance and hasattr(tff_instance, 'price_scenarios'):
                    tff_price = tff_instance.price_scenarios(
                        raw_market_scenarios=base_scenario,
                        scenario_factor_names=var_factor_names,
                        rate_pillars=tenors
                    )[0]
                    tff_base_value = tff_price
                
                # Get Full base value
                full_instance = full_model.get('pricer_instance') or full_model.get('full_pricer_instance')
                if full_instance and hasattr(full_instance, 'price_scenarios'):
                    full_price = full_instance.price_scenarios(
                        raw_market_scenarios=base_scenario,
                        scenario_factor_names=var_factor_names,
                        rate_pillars=tenors
                    )[0]
                    full_base_value = full_price
                
                # Calculate percentage difference
                if isinstance(tff_base_value, (int, float)) and isinstance(full_base_value, (int, float)) and full_base_value != 0:
                    base_diff_pct = 100 * (tff_base_value - full_base_value) / full_base_value
            
            except Exception as e:
                status = 'ERROR'
                print(f"    Error pricing {instrument_id}: {str(e)[:50]}...")
            
            # Handle calibration errors
            if tff_error:
                status = 'TFF_FAIL'
                tff_fit_time = 'ERROR'
                tff_rmse = 'ERROR'
            
            # Format values
            fit_time_str = f"{tff_fit_time:.4f}s" if isinstance(tff_fit_time, (int, float)) else str(tff_fit_time)
            rmse_str = f"{tff_rmse:.6f}" if isinstance(tff_rmse, (int, float)) else str(tff_rmse)
            tff_base_str = f"{tff_base_value:,.4f}" if isinstance(tff_base_value, (int, float)) else str(tff_base_value)
            full_base_str = f"{full_base_value:,.4f}" if isinstance(full_base_value, (int, float)) else str(full_base_value)
            diff_str = f"{base_diff_pct:+.3f}%" if isinstance(base_diff_pct, (int, float)) else str(base_diff_pct)
            
            print(f"{instrument_id:<25} | {product_type:<15} | {fit_time_str:<12} | {rmse_str:<10} | {tff_base_str:<15} | {full_base_str:<15} | {diff_str:<12} | {status:<10}")
            count += 1
        
        # Summary statistics
        print(f"\n--- TFF vs Full Summary Statistics ---")
        
        # Calculate aggregate metrics
        successful_tff = [m for m in tff_model_registry.values() if not m.get('error_tff_calibration')]
        failed_tff = [m for m in tff_model_registry.values() if m.get('error_tff_calibration')]
        
        total_instruments = len(tff_model_registry)
        success_rate = len(successful_tff) / total_instruments if total_instruments > 0 else 0
        
        print(f"Total Instruments: {total_instruments:,}")
        print(f"TFF Calibration Success Rate: {success_rate:.1%} ({len(successful_tff):,}/{total_instruments:,})")
        
        if successful_tff:
            fit_times = [m.get('tff_fit_time_seconds', 0) for m in successful_tff if isinstance(m.get('tff_fit_time_seconds'), (int, float))]
            rmse_values = [m.get('tff_rmse', 0) for m in successful_tff if isinstance(m.get('tff_rmse'), (int, float))]
            
            if fit_times:
                print(f"Average TFF Fit Time: {np.mean(fit_times):.4f}s (min: {np.min(fit_times):.4f}s, max: {np.max(fit_times):.4f}s)")
            if rmse_values:
                print(f"Average TFF RMSE: {np.mean(rmse_values):.6f} (min: {np.min(rmse_values):.6f}, max: {np.max(rmse_values):.6f})")
        
        if failed_tff:
            print(f"\nFailed TFF Calibrations ({len(failed_tff):,}):")
            for i, failed_model in enumerate(failed_tff[:5]):  # Show first 5 failures
                error_msg = str(failed_model.get('error_tff_calibration', 'Unknown error'))
                if len(error_msg) > 60:
                    error_msg = error_msg[:57] + "..."
                # Find instrument ID for this failed model
                failed_id = next((k for k, v in tff_model_registry.items() if v == failed_model), 'Unknown')
                print(f"  {failed_id}: {error_msg}")
            if len(failed_tff) > 5:
                print(f"  ... and {len(failed_tff) - 5} more failures")
    
    else:
        print("\n--- TFF vs Full Comparison SKIPPED (Full revaluation not run) ---")

    print("\n--- End of Hybrid VaR Comparison Demo ---")
    
    return {
        "full_var": var_1pct_full,
        "tff_var": var_1pct_tff,
        "hybrid_var": var_1pct_hybrid,
        "base_value_full": base_value_full,
        "base_value_tff": base_value_tff,
    }

if __name__ == "__main__":
    print("Running Hybrid VaR Demo...")
    results = []
    num_of_trials = 1
    
    num_instruments_to_generate = 40
    # set tff_batch_size such that no more than 100 batches are created
    tff_batch_size = max(1, num_instruments_to_generate // 100)
    print(f"Using tff_batch_size: {tff_batch_size:,} (for {num_instruments_to_generate:,} instruments)")
    
    for i in range(num_of_trials):
        print(f"Trial {i + 1}")
        result = run_hybrid_var_demo(
            num_instruments_to_generate=num_instruments_to_generate,
            num_var_scenarios=2000,
            n_tff_domain_scenarios=200,
            n_tff_fitting_samples=50,
            hybrid_critical_percentile=0.015,
            num_tff_workers=os.cpu_count(),
            tff_batch_size=tff_batch_size,
            random_seed=i,
            # NEW: Choose which methods to run
            run_full_reval=True,   # Set to True to enable full revaluation
            run_tff_only=True,      # Set to True to enable TFF-only
            run_hybrid=True        # Set to True to enable hybrid (requires TFF)
        )
        results.append(result)
    
    # Filter out None values for analysis
    full_var_values = [res["full_var"] for res in results if res["full_var"] is not None]
    tff_var_values = [res["tff_var"] for res in results if res["tff_var"] is not None]
    hybrid_var_values = [res["hybrid_var"] for res in results if res["hybrid_var"] is not None]
    base_value_full_values = [res["base_value_full"] for res in results if res["base_value_full"] is not None]
    base_value_tff_values = [res["base_value_tff"] for res in results if res["base_value_tff"] is not None]
    
    # Only calculate comparisons if multiple methods were run
    if full_var_values and (tff_var_values or hybrid_var_values):
        if hybrid_var_values:
            hybrid_var_equal_full = sum(
                abs(hv - fv) < 0.01 for hv, fv in zip(hybrid_var_values, full_var_values[:len(hybrid_var_values)])
            ) / len(hybrid_var_values)
            print(f"Hybrid VaR equal to Full VaR within 0.01: {hybrid_var_equal_full:.2%} of trials")
        
        if tff_var_values and full_var_values:
            diff_full_var_tff = 100 * ((np.array(full_var_values[:len(tff_var_values)]) - np.array(tff_var_values)) / np.array(base_value_tff_values[:len(tff_var_values)]))
        
        if hybrid_var_values and full_var_values:
            diff_full_var_hybrid = 100 * ((np.array(full_var_values[:len(hybrid_var_values)]) - np.array(hybrid_var_values)) / np.array(base_value_full_values[:len(hybrid_var_values)]))
        
        if base_value_full_values and base_value_tff_values:
            diff_full_base_tff = 100 * ((np.array(base_value_full_values[:len(base_value_tff_values)]) - np.array(base_value_tff_values)) / np.array(base_value_tff_values))
        
        print("\n--- Summary of Results ---")
        print(f"{'Metric (pct of base)':<30} | {'Min':<10} | {'Max':<10} | {'Avg':<10}")
        print("-" * 70)
        
        if 'diff_full_var_tff' in locals():
            print(f"{'Diff Full Var - TFF Var':<30} | {np.min(diff_full_var_tff):<10.2f} | {np.max(diff_full_var_tff):<10.2f} | {np.mean(diff_full_var_tff):<10.2f}")
        
        if 'diff_full_var_hybrid' in locals():
            print(f"{'Diff Full Var - Hybrid Var':<30} | {np.min(diff_full_var_hybrid):<10.2f} | {np.max(diff_full_var_hybrid):<10.2f} | {np.mean(diff_full_var_hybrid):<10.2f}")
        
        if 'diff_full_base_tff' in locals():
            print(f"{'Diff Full Base - TFF Base':<30} | {np.min(diff_full_base_tff):<10.2f} | {np.max(diff_full_base_tff):<10.2f} | {np.mean(diff_full_base_tff):<10.2f}")
    
    else:
        print("\n--- Single Method Results ---")
        if tff_var_values:
            print(f"Average TFF VaR: {np.mean(tff_var_values):,.2f}")
            print(f"Average TFF Base Value: {np.mean(base_value_tff_values):,.2f}")
        if full_var_values:
            print(f"Average Full VaR: {np.mean(full_var_values):,.2f}")
            print(f"Average Full Base Value: {np.mean(base_value_full_values):,.2f}")
        if hybrid_var_values:
            print(f"Average Hybrid VaR: {np.mean(hybrid_var_values):,.2f}")

    print("\n--- Hybrid VaR Demo Results ---")