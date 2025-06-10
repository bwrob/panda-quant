# tff_calibration_demo.py
"""
Demonstrates parallel TFF calibration for a large number of instruments
using the InstrumentProcessor, followed by portfolio construction and analytics.
"""
import QuantLib as ql
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import time
import os
import json

from scenario_generator import SimpleRandomScenarioGenerator
from workflow_manager import InstrumentProcessor, PortfolioAnalytics, PortfolioBuilder, generate_portfolio_specs_for_serialization

# Note: Pricer and TFFCalibrate classes are used internally by InstrumentProcessor

def generate_instrument_definitions(num_instruments: int, val_date_param: date) -> list[dict]:
    """
    Generates a list of diverse instrument definitions.
    Uses val_date_param consistently.
    """
    definitions = []
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"

    # Default pricer params for options
    default_option_pricer_params = { 'bs_risk_free_rate': 0.025, 'bs_dividend_yield': 0.01 }
    conv_s0_dynamic_fixed_params = {'dividend_yield': 0.01, 'equity_volatility': 0.25, 'credit_spread': 0.015, 's0_val': 100.0}

    # fixed the seed for reproducibility
    np.random.seed(42)
    
    for i in range(num_instruments):
        instrument_type_choice = i % 4
        instrument_id_suffix = f"INST_{i+1}"

        maturity_years = np.random.randint(2, 11)
        maturity_dt = val_date_param + relativedelta(years=maturity_years)
        coupon = 0.02 + np.random.rand() * 0.03

        if instrument_type_choice == 0: # Vanilla Bond
            instrument_id = f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_VANILLA_{instrument_id_suffix}"
            definitions.append({
                "instrument_id": instrument_id,
                "product_type": "VanillaBond", "pricing_preference": "TFF",
                "params": {
                    "valuation_date": val_date_param.isoformat(), "maturity_date": maturity_dt.isoformat(),
                    "coupon_rate": coupon, "face_value": 100.0, "currency": DEMO_CURRENCY,
                    "index_stub": DEMO_RATE_INDEX_STUB, "freq": 2, "settlement_days": 0 },
                "tff_config": {"n_train": 64, "n_test": 4, "seed": i}
            })
        elif instrument_type_choice == 1: # Callable Bond
            instrument_id = f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_CALLABLE_{instrument_id_suffix}"
            call_offset = maturity_years // 2
            call_dates_list = []
            if call_offset >=1 :
                 call_dates_list = [(val_date_param + relativedelta(years=y)).isoformat() for y in range(call_offset, maturity_years)]

            definitions.append({
                "instrument_id": instrument_id,
                "product_type": "CallableBond", "pricing_preference": "TFF",
                "params": {
                    "valuation_date": val_date_param.isoformat(), "maturity_date": maturity_dt.isoformat(),
                    "coupon_rate": coupon, "face_value": 100.0, "currency": DEMO_CURRENCY,
                    "index_stub": DEMO_RATE_INDEX_STUB, "freq": 2,
                    "call_dates": call_dates_list,
                    "call_prices": [100.0 + len(call_dates_list) - j for j in range(len(call_dates_list))] if call_dates_list else []
                },
                "pricer_params": {"g2_params": (0.01, 0.003, 0.015, 0.006, -0.75), "g2_grid_steps": 8},
                "tff_config": {"n_train":64, "n_test": 4, "seed": i}
            })

        elif instrument_type_choice == 2: # European Option
            opt_underlying_sym = f"STOCK_{i%10}"
            strike = 90 + np.random.rand() * 20
            instrument_id = f"{DEMO_CURRENCY}_{opt_underlying_sym}_EURO_CALL_1Y_K{int(strike)}_{instrument_id_suffix}"
            definitions.append({
                "instrument_id": instrument_id,
                "product_type": "EuropeanOption", "pricing_preference": "TFF",
                "params": {
                    'valuation_date': val_date_param.isoformat(), 'expiry_date': (val_date_param + relativedelta(years=1)).isoformat(),
                    'strike_price': strike, 'option_type': 'call',
                    'currency': DEMO_CURRENCY, 'underlying_symbol': opt_underlying_sym,
                },
                "pricer_params": default_option_pricer_params,
                "tff_config": {"n_train": 128, "n_test": 4, "option_feature_order": 2, "seed": i}
            })
        elif instrument_type_choice == 3: # Convertible Bond (S0 dynamic, others fixed for TFF)
            conv_underlying_sym = f"STOCK_{i%10}"
            instrument_id = f"{DEMO_CURRENCY}_{conv_underlying_sym}_CONV_S0_DYN_{instrument_id_suffix}"
            definitions.append({
            "instrument_id": instrument_id,
            "product_type": "ConvertibleBond", "pricing_preference": "TFF",
            "params": {
                'valuation_date': val_date_param.isoformat(), 'issue_date': (val_date_param - relativedelta(months=6)).isoformat(),
                'maturity_date': maturity_dt.isoformat(), 'coupon_rate': coupon,
                'conversion_ratio': 1.0, 'face_value': 1.0,
                'currency': DEMO_CURRENCY, 'index_stub': DEMO_RATE_INDEX_STUB,
                'underlying_symbol': conv_underlying_sym, 'freq': 2
            },
            "pricer_params": conv_s0_dynamic_fixed_params, # Used for fixed params in TFF training
            "tff_config": {"n_train": 128, "n_test": 8, "seed": i,
                   "convertible_tff_market_inputs_as_factors": True
                  }
            })
        # elif instrument_type_choice == 4: # MBS Pool
        #     continue # Skip MBS Pool for now
        #     instrument_id = f"{DEMO_CURRENCY}_MBS_POOL_{instrument_id_suffix}"
        #     wac = 0.03 + np.random.rand() * 0.02
        #     definitions.append({
        #         "instrument_id": instrument_id,
        #         "product_type": "MBSPool", "pricing_preference": "TFF",
        #         "params": {
        #             'valuation_date': val_date_param, 'issue_date': (val_date_param - relativedelta(months=6)),
        #             'original_balance': 100.0, 'current_balance': 95.0,
        #             'wac': wac, 'pass_through_rate': 0.0325,
        #             'original_term_months': 360, 'age_months': 6,
        #             'prepayment_model_type': 'RefiIncentive',
        #             'delay_days': 30, 'currency': DEMO_CURRENCY, 'index_stub': DEMO_RATE_INDEX_STUB
        #         },
        #         "tff_config": {"n_train": 32, "n_test": 4, "seed": i}
        #     })
    return definitions


def run_calibration_demo(
    num_instruments_to_generate: int = 100, # Changed to 1000
    num_workers: int = None, # Use all available cores by default
    batch_size: int = 4, # Batch size for parallel processing
    random_seed: int = 42 # For reproducibility
    ):
    print(f"---  TFF Calibration Demo ---")

    # --- Global Setup ---
    np.random.seed(random_seed)  # For reproducibility
    val_d_main = date(2025, 5, 18)
    numeric_rate_tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"
    default_g2_p = (0.01, 0.003, 0.015, 0.006, -0.75)

    base_rates_map = {f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t*0.001 for t in numeric_rate_tenors}

    all_underlying_symbols = list(set([f"STOCK_{i%10}" for i in range(num_instruments_to_generate)]))
    base_s0_map = {}
    base_vol_map = {}
    base_other_map = {}

    for sym in all_underlying_symbols:
        base_s0_map[f"{DEMO_CURRENCY}_{sym}_S0"] = 90 + np.random.rand() * 20
        base_vol_map[f"{DEMO_CURRENCY}_{sym}_VOL"] = 0.20 + np.random.rand() * 0.1
        base_vol_map[f"{DEMO_CURRENCY}_{sym}_EQVOL"] = 0.20 + np.random.rand() * 0.1
        base_other_map[f"{DEMO_CURRENCY}_{sym}_DIVYIELD"] = 0.01 + np.random.rand() * 0.01
        base_other_map[f"{DEMO_CURRENCY}_{sym}_CS"] = 0.01 + np.random.rand() * 0.01

    merged_s0_map = {**base_s0_map, **base_other_map}

    scenario_gen_global = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map,
        base_s0_map=merged_s0_map,
        base_vol_map=base_vol_map,
        random_seed=42
    )

    N_DOMAIN_SCENARIOS = 2000
    global_market_scenarios, global_factor_names = scenario_gen_global.generate_scenarios(N_DOMAIN_SCENARIOS)
    print(f"Generated {N_DOMAIN_SCENARIOS} global scenarios for TFF domain with {len(global_factor_names)} factors.")

    instrument_definitions = generate_instrument_definitions(num_instruments_to_generate, val_d_main)
    print(f"Created {len(instrument_definitions)} instrument definitions.")

    # --- Save Instrument Definitions ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    instrument_defs_path = os.path.join(output_dir, "instrument_definitions.json")
    with open(instrument_defs_path, 'w') as f:
        json.dump(instrument_definitions, f, indent=4)
    print(f"Instrument definitions saved to {instrument_defs_path}.")

    # --- Instrument Processing with Parallel TFF Calibration ---
    print("\\n--- Instrument Processing & TFF Calibration ---")

    instrument_processor = InstrumentProcessor(
        scenario_generator=scenario_gen_global,
        global_valuation_date=val_d_main,
        default_numeric_rate_tenors=numeric_rate_tenors,
        default_g2_params=(0.01, 0.003, 0.015, 0.006, -0.75),
        default_bs_risk_free_rate=0.025,
        default_bs_dividend_yield=0.01,
        parallel_workers_tff=num_workers,
        n_scenarios_for_tff_domain=50 # Smaller set for faster sample_and_fit domain slicing
    )

    start_processing_time = time.time()
    model_registry = instrument_processor.process_instruments(
        instrument_definitions,
        global_market_scenarios,
        global_factor_names,
        batch_size=batch_size,
    )
    total_processing_time = time.time() - start_processing_time

    print(f"\nTotal instrument processing time: {total_processing_time:.2f} seconds.")

    # --- Save Model Registry ---
    model_registry_path = os.path.join(output_dir, "model_registry.json")
    with open(model_registry_path, 'w') as f:
        json.dump(model_registry, f, indent=4)
    print(f"Model registry saved to {model_registry_path}.")
    print(f"Processed {len(model_registry)} instruments in the registry.")

    successful_tff_calibrations = 0
    failed_calibrations = 0
    for instrument_id, entry in model_registry.items():
        if entry.get('pricing_method') == 'TFF' and 'tff_model_dict' in entry and not entry.get('error_tff_calibration'):
            successful_tff_calibrations += 1
        elif entry.get('error') or entry.get('error_tff_calibration'):
            failed_calibrations +=1
            # print(f"  TFF failed for {instrument_id}. Error: {entry.get('error_tff_calibration', entry.get('error'))}")


    print("\n--- Processing Summary ---")
    print(f"Total instruments to process: {len(instrument_definitions)}")
    print(f"Successfully calibrated TFF models: {successful_tff_calibrations}")
    print(f"Failed TFF calibrations/processing errors: {failed_calibrations}")
    if len(instrument_definitions) > 0 :
        print(f"Average processing time per instrument: {total_processing_time/len(instrument_definitions):.4f} seconds.")


    # --- Portfolio Construction & Analytics ---
    port_time = time.time()
    print("\n--- Portfolio Construction & Analytics ---")
    if successful_tff_calibrations == 0:
        print("No TFF models were successfully calibrated. Skipping portfolio analytics.")
    else:
        holdings_data = []
        for instrument_id, entry in model_registry.items():
            if entry.get('pricing_method') == 'TFF' and 'tff_model_dict' in entry and not entry.get('error_tff_calibration'):
                holdings_data.append({
                    "client_id": "ClientParallelDemo",
                    "instrument_id": instrument_id,
                    "num_holdings": 1000 # As requested
                })

        if not holdings_data:
            print("No successfully calibrated TFF instruments to add to portfolio.")
        else:
            print(f"Creating portfolio with {len(holdings_data)} successfully TFF-calibrated instruments.")
            initial_portfolio_specs = generate_portfolio_specs_for_serialization(
                holdings_data=holdings_data,
                model_registry=model_registry,
                instrument_definitions_data_for_pricer_params=instrument_definitions # Pass original defs
            )

            portfolio_builder = PortfolioBuilder(model_registry)
            client_portfolios = portfolio_builder.build_portfolios_from_specs(
                portfolio_specs_list=initial_portfolio_specs,
                global_valuation_date=val_d_main,
                default_g2_params=default_g2_p, # Pass defaults for full pricer reconstruction if needed
                default_bs_rfr=0.025,
                default_bs_div=0.01
            )

            if portfolio_builder.uncalculated_instruments:
                print(f"  WARNING: Uncalculated instruments during portfolio build: {portfolio_builder.uncalculated_instruments}")

            if client_portfolios.get("ClientParallelDemo"):
                portfolio_analyzer = PortfolioAnalytics(
                    client_portfolios=client_portfolios,
                    global_market_scenarios=global_market_scenarios, # Use the smaller domain scenarios for VaR demo
                    global_factor_names=global_factor_names,
                    numeric_rate_tenors=numeric_rate_tenors,
                    scenario_generator_for_base_values=scenario_gen_global )

                print("\n--- VaR Calculation for 'ClientParallelDemo' Portfolio ---")
                var_results = portfolio_analyzer.run_var_analysis(var_percentiles=[1.0, 5.0])
            else:
                print("Portfolio 'ClientParallelDemo' not built.")
    print(f"Portfolio construction and analytics time: {time.time() - port_time:.5f} seconds.")
    print("\\n--- End of Calibration Demonstration ---")


if __name__ == "__main__":
    try:
        print(f"QuantLib version: {ql.__version__}")
        start_time = time.time()
        run_calibration_demo(
            num_instruments_to_generate=100, # As per user request
            num_workers=24,
            batch_size=32,
            random_seed=42 # For reproducibility
        )
        total_time = time.time() - start_time
        print(f"Total demo time: {total_time:.2f} seconds.")
    except NameError as e:
        # Add TFFConfigurationFactory if it's directly used and might cause NameError
        if any(cn in str(e) for cn in ['ProductStaticBase','InstrumentProcessor','PortfolioBuilder','PortfolioAnalytics', 'TFFConfigurationFactory']):
            print(f"ERROR: Class not defined. Ensure all notebook cells for class definitions are executed. Details: {e}")
        elif 'QuantLib' in str(e) or 'ql' in str(e): print("ERROR: QuantLib not found/imported.")
        else: print(f"A NameError: {e}"); import traceback; traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error: {e}"); import traceback; traceback.print_exc()
