# demo_main.py

import QuantLib as ql
import numpy as np
import json                                          # ← for JSON serialization
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import time
import os

# ← TFF and portfolio workflow classes & helpers
from workflow_manager import (
    InstrumentProcessor,
    PortfolioBuilder,
    PortfolioAnalytics,
    generate_portfolio_specs_for_serialization,
    portfolio_json_serializer
)
from scenario_generator import SimpleRandomScenarioGenerator

def run_demonstration(
    enable_parallel_tff_fitting: bool = True,
    use_hardcoded_g2_params: bool = True
    ):
    print(f"--- FastRiskDemo with Workflow Manager (Parallel TFF: {enable_parallel_tff_fitting}) ---")

    # --- Global Setup ---
    val_d = date(2025, 5, 18)
    numeric_rate_tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"
    OPT_UNDERLYING_SYMBOL = "DEMO_OPT_STOCK"
    CONV_UNDERLYING_SYMBOL = "DEMO_CONV_STOCK"

    CREDIT_CURVE_FIN_AA = f"{DEMO_CURRENCY}_FIN_AA"
    CREDIT_CURVE_CORP_BBB = f"{DEMO_CURRENCY}_CORP_BBB"

    default_g2_p = (0.01, 0.003, 0.015, 0.006, -0.75) if use_hardcoded_g2_params else None

    base_demo_currency_rates_values = np.array([0.020, 0.021, 0.022, 0.025, 0.027, 0.030, 0.032, 0.033])
    base_rates_map = {f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_{t:.2f}Y": base_demo_currency_rates_values[i] for i, t in enumerate(numeric_rate_tenors)}

    opt_s0_factor = f"{DEMO_CURRENCY}_{OPT_UNDERLYING_SYMBOL}_S0"
    opt_vol_factor = f"{DEMO_CURRENCY}_{OPT_UNDERLYING_SYMBOL}_VOL"
    conv_s0_factor = f"{DEMO_CURRENCY}_{CONV_UNDERLYING_SYMBOL}_S0"
    conv_vol_factor = f"{DEMO_CURRENCY}_{CONV_UNDERLYING_SYMBOL}_EQVOL"
    conv_div_factor = f"{DEMO_CURRENCY}_{CONV_UNDERLYING_SYMBOL}_DIVYIELD"
    conv_cs_factor = f"{DEMO_CURRENCY}_{CONV_UNDERLYING_SYMBOL}_CS"

    base_s0_map = {opt_s0_factor: 100.0, conv_s0_factor: 100.0}
    base_vol_map = {opt_vol_factor: 0.25, conv_vol_factor: 0.25}

    base_other_factors_map_for_conv = {
        conv_div_factor: 0.01,
        conv_cs_factor: 0.015  # This is the single spread for CB engine, distinct from spread curves
    }
    temp_s0_map_for_gen = {**base_s0_map, **base_other_factors_map_for_conv}
    temp_vol_map_for_gen = {**base_vol_map}

    # Define base credit spread curves
    base_credit_spread_curves = {
        CREDIT_CURVE_FIN_AA: np.array([0.0050, 0.0055, 0.0060, 0.0065, 0.0070, 0.0075, 0.0080, 0.0085]),
        CREDIT_CURVE_CORP_BBB: np.array([0.0150, 0.0155, 0.0160, 0.0165, 0.0170, 0.0175, 0.0180, 0.0185])
    }
    credit_spread_tenors = { # Assuming same tenors as risk-free for simplicity
        CREDIT_CURVE_FIN_AA: numeric_rate_tenors,
        CREDIT_CURVE_CORP_BBB: numeric_rate_tenors
    }

    scenario_gen_global = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map,
        base_s0_map=temp_s0_map_for_gen,
        base_vol_map=temp_vol_map_for_gen,
        base_credit_spread_curves_map=base_credit_spread_curves, # NEW
        credit_spread_curve_tenors_map=credit_spread_tenors,   # NEW
        random_seed=42
    )
    N_GLOBAL_SCENARIOS = 100
    global_market_scenarios, global_factor_names = scenario_gen_global.generate_scenarios(N_GLOBAL_SCENARIOS)
    print(f"Generated {N_GLOBAL_SCENARIOS} global market scenarios with factors: {global_factor_names}")


    # --- Step 1: Instrument Processing and TFF Calibration ---
    print("\\n--- Step 1: Instrument Processing & TFF Calibration ---")

    default_conv_fixed_pricer_params = {
        's0_val': 100.0,
        'dividend_yield': 0.01,
        'equity_volatility': 0.25,
        'credit_spread': 0.015 # This is the single credit spread for the CB engine
    }

    instrument_definitions_data = [
        {
            "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_VANILLA_5Y",
            "product_type": "VanillaBond", "pricing_preference": "TFF",
            "params": { "valuation_date": val_d, "maturity_date": (val_d + relativedelta(years=5)),
                        "coupon_rate": 0.03, "face_value": 100.0, "currency": DEMO_CURRENCY,
                        "index_stub": DEMO_RATE_INDEX_STUB, "freq": 2, "settlement_days": 0 },
            "tff_config": {"n_train": 64, "n_test": 10}
        },
        { # NEW: Vanilla bond with credit spread
            "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_VANILLA_SPREAD_7Y",
            "product_type": "VanillaBond", "pricing_preference": "TFF", # Or FULL
            "params": { "valuation_date": val_d, "maturity_date": (val_d + relativedelta(years=7)),
                        "coupon_rate": 0.04, "face_value": 100.0, "currency": DEMO_CURRENCY,
                        "index_stub": DEMO_RATE_INDEX_STUB, "freq": 2, "settlement_days": 0,
                        "credit_spread_curve_name": CREDIT_CURVE_FIN_AA },
            "tff_config": {"n_train": 64, "n_test": 10}
        },
        {
            "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_CALLABLE_5Y_G2",
            "product_type": "CallableBond", "pricing_preference": "TFF",
            "params": { "valuation_date": val_d, "maturity_date": (val_d + relativedelta(years=5)),
                        "coupon_rate": 0.032, "face_value": 100.0, "currency": DEMO_CURRENCY,
                        "index_stub": DEMO_RATE_INDEX_STUB, "freq": 2,
                        "call_dates": [(val_d + relativedelta(years=y)) for y in [2,3,4]],
                        "call_prices": [102.0, 101.0, 100.0]},
            "pricer_params": {"g2_params": default_g2_p},
            "tff_config": {"n_train": 128, "n_test": 10}
        },
        { # NEW: Callable bond with credit spread
            "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_CALLABLE_SPREAD_6Y_G2",
            "product_type": "CallableBond", "pricing_preference": "TFF", # Or FULL
            "params": { "valuation_date": val_d, "maturity_date": (val_d + relativedelta(years=6)),
                        "coupon_rate": 0.042, "face_value": 100.0, "currency": DEMO_CURRENCY,
                        "index_stub": DEMO_RATE_INDEX_STUB, "freq": 2,
                        "call_dates": [(val_d + relativedelta(years=y)) for y in [2,3,4,5]],
                        "call_prices": [103.0, 102.0, 101.0, 100.0],
                        "credit_spread_curve_name": CREDIT_CURVE_CORP_BBB},
            "pricer_params": {"g2_params": default_g2_p},
            "tff_config": {"n_train": 128, "n_test": 10}
        },
        {
            "instrument_id": f"{DEMO_CURRENCY}_{CONV_UNDERLYING_SYMBOL}_CONV_BOND_5Y_S0_DYNAMIC",
            "product_type": "ConvertibleBond", "pricing_preference": "TFF",
            "params": {
                'valuation_date': val_d, 'issue_date': (val_d - relativedelta(months=6)),
                'maturity_date': (val_d + relativedelta(years=5, months=-6)), 'coupon_rate': 0.02,
                'conversion_ratio': 20.0, 'face_value': 100.0, 'currency': DEMO_CURRENCY,
                'index_stub': DEMO_RATE_INDEX_STUB, 'underlying_symbol': CONV_UNDERLYING_SYMBOL, 'freq': 2
                # No credit_spread_curve_name here, CB pricer uses single credit_spread from pricer_params
            },
            "pricer_params": default_conv_fixed_pricer_params,
            "tff_config": {"n_train": 128, "n_test": 10,
                           "convertible_tff_market_inputs_as_factors": False
                          }
        },
        {
            "instrument_id": f"{DEMO_CURRENCY}_{CONV_UNDERLYING_SYMBOL}_CONV_BOND_5Y_ALL_DYNAMIC",
            "product_type": "ConvertibleBond", "pricing_preference": "TFF",
            "params": {
                'valuation_date': val_d, 'issue_date': (val_d - relativedelta(months=6)),
                'maturity_date': (val_d + relativedelta(years=5, months=-6)), 'coupon_rate': 0.02,
                'conversion_ratio': 20.0, 'face_value': 100.0, 'currency': DEMO_CURRENCY,
                'index_stub': DEMO_RATE_INDEX_STUB, 'underlying_symbol': CONV_UNDERLYING_SYMBOL, 'freq': 2
            },
            "tff_config": {"n_train": 128, "n_test": 10,
                           "convertible_tff_market_inputs_as_factors": True
                          }
        },
        {
            "instrument_id": f"{DEMO_CURRENCY}_{OPT_UNDERLYING_SYMBOL}_EURO_CALL_1Y_STRIKE105_ORD2",
            "product_type": "EuropeanOption", "pricing_preference": "TFF",
            "params": { 'valuation_date': val_d, 'expiry_date': (val_d + relativedelta(years=1)),
                        'strike_price': 105.0, 'option_type': 'call',
                        'currency': DEMO_CURRENCY, 'underlying_symbol': OPT_UNDERLYING_SYMBOL,
            }, "pricer_params": { 'bs_risk_free_rate': 0.025, 'bs_dividend_yield': 0.01 },
            "tff_config": {"n_train": 128, "n_test": 10, "option_feature_order": 2}
        },
        { "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_VANILLA_10Y_FULL",
          "product_type": "VanillaBond", "pricing_preference": "FULL",
          "params": { "valuation_date": val_d, "maturity_date": (val_d + relativedelta(years=10)),
                      "coupon_rate": 0.035, "face_value": 100.0, "currency": DEMO_CURRENCY,
                      "index_stub": DEMO_RATE_INDEX_STUB, "freq": 2 }}
    ]

    instrument_processor = InstrumentProcessor(
        scenario_generator=scenario_gen_global, global_valuation_date=val_d,
        default_numeric_rate_tenors=numeric_rate_tenors, default_g2_params=default_g2_p,
        default_bs_risk_free_rate=0.025, default_bs_dividend_yield=0.01,
        parallel_workers_tff=os.cpu_count() if enable_parallel_tff_fitting else False,
        n_scenarios_for_tff_domain=500 )

    model_registry = instrument_processor.process_instruments(
        instrument_definitions_data, global_market_scenarios, global_factor_names )

    instrument_processor.save_model_registry("model_registry.json")

    # --- Step 2: Portfolio Construction ---
    print("\\n--- Step 2: Portfolio Construction ---")
    holdings_data = [
        {"client_id": "ClientA", "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_VANILLA_5Y", "num_holdings": 100},
        {"client_id": "ClientA", "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_CALLABLE_5Y_G2", "num_holdings": 50},
        {"client_id": "ClientB", "instrument_id": f"{DEMO_CURRENCY}_{CONV_UNDERLYING_SYMBOL}_CONV_BOND_5Y_S0_DYNAMIC", "num_holdings": 35},
        {"client_id": "ClientB", "instrument_id": f"{DEMO_CURRENCY}_{CONV_UNDERLYING_SYMBOL}_CONV_BOND_5Y_ALL_DYNAMIC", "num_holdings": 40},
        {"client_id": "ClientB", "instrument_id": f"{DEMO_CURRENCY}_{OPT_UNDERLYING_SYMBOL}_EURO_CALL_1Y_STRIKE105_ORD2", "num_holdings": 200},
        {"client_id": "ClientA", "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_VANILLA_10Y_FULL", "num_holdings": 80},
        {"client_id": "ClientA", "instrument_id": "MISSING_INSTRUMENT_ID_EXAMPLE", "num_holdings": 10},
        # Add new spread bonds to holdings
        {"client_id": "ClientA", "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_VANILLA_SPREAD_7Y", "num_holdings": 70},
        {"client_id": "ClientB", "instrument_id": f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_CALLABLE_SPREAD_6Y_G2", "num_holdings": 60},
    ]

    initial_portfolio_specs = generate_portfolio_specs_for_serialization(
        holdings_data=holdings_data, model_registry=model_registry,
        instrument_definitions_data_for_pricer_params=instrument_definitions_data )

    portfolio_builder_initial = PortfolioBuilder(model_registry)
    client_portfolios = portfolio_builder_initial.build_portfolios_from_specs(
        portfolio_specs_list=initial_portfolio_specs, global_valuation_date=val_d,
        default_g2_params=default_g2_p, default_bs_rfr=0.025, default_bs_div=0.01 )

    if portfolio_builder_initial.uncalculated_instruments:
        print(f"  WARNING (Initial Build): Uncalculated instruments: {portfolio_builder_initial.uncalculated_instruments}")

    # --- Step 3: Portfolio Pricing / VaR Calculation (using PortfolioAnalytics) ---
    print("\\n--- Step 3: Portfolio Pricing / VaR (using PortfolioAnalytics) ---")
    if client_portfolios:
        portfolio_analyzer = PortfolioAnalytics(
            client_portfolios=client_portfolios, global_market_scenarios=global_market_scenarios,
            global_factor_names=global_factor_names, numeric_rate_tenors=numeric_rate_tenors,
            scenario_generator_for_base_values=scenario_gen_global )
        var_results = portfolio_analyzer.run_var_analysis(var_percentiles=[1.0, 5.0])
    else: print("  No client portfolios were built, skipping VaR analysis.")

    # --- Step 4: JSON Serialization/Deserialization Demo for Portfolio Specs ---
    print("\\n--- Step 4: Portfolio JSON Serialization/Deserialization Demo ---")
    portfolio_json_string = None
    if initial_portfolio_specs:
        try:
            portfolio_json_string = json.dumps(initial_portfolio_specs, indent=4, default=portfolio_json_serializer)
            print("\\n--- Portfolio Specifications (JSON Serialized) ---")
            if initial_portfolio_specs:
                 print("Sample of first item in JSON specs:"); print(json.dumps(initial_portfolio_specs[0], indent=4, default=portfolio_json_serializer))
        except Exception as e: print(f"   ERROR serializing portfolio specs to JSON: {e}"); portfolio_json_string = None
    else: print("\\n--- No valid portfolio specifications to serialize to JSON ---")

    if portfolio_json_string:
        print("\\n--- Loading and Pricing Portfolio from JSON String ---")
        try:
            loaded_portfolio_specs_from_str = json.loads(portfolio_json_string)
            portfolio_builder_from_json = PortfolioBuilder(model_registry)
            client_portfolios_from_json = portfolio_builder_from_json.build_portfolios_from_specs(
                portfolio_specs_list=loaded_portfolio_specs_from_str, global_valuation_date=val_d,
                default_g2_params=default_g2_p, default_bs_rfr=0.025, default_bs_div=0.01 )
            if portfolio_builder_from_json.uncalculated_instruments:
                 print(f"  WARNING (JSON Load): Uncalculated instruments: {portfolio_builder_from_json.uncalculated_instruments}")
            if client_portfolios_from_json:
                reloaded_portfolio_analyzer = PortfolioAnalytics(
                    client_portfolios=client_portfolios_from_json, global_market_scenarios=global_market_scenarios,
                    global_factor_names=global_factor_names, numeric_rate_tenors=numeric_rate_tenors,
                    scenario_generator_for_base_values=scenario_gen_global )
                print("  Results for reloaded portfolio from JSON:")
                reloaded_var_results = reloaded_portfolio_analyzer.run_var_analysis(var_percentiles=[1.0, 5.0])
            else: print("  No client portfolios were built from JSON specs.")
        except Exception as e:
            print(f"   ERROR loading or pricing portfolio from JSON string: {e}")
            import traceback; traceback.print_exc()

    print("\\n--- End of Demonstration ---")

if __name__ == "__main__":
    try:
        print(f"QuantLib version: {ql.__version__}")
        run_demonstration( enable_parallel_tff_fitting=False )
    except NameError as e:
        if any(cn in str(e) for cn in ['ProductStaticBase','InstrumentProcessor','PortfolioBuilder','PortfolioAnalytics']):
            print(f"ERROR: Class not defined. Ensure all notebook cells for class definitions are executed. Details: {e}")
        elif 'QuantLib' in str(e) or 'ql' in str(e): print("ERROR: QuantLib not found/imported.")
        else: print(f"A NameError: {e}"); import traceback; traceback.print_exc()
    except Exception as e:
        print(f"An unexpected error: {e}"); import traceback; traceback.print_exc()
