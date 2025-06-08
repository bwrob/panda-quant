"""
Demonstrates full‐pricer valuation (no TFF) for a portfolio of instruments.
"""
import QuantLib as ql
import numpy as np
import time
from datetime import date
from dateutil.relativedelta import relativedelta

from scenario_generator import SimpleRandomScenarioGenerator
from demo_calibration import generate_instrument_definitions
from workflow_manager import (
    InstrumentProcessor, PortfolioBuilder,
    PortfolioAnalytics, generate_portfolio_specs_for_serialization
)

def run_full_valuation_demo(
    num_instruments_to_generate: int = 20,
    num_workers: int = None,
    batch_size: int = 4
):
    # --- Global Setup ---
    val_date = date(2025, 5, 18)
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0])
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"
    default_g2_params = (0.01, 0.003, 0.015, 0.006, -0.75)
    default_bs_rfr  = 0.025
    default_bs_div  = 0.01

    # Build base‐maps for scenario generator
    symbols = [f"STOCK_{i%10}" for i in range(num_instruments_to_generate)]
    base_rates_map = {
        f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t*0.001
        for t in tenors
    }
    base_s0_map = {
        f"{DEMO_CURRENCY}_{sym}_S0": 90 + np.random.rand()*20
        for sym in symbols
    }
    base_vol_map = {
        f"{DEMO_CURRENCY}_{sym}_VOL": 0.20 + np.random.rand()*0.1
        for sym in symbols
    }
    base_other = {}
    for sym in symbols:
        base_other[f"{DEMO_CURRENCY}_{sym}_DIVYIELD"] = 0.01 + np.random.rand()*0.01
        base_other[f"{DEMO_CURRENCY}_{sym}_CS"]         = 0.01 + np.random.rand()*0.01

    scenario_generator = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates_map,
        base_s0_map={**base_s0_map, **base_other},
        base_vol_map=base_vol_map,
        random_seed=42
    )
    global_scenarios, factor_names = scenario_generator.generate_scenarios(200)

    # --- Instrument Definitions (force FULL) ---
    defs = generate_instrument_definitions(num_instruments_to_generate, val_date)
    for d in defs:
        d["pricing_preference"] = "FULL"

    print(f"Valuating {len(defs)} instruments with full pricers...")
    iproc = InstrumentProcessor(
        scenario_generator=scenario_generator,
        global_valuation_date=val_date,
        default_numeric_rate_tenors=tenors,
        default_g2_params=default_g2_params,
        default_bs_risk_free_rate=default_bs_rfr,
        default_bs_dividend_yield=default_bs_div,
        parallel_workers_tff=False,
        n_scenarios_for_tff_domain=0
    )

    t0 = time.time()
    registry = iproc.process_instruments(
        defs, global_scenarios, factor_names, batch_size=batch_size
    )
    print(f"Total full‐valuation time: {time.time()-t0:.2f}s")

    # --- Portfolio Construction & Analytics ---
    holdings = [
        {"client_id": "ClientFullDemo",
         "instrument_id": d["instrument_id"],
         "num_holdings": 100}
        for d in defs
    ]
    specs = generate_portfolio_specs_for_serialization(holdings, registry)
    builder = PortfolioBuilder(registry)
    portfolios = builder.build_portfolios_from_specs(
        specs, val_date,
        default_g2_params, default_bs_rfr, default_bs_div
    )

    pa = PortfolioAnalytics(
        client_portfolios=portfolios,
        global_market_scenarios=global_scenarios,
        global_factor_names=factor_names,
        numeric_rate_tenors=tenors,
        scenario_generator_for_base_values=scenario_generator
    )
    var = pa.run_var_analysis(var_percentiles=[1.0, 5.0])
    print("VaR results:", var)

if __name__ == "__main__":
    run_full_valuation_demo()