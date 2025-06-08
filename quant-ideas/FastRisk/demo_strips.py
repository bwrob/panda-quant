import time
import numpy as np
from datetime import date
from dateutil.relativedelta import relativedelta

from scenario_generator import SimpleRandomScenarioGenerator
from demo_calibration import generate_instrument_definitions
# Removed: from custom_importer import create_product_static_from_dict
# Removed: from pricers import create_pricer
from workflow_manager import InstrumentProcessor, generate_price_strips # Added

def run_full_price_strips(
    num_instruments: int = 20, # Increased default for better demo
    num_scenarios: int = 1000,
    num_workers: int = None,
    random_seed: int = 42
) -> dict[str, np.ndarray]:
    """
    Generate full‐pricer price strips for a list of demo instruments
    using the generate_price_strips utility, with parameters aligned
    with demo_calibration.py.
    Returns a dict mapping instrument_id to array of prices (length=num_scenarios).
    """
    # 1) Global setup (aligned with demo_calibration.py)
    val_date = date(2025, 5, 18) # From demo_calibration.py
    tenors = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0], dtype=float) # From demo_calibration.py
    DEMO_CURRENCY = "USD"
    DEMO_RATE_INDEX_STUB = "IR"
    default_g2_params = (0.01, 0.003, 0.015, 0.006, -0.75) # From demo_calibration.py
    default_bs_rfr  = 0.025
    default_bs_div  = 0.01

    # Build base maps for scenario generator (aligned with demo_calibration.py)
    base_rates = {f"{DEMO_CURRENCY}_{DEMO_RATE_INDEX_STUB}_{t:.2f}Y": 0.02 + t*0.001
                  for t in tenors} # Formula from demo_calibration.py
    
    symbols = list(set([f"STOCK_{i%10}" for i in range(num_instruments)])) # Ensure unique symbols
    base_s0_map = {}
    base_vol_map = {}
    base_other_factors = {}

    for sym in symbols:
        base_s0_map[f"{DEMO_CURRENCY}_{sym}_S0"] = 90 + np.random.rand()*20
        base_vol_map[f"{DEMO_CURRENCY}_{sym}_VOL"] = 0.20 + np.random.rand()*0.1
        base_vol_map[f"{DEMO_CURRENCY}_{sym}_EQVOL"] = 0.20 + np.random.rand() * 0.1 # For convertibles
        base_other_factors[f"{DEMO_CURRENCY}_{sym}_DIVYIELD"] = 0.01 + np.random.rand() * 0.01
        base_other_factors[f"{DEMO_CURRENCY}_{sym}_CS"] = 0.01 + np.random.rand() * 0.01

    merged_s0_map = {**base_s0_map, **base_other_factors} # As in demo_calibration

    scen_gen = SimpleRandomScenarioGenerator(
        base_rates_map=base_rates,
        base_s0_map=merged_s0_map, 
        base_vol_map=base_vol_map,
        random_seed=random_seed
    )
    scenarios, factor_names = scen_gen.generate_scenarios(num_scenarios)
    # 2) Instrument definitions
    # generate_instrument_definitions already provides 'params' and 'pricer_params'
    # which are used by generate_price_strips and InstrumentProcessor
    defs = generate_instrument_definitions(num_instruments, val_date)
    # Ensure all instruments are set for FULL pricing if generate_price_strips relies on this
    # (generate_price_strips inherently uses full pricers via _create_pricer_template)
    for d_spec in defs:
        d_spec["pricing_preference"] = "FULL" # Retained for clarity, though generate_price_strips implies full

    # 3) Instantiate InstrumentProcessor
    iproc = InstrumentProcessor(
        scenario_generator=scen_gen,
        global_valuation_date=val_date,
        default_numeric_rate_tenors=tenors,
        default_g2_params=default_g2_params,
        default_bs_risk_free_rate=default_bs_rfr,
        default_bs_dividend_yield=default_bs_div,
        parallel_workers_tff=False, # Not used for full pricing
        n_scenarios_for_tff_domain=0 # Not used for full pricing
    )

    # 4) Generate price strips using the utility function
    strips = generate_price_strips(
        instrument_specs=defs,
        global_market_scenarios=scenarios,
        global_factor_names=factor_names,
        iproc=iproc,
        num_workers=num_workers
    )

    # Go thru the strips do a running total of vector, strips is key by a string and contain a 1d vector
    holding_size = 1000
    total_price = 0.0
    for instrument_id, prices in strips.items():
        if isinstance(prices, np.ndarray) and prices.ndim == 1:
            total_price += prices * holding_size
            
    # work out the percentile of the portfolio, 95% and 99%
    total_price = np.array(total_price)
    base_price = total_price[0]
    percentile_95 = np.percentile(total_price, 95) - base_price
    percentile_99 = np.percentile(total_price, 99) - base_price
    print(f"Base price of portfolio: {base_price:.2f}")
    print(f"95th percentile of portfolio price: {percentile_95:.2f}")
    print(f"99th percentile of portfolio price: {percentile_99:.2f}")

    return strips

if __name__ == "__main__":
    start_time = time.time()
    print(f"Running demo to generate full price strips...")
    price_strips = run_full_price_strips(num_instruments=10000, num_scenarios=2000, num_workers=24)
    print(f"\nGenerated price strips for {len(price_strips)} instruments")
    total_time= time.time() - start_time
    print(f"Total time taken: {total_time:.2f} seconds")
    print(f"Average time per instrument: {total_time / len(price_strips):.4f} seconds")