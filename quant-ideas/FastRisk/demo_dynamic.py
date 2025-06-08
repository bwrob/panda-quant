from custom_importer  import create_product_static_from_dict
from pricers          import create_pricer
from datetime         import date
from dateutil.relativedelta import relativedelta
import numpy as np

def demo_dynamic_option():
    """
    Demonstrates how to dynamically create and price a European option using a pricer.
    This function builds a static product definition, creates a pricer instance,
    and then prices the option for given scenarios.
    """
    # 1) Build your static product
    option_config = {
        "module_name":     "product_definitions",
        "product_type":    "EuropeanOptionStatic",
        "valuation_date":  date.today().isoformat(),
        "expiry_date":     (date.today()+relativedelta(years=1)).isoformat(),
        "strike_price":    150.0,
        "option_type":     "call",
        "currency":        "USD",
        "underlying_symbol": "AAPL"
    }
    static_opt = create_product_static_from_dict(option_config)

    # 2) Build your pricer _instance_
    pricer_cfg = {
        "pricer_module_name": "black_scholes_pricer",
        "pricer_class_name":  "BlackScholesPricer",
        "pricer_params": {
        }
    }
    opt_pricer = create_pricer(static_opt, pricer_cfg)

    price0 = opt_pricer.price(160.0, 0.2, risk_free_rate=0.01, dividend_yield=0.02)
    print("single‐scenario price:", price0)

    #   Or arrays
    S_arr   = [150.0,160.0,170.0]
    vol_arr = [0.2,0.25,0.3]
    print("vectorized prices:", opt_pricer.price(S_arr, vol_arr, risk_free_rate=0.01, dividend_yield=0.02))
    
def demo_dynamic_bond():
    """
    Demonstrates how to dynamically create and price a bond using a pricer.
    This function builds a static product definition, creates a pricer instance,
    and then prices the bond for given scenarios.
    """
    # 1) Build your static product
    bond_config = {
        "module_name":     "product_definitions",
        "product_type":    "QuantLibBondStaticBase",
        "valuation_date":  date.today().isoformat(),
        "maturity_date":   (date.today()+relativedelta(years=5)).isoformat(),
        "face_value":      100.0,
        "coupon_rate":     0.05,
        "currency":        "USD"
    }
    static_bond = create_product_static_from_dict(bond_config)

    # 2) Build your pricer _instance_
    pricer_cfg = {
        "pricer_module_name": "quantlib_bond_pricer",
        "pricer_class_name":  "QuantLibBondPricer",
        "pricer_params": {
        }
    }
    bond_pricer = create_pricer(static_bond, pricer_cfg)
    
    bond_price = bond_pricer.price(market_scenario_data=np.array([.02, 0.025, 0.03, 0.035, 0.04, 0.045]),
                      pillar_times=np.array([0.5, 1.0, 2.0, 3.0, 4.0, 5.0]))
    
    print("Bond static definition:", static_bond.to_dict())
    print("Bond price:", bond_price)
    
def demo_dynamic_callable_bond():
    """
    Demonstrates how to dynamically create and price a callable bond using a pricer.
    This function builds a static product definition, creates a pricer instance,
    and then prices the callable bond for given scenarios.
    """
    # 1) Build your static product
    callable_bond_config = {
        "module_name":     "product_definitions",
        "product_type":    "CallableBondStaticBase",
        "valuation_date":  date.today().isoformat(),
        "maturity_date":   (date.today()+relativedelta(years=5)).isoformat(),
        "face_value":      100.0,
        "coupon_rate":     0.05,
        "currency":        "USD",
        "call_dates":      [(date.today()+relativedelta(years=2)).isoformat()],
        "call_prices":     [102.0]
    }
    static_callable_bond = create_product_static_from_dict(callable_bond_config)

    # 2) Build your pricer _instance_
    pricer_cfg = {
        "pricer_module_name": "quantlib_bond_pricer",
        "pricer_class_name":  "QuantLibBondPricer",
        "pricer_params": {
            "method": "g2",
            "grid_steps": 16
        }
    }
    callable_bond_pricer = create_pricer(static_callable_bond, pricer_cfg)
    
    g2_p = [0.01, 0.003, 0.015, 0.006, -0.75]  # (a,σ,b,η,ρ)
    pillar_times   = np.array([0.5,1.0,2.0,3.0,4.0,5.0])
    scenarios      = np.array([[0.02,0.025,0.03,0.035,0.04,0.045]])

    price = callable_bond_pricer.price(
        pillar_times=pillar_times,
        market_scenario_data=scenarios,
        g2_params=g2_p            # <— pass your G2 parameters here
    )
    
    print("Callable Bond static definition:", static_callable_bond.to_dict())
    print("Callable Bond price:", price)


def demo_dynamic_convertible_bond():
    """
    Demonstrates how to dynamically create and price a convertible bond using a pricer.
    """
    # 1) Build your static product
    conv_config = {
        "module_name":      "product_definitions",
        "product_type":     "ConvertibleBondStaticBase",
        "valuation_date":   date.today().isoformat(),
        "issue_date":       (date.today() - relativedelta(months=6)).isoformat(),
        "maturity_date":    (date.today() + relativedelta(years=5)).isoformat(),
        "coupon_rate":      0.04,
        "conversion_ratio": 5.0,
        "face_value":       100.0,
        "freq":             2,
        "currency":         "USD",
        "underlying_symbol":"DEMO_STOCK"
    }
    static_conv = create_product_static_from_dict(conv_config)

    # 2) Build your pricer instance for convertible‐binomial method
    pricer_cfg = {
        "pricer_module_name": "quantlib_bond_pricer",
        "pricer_class_name":  "QuantLibBondPricer",
        "pricer_params": {
            "method":         "convertible_binomial",
            "convertible_engine_steps": 100
        }
    }
    conv_pricer = create_pricer(static_conv, pricer_cfg)

    # 3) Prepare scenario: risk‐free pillars only and static credit for engine
    pillar_times = np.array([0.5,1.0,2.0,3.0,4.0,5.0], dtype=float)
    rf_curve     = np.array([[0.02,0.025,0.03,0.035,0.04,0.045]])
    # static equity & credit inputs
    price = conv_pricer.price(
        pillar_times=pillar_times,
        market_scenario_data=rf_curve,
        s0_val=100.0,
        dividend_yield=0.01,
        equity_volatility=0.20,
        credit_spread=0.015
    )

    print("Convertible Bond static definition:", static_conv.to_dict())
    print("Convertible Bond price:", price)

if __name__ == "__main__":
    demo_dynamic_option()
    demo_dynamic_bond()
    demo_dynamic_callable_bond()
    demo_dynamic_convertible_bond()
