import QuantLib as ql
import numpy as np
from datetime import date

from base_pricer import PricerBase
from product_definitions import QuantLibBondStaticBase

class FastBondPricer(PricerBase):
    """
    Fast NumPy-based pricer for vanilla fixed-rate bonds.
    Can optionally handle a simple additive credit spread.
    """
    def __init__(self, bond_static: QuantLibBondStaticBase):
        super().__init__(bond_static)
        if not isinstance(bond_static, QuantLibBondStaticBase):
            raise TypeError("FastBondPricer requires a QuantLibBondStaticBase instance.")
        self._gen_cashflows()

    def _gen_cashflows(self):
        bond_def: QuantLibBondStaticBase = self.product_static
        # Ensure QL eval date is set for cashflow generation if not already
        if ql.Settings.instance().evaluationDate != bond_def.ql_valuation_date:
            ql.Settings.instance().evaluationDate = bond_def.ql_valuation_date

        ql_sched = bond_def.schedule
        dc = bond_def.day_count_ql
        ql_val = bond_def.ql_valuation_date
        coupon_amt = bond_def.face_value * bond_def.coupon_rate / bond_def.freq

        cf_d, cf_t, cf_a = [],[],[]
        for i in range(len(ql_sched)):
            payment_date_ql = ql_sched[i]
            if payment_date_ql <= ql_val: continue # Skip past cashflows

            cf_d.append(payment_date_ql.to_date()) # Convert ql.Date to datetime.date
            # Calculate time to payment from valuation date
            t = dc.yearFraction(ql_val, payment_date_ql)
            cf_t.append(t)

            current_cf_amount = coupon_amt
            # Check if it's the maturity date by comparing with the bond's maturity date
            if payment_date_ql == bond_def.ql_maturity_date:
                current_cf_amount += bond_def.face_value
            cf_a.append(current_cf_amount)

        self.cf_dates: list[date] = cf_d
        self.cf_times: np.ndarray = np.array(cf_t, dtype=float)
        self.cf_amounts: np.ndarray = np.array(cf_a,  dtype=float)

        # Filter out any cashflows with non-positive time (should not happen with above logic but good practice)
        valid_cfs = self.cf_times > 1e-9
        self.cf_times = self.cf_times[valid_cfs]
        self.cf_amounts = self.cf_amounts[valid_cfs]
        self.cf_dates = [dt_obj for i, dt_obj in enumerate(self.cf_dates) if valid_cfs[i]]


    def price(self, pillar_times: np.ndarray, market_scenario_data: np.ndarray,
              credit_spread_pillar_times: np.ndarray = None,
              **kwargs) -> np.ndarray:

        num_rf_pillars = len(pillar_times)

        if market_scenario_data.ndim == 1:
            market_scenario_data = market_scenario_data.reshape(1, -1) # Ensure 2D for consistency

        risk_free_rates_scen = market_scenario_data[:, :num_rf_pillars]
        final_rates_scen = risk_free_rates_scen.copy()

        if hasattr(self.product_static, 'credit_spread_curve_name') and \
           self.product_static.credit_spread_curve_name and \
           credit_spread_pillar_times is not None and credit_spread_pillar_times.size > 0:

            num_cs_pillars = len(credit_spread_pillar_times)
            if market_scenario_data.shape[1] < num_rf_pillars + num_cs_pillars:
                raise ValueError(f"Market scenario data has insufficient columns for risk-free rates ({num_rf_pillars}) and credit spreads ({num_cs_pillars}). Got {market_scenario_data.shape[1]} cols.")

            credit_spreads_scen = market_scenario_data[:, num_rf_pillars : num_rf_pillars + num_cs_pillars]

            # Interpolate credit spreads to risk-free pillar times before adding
            interp_credit_spreads_matrix = np.array(
                [np.interp(pillar_times, credit_spread_pillar_times, cs_row) for cs_row in credit_spreads_scen]
            )
            final_rates_scen += interp_credit_spreads_matrix

        if not self.cf_times.size:
            return np.zeros(final_rates_scen.shape[0])

        prices = np.zeros(final_rates_scen.shape[0])
        for i, single_scenario_final_rates in enumerate(final_rates_scen):
            r_cf = np.interp(self.cf_times, pillar_times, single_scenario_final_rates)
            dfs  = np.exp(-r_cf * self.cf_times)
            prices[i] = float(self.cf_amounts.dot(dfs))

        return prices
    
    def price_scenarios(
        self,
        raw_market_scenarios: np.ndarray,
        scenario_factor_names: list[str],
        rate_pillars: np.ndarray | None = None,
        **price_kwargs
    ) -> np.ndarray:
        # reuse QuantLibBondPricer’s slicing logic
        from quantlib_bond_pricer import QuantLibBondPricer
        return QuantLibBondPricer.price_scenarios(
            self, raw_market_scenarios, scenario_factor_names,
            rate_pillars, **price_kwargs
        )