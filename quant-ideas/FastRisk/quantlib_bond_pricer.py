import QuantLib as ql
import numpy as np

from base_pricer import PricerBase
from product_definitions import (
    QuantLibBondStaticBase, CallableBondStaticBase, ConvertibleBondStaticBase
)

class QuantLibBondPricer(PricerBase):
    def __init__(self, bond_static: QuantLibBondStaticBase, method: str = 'discount',
                 grid_steps: int = 100, convertible_engine_steps: int = 256):
        if not isinstance(bond_static, QuantLibBondStaticBase):
            raise TypeError("Requires QuantLibBondStaticBase derivative.")
        super().__init__(bond_static)
        self.method = method.lower()
        self.grid_steps = grid_steps
        self.convertible_engine_steps = convertible_engine_steps
        self.is_callable = isinstance(bond_static, CallableBondStaticBase)
        self.is_convertible = isinstance(bond_static, ConvertibleBondStaticBase)

    @staticmethod
    def _build_ql_dates(product_static_obj, base_d: ql.Date, pillar_times) -> ql.DateVector:
        dates = ql.DateVector()
        # ensure pillar_times is a numpy array so .size works
        import numpy as _np
        if not hasattr(pillar_times, "size"):
            pillar_times = _np.array(pillar_times, dtype=float)
        # If pillar_times is empty or first pillar is > 0, add base_d (t=0 point)
        if pillar_times.size == 0 or (pillar_times.size > 0 and pillar_times[0] > 1e-6):
            dates.push_back(base_d)
        for t_val in pillar_times:
            # Ensure t_val is treated as days for Period construction if it's a year fraction
            # ql.Period expects an integer number of units (Days, Weeks, Months, Years)
            # A common convention is to convert year fractions to days.
            days_to_add = int(round(t_val * 365.0)) # Approximate days
            dates.push_back(base_d + ql.Period(days_to_add, ql.Days))
        return dates

    @staticmethod
    def _align_rates(product_static_obj, ql_dates: ql.DateVector, rates_vec: np.ndarray) -> list[float]:
        eff_rates = list(rates_vec)
        # If ql_dates includes t=0 but rates_vec doesn't, prepend first rate
        if len(ql_dates) == len(rates_vec) + 1 and ql_dates[0] == product_static_obj.ql_valuation_date:
            if rates_vec.size > 0:
                eff_rates.insert(0, rates_vec[0]) # Use first rate for t=0
            else: # rates_vec is empty, but ql_dates has t=0
                eff_rates.insert(0, 0.0) # Default to 0 if no rates provided
        elif len(ql_dates) != len(rates_vec):
            # This is a more general mismatch, log a warning
            # print(f"Warning: Date vector length {len(ql_dates)} and rate vector length {len(eff_rates)} mismatch. This might lead to issues.")
            # Attempt to pad or truncate, but this is risky. Best if inputs are aligned.
            if len(eff_rates) < len(ql_dates): eff_rates.extend([eff_rates[-1] if eff_rates else 0.0] * (len(ql_dates) - len(eff_rates)))
            else: eff_rates = eff_rates[:len(ql_dates)]
        return eff_rates


    def _make_term_structure(self,
                             pillar_times_rf: np.ndarray,
                             rates_vec_rf: np.ndarray,
                             pillar_times_cs: np.ndarray = None,
                             spreads_vec_cs: np.ndarray = None
                             ) -> ql.YieldTermStructureHandle:
        base_d: ql.Date = self.product_static.ql_valuation_date
        cal = self.product_static.calendar_ql if hasattr(self.product_static, 'calendar_ql') else ql.TARGET()
        dc = self.product_static.day_count_ql if hasattr(self.product_static, 'day_count_ql') else ql.ActualActual(ql.ActualActual.ISDA)

        dates_rf = self._build_ql_dates(self.product_static, base_d, pillar_times_rf)
        eff_rates_rf = self._align_rates(self.product_static, dates_rf, rates_vec_rf)

        base_curve = ql.ZeroCurve(dates_rf, eff_rates_rf, dc, cal, ql.Linear(), ql.Continuous, ql.Annual)
        base_curve.enableExtrapolation()

        if hasattr(self.product_static, 'credit_spread_curve_name') and \
           self.product_static.credit_spread_curve_name and \
           pillar_times_cs is not None and spreads_vec_cs is not None and spreads_vec_cs.size > 0:

            # Add spreads to the risk-free zero rates to get the discount curve
            if not np.array_equal(pillar_times_rf, pillar_times_cs):
                # Interpolate spreads to RF pillar times if they are different
                interp_spreads = np.interp(pillar_times_rf, pillar_times_cs, spreads_vec_cs)
                combined_rates = rates_vec_rf + interp_spreads
            else:
                combined_rates = rates_vec_rf + spreads_vec_cs
            
            # Align combined_rates with dates_rf (which might include t=0)
            eff_combined_rates = self._align_rates(self.product_static, dates_rf, combined_rates)

            final_curve = ql.ZeroCurve(dates_rf, eff_combined_rates, dc, cal, ql.Linear(), ql.Continuous, ql.Annual)
            final_curve.enableExtrapolation()
            return ql.YieldTermStructureHandle(final_curve)

        return ql.YieldTermStructureHandle(base_curve)

    @staticmethod
    def _price_vanilla_static(bond: ql.Bond, ts_handle: ql.YieldTermStructureHandle) -> float:
        eng = ql.DiscountingBondEngine(ts_handle); bond.setPricingEngine(eng); return bond.NPV()
    @staticmethod
    def _price_callable_static(bond: ql.CallableFixedRateBond, ts_handle: ql.YieldTermStructureHandle,
                               params: tuple, steps: int) -> float:
        if params is None: raise ValueError("G2 model parameters must be provided for callable bond pricing.")
        a,sig,b,eta,rho=params; model=ql.G2(ts_handle,a,sig,b,eta,rho)
        eng=ql.TreeCallableFixedRateBondEngine(model,steps); bond.setPricingEngine(eng); return bond.cleanPrice() # QL returns clean price
    
    @staticmethod
    def _price_convertible_static(
        bond: ql.ConvertibleFixedCouponBond,
        discounting_ts_handle: ql.YieldTermStructureHandle, # For discounting bond cashflows
        equity_ts_handle: ql.YieldTermStructureHandle,      # For equity process forward curve
        static_def: ConvertibleBondStaticBase, eng_steps: int,
        s0: float, div_yield: float, eq_vol: float, credit_spread_for_engine: float # Market params
    ) -> float:
        eval_d = ql.Settings.instance().evaluationDate
        s0_h = ql.QuoteHandle(ql.SimpleQuote(s0))
        day_count = static_def.day_count_ql # Use bond's day count for consistency
        calendar = static_def.calendar_ql

        # Dividend yield term structure
        div_h = ql.YieldTermStructureHandle(ql.FlatForward(eval_d, div_yield, day_count))
        # Equity volatility term structure
        vol_h = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(eval_d, calendar, eq_vol, day_count))
        # Credit spread for the binomial engine (hazard rate approach)
        cs_h = ql.QuoteHandle(ql.SimpleQuote(credit_spread_for_engine))
        # Geometric Brownian Motion process for the underlying stock
        proc = ql.BlackScholesMertonProcess(s0_h, div_h, equity_ts_handle, vol_h)
        # Binomial engine for convertible bonds
        engine = ql.BinomialCRRConvertibleEngine(proc, eng_steps, cs_h)
        
        bond.setPricingEngine(engine)
        return bond.NPV()


    def price(self, pillar_times: np.ndarray, market_scenario_data: np.ndarray,
              credit_spread_pillar_times: np.ndarray = None,
              **kwargs) -> np.ndarray:
        if market_scenario_data.ndim == 1: market_scenario_data = market_scenario_data.reshape(1, -1)
        prices = []
        is_g2_single = self.method=='g2' and (isinstance(kwargs.get('g2_params'), tuple) or (isinstance(kwargs.get('g2_params'), list) and kwargs.get('g2_params') and isinstance(kwargs.get('g2_params')[0], (float,int))))
        if self.method=='g2' and kwargs.get('g2_params') is not None and not is_g2_single and len(kwargs.get('g2_params')) != market_scenario_data.shape[0]:
            raise ValueError("List of g2_params must match number of scenarios for G2 method.")

        for i, scen_data_row in enumerate(market_scenario_data):
            current_g2_p = None
            if self.method == 'g2':
                 current_g2_p = kwargs.get('g2_params') if is_g2_single else (kwargs.get('g2_params')[i] if kwargs.get('g2_params') else None)

            prices.append(self._price_single_curve_logic(pillar_times, scen_data_row, current_g2_p,
                                                         credit_spread_pillar_times=credit_spread_pillar_times,
                                                         **kwargs))
        return np.array(prices)

    def _price_single_curve_logic(self,
                                  pillar_times_np_rf: np.ndarray,
                                  market_data_for_scenario_row: np.ndarray,
                                  g2_p_for_this_scen=None,
                                  credit_spread_pillar_times: np.ndarray = None,
                                  **other_fixed_params) -> float:
        if self.product_static and hasattr(self.product_static, 'ql_valuation_date'):
             ql.Settings.instance().evaluationDate = self.product_static.ql_valuation_date
        else: # Fallback, should ideally not be needed if product_static is always well-formed
             today = ql.Date().todaysDate(); ql.Settings.instance().evaluationDate = today

        num_rf_pillars = len(pillar_times_np_rf)
        rates_scen_rf = market_data_for_scenario_row[:num_rf_pillars]
        current_scen_idx = num_rf_pillars

        spreads_scen_cs = None
        cs_pillar_times_to_use = credit_spread_pillar_times

        if hasattr(self.product_static, 'credit_spread_curve_name') and \
           self.product_static.credit_spread_curve_name and \
           cs_pillar_times_to_use is not None and cs_pillar_times_to_use.size > 0:

            num_cs_pillars = len(cs_pillar_times_to_use)
            if len(market_data_for_scenario_row) >= current_scen_idx + num_cs_pillars:
                spreads_scen_cs = market_data_for_scenario_row[current_scen_idx : current_scen_idx + num_cs_pillars]
                current_scen_idx += num_cs_pillars
            else: # Not enough data for credit spreads, so ignore them for this scenario
                cs_pillar_times_to_use = None

        discounting_ts_handle = self._make_term_structure(pillar_times_np_rf, rates_scen_rf,
                                                          cs_pillar_times_to_use, spreads_scen_cs)
        # For convertibles, the equity process also needs a risk-free curve (without credit spread of bond issuer)
        equity_process_ts_handle = self._make_term_structure(pillar_times_np_rf, rates_scen_rf)

        if self.is_convertible:
            if not isinstance(self.product_static, ConvertibleBondStaticBase):
                raise TypeError("Product is not ConvertibleBondStaticBase for CB pricing.")

            num_dynamic_equity_related_factors = len(market_data_for_scenario_row) - current_scen_idx
            s0_val = other_fixed_params.get('s0_val')
            div_val = other_fixed_params.get('dividend_yield')
            eq_vol_val = other_fixed_params.get('equity_volatility')
            cs_for_engine_val = other_fixed_params.get('credit_spread') 

            if num_dynamic_equity_related_factors >= 1:
                s0_val = market_data_for_scenario_row[current_scen_idx]; current_scen_idx +=1
            if num_dynamic_equity_related_factors >= 2:
                div_val = market_data_for_scenario_row[current_scen_idx]; current_scen_idx +=1
            if num_dynamic_equity_related_factors >= 3:
                eq_vol_val = market_data_for_scenario_row[current_scen_idx]; current_scen_idx +=1
            if num_dynamic_equity_related_factors >= 4: 
                cs_for_engine_val = market_data_for_scenario_row[current_scen_idx]

            if s0_val is None: raise ValueError("S0 value missing for CB pricing.")
            if div_val is None: raise ValueError("Dividend yield missing for CB pricing.")
            if eq_vol_val is None: raise ValueError("Equity volatility missing for CB pricing.")
            if cs_for_engine_val is None: raise ValueError("Credit spread (for engine) missing for CB pricing.")

            return self._price_convertible_static(
                self.product_static.bond, discounting_ts_handle, equity_process_ts_handle,
                self.product_static, self.convertible_engine_steps,
                s0_val, div_val, eq_vol_val, cs_for_engine_val)

        elif self.is_callable and self.method == 'g2':
            if g2_p_for_this_scen is None: raise ValueError("g2_params needed for G2 pricing.")
            return self._price_callable_static(self.product_static.bond, discounting_ts_handle, g2_p_for_this_scen, self.grid_steps)
        elif self.method == 'discount': 
            return self._price_vanilla_static(self.product_static.bond, discounting_ts_handle)
        raise ValueError(f"Unsupported method '{self.method}' or product configuration for QL BondPricer.")
    
    def price_scenarios(
        self,
        raw_market_scenarios: np.ndarray,
        scenario_factor_names: list[str],
        rate_pillars: np.ndarray | None = None,
        **price_kwargs
    ) -> np.ndarray:
        if rate_pillars is None:
            raise ValueError("rate_pillars must be provided for QuantLibBondPricer.")
        n = len(rate_pillars)
        try:
            # Attempt to find indices of rate_pillars in scenario_factor_names, with currency conversion to str
            # This assumes rate_pillars are in the same format as scenario_factor_names
            # in the form ccy_subindex_0.5Y, ccy_subindex_1Y, etc.
            rate_pillars_name = [f"{self.product_static.currency}_{self.product_static.index_stub}_{t:.2f}Y" for t in rate_pillars]
            # Find indices of rate_pillars in scenario_factor_names
            idx = [scenario_factor_names.index(str(t)) for t in rate_pillars_name]
            if len(idx) != n:
                idx = list(range(n))
        except Exception:
            idx = list(range(n))
        data = raw_market_scenarios[:, idx]
        if self.is_convertible:
            # For convertible bonds, we need to append S0, dividend yield, equity volatility, and credit spread
            # look at the instrument static definition to find the S0 factor name
            if not hasattr(self.product_static, 'underlying_symbol'):
                raise ValueError("Convertible bond static definition must have an underlying symbol for S0 factor.")
            underlying_symbol = self.product_static.underlying_symbol
            ccy = self.product_static.currency
            s0_factor_name = f"{ccy}_{underlying_symbol}_S0"
            # Check if S0 factor name exists in scenario_factor_names
            if s0_factor_name not in scenario_factor_names:
                raise ValueError(f"S0 factor '{s0_factor_name}' not found in scenario factor names.")
            i = scenario_factor_names.index(s0_factor_name)
            data = np.hstack((data, raw_market_scenarios[:, i:i+1]))
            
        return self.price(
            pillar_times=rate_pillars,
            market_scenario_data=data,
            **price_kwargs
        )