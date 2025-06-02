"""
Contains pricer classes for different financial products.
Pricers take a static product definition and market data to calculate a price.
Convertible bond pricer now correctly handles dynamic vs. fixed market parameters.
QuantLibBondPricer can now handle an optional credit spread curve.
Added MBSPricer for Phase 1 (simple prepayment models).
"""
import QuantLib as ql
import numpy as np
from scipy.stats import norm
import abc
from datetime import date
from dateutil.relativedelta import relativedelta # For MBSPricer

# Imports from other modules in this project
from product_definitions import (
    ProductStaticBase, QuantLibBondStaticBase, CallableBondStaticBase,
    ConvertibleBondStaticBase, EuropeanOptionStatic, MBSPoolStatic
)
from prepayment_models import (
    PrepaymentModelBase, ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel
)


class PricerBase(abc.ABC):
    """
    Abstract base class for all pricers.
    """
    def __init__(self, product_static: ProductStaticBase):
        self.product_static: ProductStaticBase = product_static
        # Common QL settings setup, can be overridden if needed by specific pricers
        if hasattr(product_static, 'ql_valuation_date'):
            ql.Settings.instance().evaluationDate = product_static.ql_valuation_date
        elif hasattr(product_static, 'valuation_date_py'):
             eval_date = product_static.valuation_date_py
             ql.Settings.instance().evaluationDate = ql.Date(eval_date.day, eval_date.month, eval_date.year)


    @abc.abstractmethod
    def price(self, **kwargs) -> np.ndarray:
        pass

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


class QuantLibBondPricer(PricerBase):
    def __init__(self, bond_static: QuantLibBondStaticBase, method: str = 'discount',
                 grid_steps: int = 100, convertible_engine_steps: int = 100):
        if not isinstance(bond_static, QuantLibBondStaticBase):
            raise TypeError("Requires QuantLibBondStaticBase derivative.")
        super().__init__(bond_static)
        self.method, self.grid_steps, self.convertible_engine_steps = method.lower(), grid_steps, convertible_engine_steps
        self.is_callable = isinstance(bond_static, CallableBondStaticBase)
        self.is_convertible = isinstance(bond_static, ConvertibleBondStaticBase)

    @staticmethod
    def _build_ql_dates(product_static_obj, base_d: ql.Date, pillar_times: np.ndarray) -> ql.DateVector:
        dates = ql.DateVector()
        # If pillar_times is empty or first pillar is > 0, add base_d (t=0 point)
        if not pillar_times.size or (pillar_times.size > 0 and pillar_times[0] > 1e-6):
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
        # The engine uses the discounting_ts_handle for bond cashflows and cs_h for default probability
        # engine = ql.BinomialConvertibleEngine(proc, "crr", eng_steps, cs_h, discounting_ts_handle)
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


        if self.is_convertible and self.method == 'convertible_binomial':
            if not isinstance(self.product_static, ConvertibleBondStaticBase):
                raise TypeError("Product is not ConvertibleBondStaticBase for CB pricing.")

            # Determine how many equity-related factors are dynamic (from market_data_for_scenario_row)
            # vs. fixed (from other_fixed_params)
            num_dynamic_equity_related_factors = len(market_data_for_scenario_row) - current_scen_idx

            # Start with fixed params, then override if dynamic ones are available
            s0_val = other_fixed_params.get('s0_val')
            div_val = other_fixed_params.get('dividend_yield')
            eq_vol_val = other_fixed_params.get('equity_volatility')
            cs_for_engine_val = other_fixed_params.get('credit_spread') # This is the credit spread for the binomial engine

            # Override with dynamic values if present in the scenario data
            if num_dynamic_equity_related_factors >= 1:
                s0_val = market_data_for_scenario_row[current_scen_idx]; current_scen_idx +=1
            if num_dynamic_equity_related_factors >= 2:
                div_val = market_data_for_scenario_row[current_scen_idx]; current_scen_idx +=1
            if num_dynamic_equity_related_factors >= 3:
                eq_vol_val = market_data_for_scenario_row[current_scen_idx]; current_scen_idx +=1
            if num_dynamic_equity_related_factors >= 4: # Assuming order: S0, Div, Vol, CS_engine
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
        elif self.method == 'discount': # For vanilla bonds or non-G2 callable pricing (if simplified)
            return self._price_vanilla_static(self.product_static.bond, discounting_ts_handle)
        raise ValueError(f"Unsupported method '{self.method}' or product configuration for QL BondPricer.")


class BlackScholesPricer(PricerBase):
    def __init__(self, option_static: EuropeanOptionStatic,
                 risk_free_rate: float, dividend_yield: float = 0.0):
        if not isinstance(option_static, EuropeanOptionStatic):
            raise TypeError("Requires EuropeanOptionStatic.")
        super().__init__(option_static)
        if risk_free_rate is None: raise ValueError("'risk_free_rate' cannot be None for BlackScholesPricer")
        self.risk_free_rate, self.dividend_yield = risk_free_rate, dividend_yield

    def price(self, stock_price: np.ndarray, volatility: np.ndarray, **kwargs) -> np.ndarray:
        S_in, sig_in = np.asarray(stock_price), np.asarray(volatility)
        opt_static: EuropeanOptionStatic = self.product_static
        K, T = opt_static.strike_price, opt_static.time_to_expiry
        r, q = self.risk_free_rate, self.dividend_yield
        option_type = opt_static.option_type.lower()

        # Handle scalar inputs by converting to 1-element arrays for consistent processing
        S = np.atleast_1d(S_in)
        sig = np.atleast_1d(sig_in)

        # If one is scalar and other is array, broadcast scalar to match array shape
        if S.ndim == 1 and S.shape[0] == 1 and sig.ndim == 1 and sig.shape[0] > 1:
            S = np.full_like(sig, S[0])
        elif sig.ndim == 1 and sig.shape[0] == 1 and S.ndim == 1 and S.shape[0] > 1:
            sig = np.full_like(S, sig[0])
        elif S.shape != sig.shape:
            raise ValueError("Stock price and volatility arrays must have the same shape or be broadcastable.")

        prices = np.zeros_like(S, dtype=float)

        # Handle expired or zero-time options
        if T <= 1e-9:
            if option_type == 'call':
                prices = np.maximum(S - K, 0.0)
            else: # put
                prices = np.maximum(K - S, 0.0)
            return prices[0] if S_in.ndim == 0 and sig_in.ndim == 0 else prices

        # Avoid issues with zero volatility or stock price
        # For very small sig, d1/d2 can blow up. Use intrinsic value.
        # For S <= 0, call is 0, put is K*exp(-rT)
        
        # Create masks for different conditions
        valid_mask = (sig > 1e-9) & (S > 1e-9) # Standard BS calculation
        zero_vol_mask = (sig <= 1e-9) & (S > 1e-9) # Zero volatility but positive stock price
        zero_stock_mask = (S <= 1e-9) # Zero or negative stock price

        # --- Standard Black-Scholes calculation for valid_mask ---
        if np.any(valid_mask):
            S_v, sig_v = S[valid_mask], sig[valid_mask]
            d1 = (np.log(S_v / K) + (r - q + 0.5 * sig_v**2) * T) / (sig_v * np.sqrt(T))
            d2 = d1 - sig_v * np.sqrt(T)
            if option_type == 'call':
                prices[valid_mask] = (S_v * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
            else: # put
                prices[valid_mask] = (K * np.exp(-r * T) * norm.cdf(-d2) - S_v * np.exp(-q * T) * norm.cdf(-d1))

        # --- Zero volatility case (intrinsic value discounted) ---
        if np.any(zero_vol_mask):
            S_zv = S[zero_vol_mask]
            if option_type == 'call':
                prices[zero_vol_mask] = np.maximum(S_zv * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
            else: # put
                prices[zero_vol_mask] = np.maximum(K * np.exp(-r * T) - S_zv * np.exp(-q * T), 0.0)
        
        # --- Zero stock price case ---
        if np.any(zero_stock_mask):
            if option_type == 'call':
                prices[zero_stock_mask] = 0.0
            else: # put
                prices[zero_stock_mask] = K * np.exp(-r * T)
        
        prices = np.maximum(prices, 0.0) # Ensure non-negativity

        return prices[0] if S_in.ndim == 0 and sig_in.ndim == 0 else prices


class MBSPricer(PricerBase):
    def __init__(self, mbs_static: MBSPoolStatic, prepayment_model: PrepaymentModelBase):
        if not isinstance(mbs_static, MBSPoolStatic):
            raise TypeError("MBSPricer requires an MBSPoolStatic instance.")
        if not isinstance(prepayment_model, PrepaymentModelBase): # Check against PrepaymentModelBase
            raise TypeError("MBSPricer requires a PrepaymentModelBase instance.")
        super().__init__(mbs_static)
        self.mbs_static: MBSPoolStatic = mbs_static # For type hinting
        self.prepayment_model = prepayment_model

    def _calculate_scheduled_payment_and_principal(self, balance, wac_monthly, remaining_periods):
        """Calculates monthly scheduled payment and principal portion."""
        if wac_monthly < 1e-9: # Effectively zero interest rate
            if remaining_periods == 0: return 0.0, 0.0
            payment = balance / remaining_periods if remaining_periods > 0 else 0.0
            return payment, payment # All payment is principal

        if remaining_periods <=0 : return 0.0, 0.0

        # Standard mortgage payment formula: M = P * [r(1+r)^n] / [(1+r)^n – 1]
        payment = balance * (wac_monthly * (1 + wac_monthly)**remaining_periods) / \
                  ((1 + wac_monthly)**remaining_periods - 1)
        interest_portion = balance * wac_monthly
        scheduled_principal = payment - interest_portion
        return payment, scheduled_principal


    def price(self, pillar_times_rf: np.ndarray, market_scenario_data: np.ndarray,
              credit_spread_pillar_times: np.ndarray = None,
              fixed_market_mortgage_rate_for_prepay: float = None, # For simple prepayment models
              **kwargs) -> np.ndarray:

        if market_scenario_data.ndim == 1:
            market_scenario_data = market_scenario_data.reshape(1, -1)

        num_scenarios = market_scenario_data.shape[0]
        npvs = np.zeros(num_scenarios)

        eval_date_py = self.mbs_static.valuation_date_py
        eval_date_ql = ql.Date(eval_date_py.day, eval_date_py.month, eval_date_py.year)
        
        # Use a common day counter for discounting, e.g., Actual/Actual ISDA
        dc_discount = ql.ActualActual(ql.ActualActual.ISDA)
        calendar_discount = ql.TARGET() # A common calendar

        for i in range(num_scenarios):
            scen_data_row = market_scenario_data[i, :]
            ql.Settings.instance().evaluationDate = eval_date_ql # Ensure eval date is set for each scenario context

            # Construct discount curve for this scenario
            num_rf_pillars = len(pillar_times_rf)
            rates_scen_rf = scen_data_row[:num_rf_pillars]
            current_scen_idx = num_rf_pillars

            spreads_scen_cs = None
            cs_pillar_times_to_use = credit_spread_pillar_times

            if hasattr(self.mbs_static, 'credit_spread_curve_name') and \
               self.mbs_static.credit_spread_curve_name and \
               cs_pillar_times_to_use is not None and cs_pillar_times_to_use.size > 0:
                num_cs_pillars = len(cs_pillar_times_to_use)
                if len(scen_data_row) >= current_scen_idx + num_cs_pillars:
                    spreads_scen_cs = scen_data_row[current_scen_idx : current_scen_idx + num_cs_pillars]
                else: cs_pillar_times_to_use = None # Not enough data

            # Build discount curve using QuantLibBondPricer's static helper methods
            # Need to pass a dummy product_static or self.mbs_static if methods are not static
            # For simplicity, let's assume QuantLibBondPricer._build_ql_dates and _align_rates can be adapted or made static
            # Or, we replicate simplified logic here.
            
            # Simplified curve construction for MBSPricer
            dates_rf_scen = QuantLibBondPricer._build_ql_dates(self.mbs_static, eval_date_ql, pillar_times_rf)
            eff_rates_rf_scen = QuantLibBondPricer._align_rates(self.mbs_static, dates_rf_scen, rates_scen_rf)
            
            final_rates_for_curve = eff_rates_rf_scen
            if cs_pillar_times_to_use is not None and spreads_scen_cs is not None:
                # Align spreads to RF pillars if necessary
                if not np.array_equal(pillar_times_rf, cs_pillar_times_to_use):
                    interp_spreads = np.interp(pillar_times_rf, cs_pillar_times_to_use, spreads_scen_cs)
                else:
                    interp_spreads = spreads_scen_cs
                
                # Add spreads to RF rates (assuming rates_scen_rf are the pillar values, not yet aligned with t=0)
                combined_pillar_rates = rates_scen_rf + interp_spreads
                final_rates_for_curve = QuantLibBondPricer._align_rates(self.mbs_static, dates_rf_scen, combined_pillar_rates)

            discount_curve_scen = ql.ZeroCurve(dates_rf_scen, final_rates_for_curve, dc_discount, calendar_discount, ql.Linear(), ql.Continuous, ql.Annual)
            discount_curve_scen.enableExtrapolation()
            discounting_ts_handle_scen = ql.YieldTermStructureHandle(discount_curve_scen)


            # MBS Cash Flow Generation
            current_balance = self.mbs_static.current_balance
            total_npv_for_scenario = 0.0
            wac_monthly = self.mbs_static.wac / 12.0
            pass_through_monthly = self.mbs_static.pass_through_rate / 12.0

            for month_idx in range(1, self.mbs_static.remaining_term_months + 1):
                if current_balance < 1e-2: break # Pool paid down

                age_at_period_start = self.mbs_static.age_months + month_idx -1 # Current age for SMM calc
                remaining_periods_at_period_start = self.mbs_static.original_term_months - age_at_period_start
                if remaining_periods_at_period_start <= 0: break

                interest_accrued_on_wac = current_balance * wac_monthly

                # Scheduled Principal
                _, scheduled_principal = self._calculate_scheduled_payment_and_principal(
                    current_balance, wac_monthly, remaining_periods_at_period_start
                )
                scheduled_principal = min(scheduled_principal, current_balance) # Cannot pay more than balance

                # Prepayment
                period_info = {'age_months': age_at_period_start, 'wac_pool': self.mbs_static.wac}
                if isinstance(self.prepayment_model, RefiIncentivePrepaymentModel):
                    cm_proxy = fixed_market_mortgage_rate_for_prepay
                    if cm_proxy is None: # Fallback if not provided: use a point from the discount curve
                         # Example: 10Y zero rate as proxy for current mortgage rate. This is a simplification.
                        cm_proxy = discounting_ts_handle_scen.zeroRate(10.0, ql.Continuous, ql.Annual).rate()
                    period_info['current_market_mortgage_rate_cm'] = cm_proxy
                
                smm = self.prepayment_model.get_smm(period_info)
                prepayment_amount = (current_balance - scheduled_principal) * smm # Prepay from remaining balance after scheduled principal
                prepayment_amount = min(prepayment_amount, current_balance - scheduled_principal) # Clamp

                total_principal_paid = scheduled_principal + prepayment_amount

                # Interest passed to investor (based on pass-through rate and balance at start of month)
                interest_passed_through = current_balance * pass_through_monthly

                total_investor_cashflow = interest_passed_through + total_principal_paid

                # Discounting: Payment date includes delay
                # Cashflow occurs at end of month_idx, then delayed
                payment_date_actual = eval_date_py + relativedelta(months=month_idx, days=self.mbs_static.delay_days)
                payment_date_ql = ql.Date(payment_date_actual.day, payment_date_actual.month, payment_date_actual.year)
                
                df = discounting_ts_handle_scen.discount(payment_date_ql)

                total_npv_for_scenario += total_investor_cashflow * df
                current_balance -= total_principal_paid

            npvs[i] = total_npv_for_scenario
        return npvs
