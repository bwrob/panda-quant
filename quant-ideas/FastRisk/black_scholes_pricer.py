import numpy as np
from scipy.stats import norm
import QuantLib as ql
from base_pricer import PricerBase
from product_definitions import EuropeanOptionStatic

class BlackScholesPricer(PricerBase):
    def __init__(self, option_static: EuropeanOptionStatic):
        if not isinstance(option_static, EuropeanOptionStatic):
            raise TypeError("Requires EuropeanOptionStatic.")
        super().__init__(option_static)

    def price(self,
              stock_price: np.ndarray,
              volatility: np.ndarray,
              risk_free_rate: float,
              dividend_yield: float = 0.0,
              **kwargs
    ) -> np.ndarray:
        S_in, sig_in = np.asarray(stock_price), np.asarray(volatility)
        opt_static: EuropeanOptionStatic = self.product_static
        K, T = opt_static.strike_price, opt_static.time_to_expiry
        r, q = risk_free_rate, dividend_yield
        option_type = opt_static.option_type.lower()

        S = np.atleast_1d(S_in)
        sig = np.atleast_1d(sig_in)

        if S.ndim == 1 and S.shape[0] == 1 and sig.ndim == 1 and sig.shape[0] > 1:
            S = np.full_like(sig, S[0])
        elif sig.ndim == 1 and sig.shape[0] == 1 and S.ndim == 1 and S.shape[0] > 1:
            sig = np.full_like(S, sig[0])
        elif S.shape != sig.shape:
            raise ValueError("Stock price and volatility arrays must have the same shape or be broadcastable.")

        prices = np.zeros_like(S, dtype=float)

        if T <= 1e-9:
            if option_type == 'call':
                prices = np.maximum(S - K, 0.0)
            else: # put
                prices = np.maximum(K - S, 0.0)
            return prices[0] if S_in.ndim == 0 and sig_in.ndim == 0 else prices
        
        valid_mask = (sig > 1e-9) & (S > 1e-9) 
        zero_vol_mask = (sig <= 1e-9) & (S > 1e-9) 
        zero_stock_mask = (S <= 1e-9)

        if np.any(valid_mask):
            S_v, sig_v = S[valid_mask], sig[valid_mask]
            d1 = (np.log(S_v / K) + (r - q + 0.5 * sig_v**2) * T) / (sig_v * np.sqrt(T))
            d2 = d1 - sig_v * np.sqrt(T)
            if option_type == 'call':
                prices[valid_mask] = (S_v * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
            else: # put
                prices[valid_mask] = (K * np.exp(-r * T) * norm.cdf(-d2) - S_v * np.exp(-q * T) * norm.cdf(-d1))

        if np.any(zero_vol_mask):
            S_zv = S[zero_vol_mask]
            if option_type == 'call':
                prices[zero_vol_mask] = np.maximum(S_zv * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
            else: # put
                prices[zero_vol_mask] = np.maximum(K * np.exp(-r * T) - S_zv * np.exp(-q * T), 0.0)
        
        if np.any(zero_stock_mask):
            if option_type == 'call':
                prices[zero_stock_mask] = 0.0
            else: # put
                prices[zero_stock_mask] = K * np.exp(-r * T)
        
        prices = np.maximum(prices, 0.0) 

        return prices[0] if S_in.ndim == 0 and sig_in.ndim == 0 else prices
    
    def price_scenarios(
        self,
        raw_market_scenarios: np.ndarray,
        scenario_factor_names: list[str],
        rate_pillars: np.ndarray | None = None,
        **price_kwargs
    ) -> np.ndarray:
        prod = self.product_static
        key_s0  = f"{prod.currency}_{prod.underlying_symbol}_S0"
        key_vol = f"{prod.currency}_{prod.underlying_symbol}_VOL"
        if key_s0 not in scenario_factor_names or key_vol not in scenario_factor_names:
            raise ValueError(f"Could not locate S0/Vol factors for '{prod.underlying_symbol}'.")
        iS = scenario_factor_names.index(key_s0)
        iV = scenario_factor_names.index(key_vol)
        S = raw_market_scenarios[:, iS]
        V = raw_market_scenarios[:, iV]
        return self.price(
            stock_price=S,
            volatility=V,
            **price_kwargs
        )