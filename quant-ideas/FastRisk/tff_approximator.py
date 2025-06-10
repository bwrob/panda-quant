"""
Contains classes and functions for Tensor Functional Form (TFF) approximation.
TensorFunctionalFormCalibrate.__init__ is simplified.
Worker function (_price_one_scenario_for_tff) now handles MBSPoolStatic.
"""
import numpy as np
from scipy.stats.qmc import LatinHypercube, Sobol, scale
from concurrent.futures import ProcessPoolExecutor
import QuantLib as ql
from datetime import date, datetime
import re
import os
import warnings

from product_definitions import (
    ProductStaticBase, QuantLibBondStaticBase, CallableBondStaticBase,
    ConvertibleBondStaticBase, EuropeanOptionStatic, MBSPoolStatic,
    _parse_date_input # For worker
)
# Updated pricer imports
from base_pricer import PricerBase
from quantlib_bond_pricer import QuantLibBondPricer
from black_scholes_pricer import BlackScholesPricer
from mbs_pricer import MBSPricer

from prepayment_models import ( # For worker with MBS
    ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel
)

from features_generator import FeatureGenerator, engineer_option_features, normalize_features

# --- Feature Engineering and Normalization for Options ---
def _parse_numeric_pillars_from_factor_names(factor_names: list[str]) -> np.ndarray:
    parsed_pillars = []
    for name_str in factor_names:
        match = re.search(r'(\d+(\.\d+)?)(?=Y)', name_str)
        if not match: match = re.search(r'(\d+(\.\d+)?)', name_str)
        if match:
            try:
                # Only parse if it looks like a rate or spread curve factor name
                if (any(sub.upper() in name_str.upper() for sub in ["RATE", "IR", "CURVE", "YIELD", "_CS", "_SPREAD"]) and "Y" in name_str.upper()):
                    parsed_pillars.append(float(match.group(1)))
            except ValueError: pass
        else:
            try:
                if not any(equity_tag in name_str.upper() for equity_tag in ["S0", "VOL", "EQUITY", "STOCK", "DIVYIELD"]): # Exclude single value equity factors
                    # This path is less likely with explicit factor naming
                    parsed_pillars.append(float(name_str))
            except ValueError: pass
    return np.array(sorted(list(set(parsed_pillars))), dtype=float) if parsed_pillars else np.array([], dtype=float)


class TensorFunctionalForm:
    def __init__(self, A: np.ndarray, b: np.ndarray, c: float):
        self.A, self.b, self.c = np.asarray(A,float), np.asarray(b,float), float(c)
        if self.A.ndim!=2 or self.A.shape[0]!=self.A.shape[1]: raise ValueError("A must be square.")
        if self.b.ndim!=1 or self.b.shape[0]!=self.A.shape[0]: raise ValueError("b dim must match A.")
        self.D: int = self.A.shape[0]
    def __call__(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.asarray(x)
        if x_arr.ndim == 1:
            if x_arr.shape[0]!=self.D: raise ValueError(f"Input dim {x_arr.shape[0]} != model dim {self.D}.")
            return float(x_arr @ self.A @ x_arr + self.b @ x_arr + self.c)
        elif x_arr.ndim == 2:
            if x_arr.shape[1]!=self.D: raise ValueError(f"Input shape {x_arr.shape}, expected (N, {self.D}).")
            return np.sum((x_arr @ self.A) * x_arr, axis=1) + x_arr @ self.b + self.c
        raise ValueError(f"Input must be 1D or 2D, got ndim={x_arr.ndim}")
    def to_dict(self) -> dict: return {'A':self.A.tolist(),'b':self.b.tolist(),'c':self.c,'D':self.D}
    @classmethod
    def from_dict(cls, data:dict) -> 'TensorFunctionalForm':
        if not all(k in data for k in ['A','b','c']): raise ValueError("Missing keys in TFF data dict.")
        return cls(np.array(data['A'],float), np.array(data['b'],float), data['c'])


def _price_one_scenario_for_tff(worker_args: tuple) -> float:
    (product_static_params_dict, pricer_config_for_worker,
     factor_names_for_tff, single_market_scenario_data,
     valuation_date_for_worker_iso, price_kwargs_dict) = worker_args

    valuation_date_for_worker = _parse_date_input(valuation_date_for_worker_iso)
    ql_val_date = ql.Date(valuation_date_for_worker.day, valuation_date_for_worker.month, valuation_date_for_worker.year)
    ql.Settings.instance().evaluationDate = ql_val_date

    product_type = product_static_params_dict['product_type']
    actual_rate_pillars_for_worker = np.asarray(product_static_params_dict.get('actual_rate_pillars', []), dtype=float)

    current_static_params = product_static_params_dict.copy()
    current_static_params['valuation_date'] = valuation_date_for_worker

    if product_type == 'VanillaBond': product_static_obj = QuantLibBondStaticBase.from_dict(current_static_params)
    elif product_type == 'CallableBond': product_static_obj = CallableBondStaticBase.from_dict(current_static_params)
    elif product_type == 'ConvertibleBond': product_static_obj = ConvertibleBondStaticBase.from_dict(current_static_params)
    elif product_type == 'EuropeanOption': product_static_obj = EuropeanOptionStatic.from_dict(current_static_params)
    elif product_type == 'MBSPool': product_static_obj = MBSPoolStatic.from_dict(current_static_params) # NEW
    else: raise ValueError(f"Unknown product type for TFF worker: {product_type}")

    if product_type in ['VanillaBond', 'CallableBond', 'ConvertibleBond']:
        pricer_instance = QuantLibBondPricer(product_static_obj, **pricer_config_for_worker.get('bond_pricer_config',{}))
        market_data_for_ql_pricer = np.array([single_market_scenario_data])

        # For bonds with credit spread curves, pillar_times for credit spreads might be needed
        # This assumes TFFConfigurationFactory passes credit_spread_pillar_times if applicable
        credit_spread_pillars = price_kwargs_dict.pop('credit_spread_pillar_times', actual_rate_pillars_for_worker) # Default to RF pillars if not specified

        price_result_array = pricer_instance.price(
            pillar_times=actual_rate_pillars_for_worker,
            market_scenario_data=market_data_for_ql_pricer,
            credit_spread_pillar_times=credit_spread_pillars,
            **price_kwargs_dict
        )
        return price_result_array[0]

    elif product_type == 'MBSPool': # NEW
        mbs_static_obj: MBSPoolStatic = product_static_obj

        # Instantiate Prepayment Model based on static info and fixed params from price_kwargs_dict
        prepay_model_type = mbs_static_obj.prepayment_model_type
        prepay_rate_param = mbs_static_obj.prepayment_rate_param # CPR or PSA multiplier
        prepayment_model_instance = None

        if prepay_model_type == "CPR":
            prepayment_model_instance = ConstantCPRModel(prepay_rate_param)
        elif prepay_model_type == "PSA":
            prepayment_model_instance = PSAModel(prepay_rate_param)
        elif prepay_model_type == "RefiIncentive":
            # Coefficients for RefiIncentive model might be passed via price_kwargs_dict
            # or could be hardcoded/defaults in the RefiIncentivePrepaymentModel itself
            refi_A = price_kwargs_dict.get('refi_A')
            refi_B = price_kwargs_dict.get('refi_B')
            refi_C = price_kwargs_dict.get('refi_C')
            refi_D = price_kwargs_dict.get('refi_D')
            if all(p is not None for p in [refi_A, refi_B, refi_C, refi_D]):
                 prepayment_model_instance = RefiIncentivePrepaymentModel(refi_A, refi_B, refi_C, refi_D)
            else: # Use defaults if not all provided
                 prepayment_model_instance = RefiIncentivePrepaymentModel()
        else:
            raise ValueError(f"Unsupported prepayment_model_type: {prepay_model_type} for MBS TFF worker.")

        pricer_instance = MBSPricer(mbs_static_obj, prepayment_model=prepayment_model_instance)
        market_data_for_mbs_pricer = np.array([single_market_scenario_data])

        # The MBSPricer.price method will also need pillar_times_rf and potentially credit_spread_pillar_times
        # These are derived from factor_names_for_tff by TFFConfigurationFactory and stored in
        # product_static_params_for_worker['actual_rate_pillars'] (for RF)
        # and potentially a new field for credit spread pillars if they differ.
        # For now, assume actual_rate_pillars_for_worker covers all discount curve pillars.

        # fixed_market_mortgage_rate_for_prepay is expected in price_kwargs_dict if RefiIncentive is used
        # and its C_M is not derived from G2 model (Phase 1)

        price_result_array = pricer_instance.price(
            pillar_times_rf=actual_rate_pillars_for_worker, # Assuming these are RF pillars
            market_scenario_data=market_data_for_mbs_pricer,
            # credit_spread_pillar_times might be needed if different from RF pillars
            # and if credit spreads are dynamic TFF factors.
            # This part needs careful alignment with how TFFConfigurationFactory sets up factors.
            # For now, if credit spreads are dynamic, their pillars are assumed to match RF pillars.
            credit_spread_pillar_times=actual_rate_pillars_for_worker if mbs_static_obj.credit_spread_curve_name else None,
            **price_kwargs_dict
        )
        return price_result_array[0]

    elif product_type == 'EuropeanOption':
        bs_cfg = pricer_config_for_worker.get('bs_pricer_config', {})
        pricer_instance = BlackScholesPricer(product_static_obj)
        return pricer_instance.price(
            stock_price=single_market_scenario_data[0],
            volatility=single_market_scenario_data[1],
            risk_free_rate=bs_cfg['risk_free_rate'],
            dividend_yield=bs_cfg.get('dividend_yield', 0.0)
        )
    raise ValueError(f"Pricer path failed for product type: {product_type}")


class TensorFunctionalFormCalibrate:
    def __init__(
        self,
        pricer_template: PricerBase,
        tff_input_raw_factor_names: list[str],
        tff_input_raw_base_values: np.ndarray,
        product_static_params_for_worker: dict,
        pricer_config_for_worker: dict,
        feature_generation: FeatureGenerator = None,
        actual_rate_pillars: np.ndarray = None
    ):
        self.pricer_template = pricer_template
        self.product_static: ProductStaticBase = pricer_template.product_static

        self.tff_input_raw_factor_names = tff_input_raw_factor_names
        self.tff_input_raw_base_values = tff_input_raw_base_values

        self.product_static_params_for_worker = product_static_params_for_worker
        self.pricer_config_for_worker = pricer_config_for_worker
        self.actual_rate_pillars = actual_rate_pillars if actual_rate_pillars is not None else np.array([])

        val_date_from_params = self.product_static_params_for_worker.get('valuation_date')
        if isinstance(val_date_from_params, str):
            self.valuation_date_for_ql_settings_in_worker = date.fromisoformat(val_date_from_params)
        elif isinstance(val_date_from_params, date):
            self.valuation_date_for_ql_settings_in_worker = val_date_from_params
        elif self.product_static and hasattr(self.product_static, 'valuation_date_py'):
             self.valuation_date_for_ql_settings_in_worker = self.product_static.valuation_date_py
        else:
            raise TypeError("valuation_date in product_static_params_for_worker must be an ISO string or date object, or available on product_static.")


        self.product_type_str = self.product_static_params_for_worker.get('product_type')
        if not self.product_type_str:
            if isinstance(self.product_static, EuropeanOptionStatic): self.product_type_str = 'EuropeanOption'
            elif isinstance(self.product_static, CallableBondStaticBase): self.product_type_str = 'CallableBond'
            elif isinstance(self.product_static, ConvertibleBondStaticBase): self.product_type_str = 'ConvertibleBond'
            elif isinstance(self.product_static, MBSPoolStatic): self.product_type_str = 'MBSPool' # NEW
            elif isinstance(self.product_static, QuantLibBondStaticBase): self.product_type_str = 'VanillaBond'
            else: raise TypeError(f"Cannot determine product_type_str for TFFCalibrate from {type(self.product_static)}")

        if self.product_type_str in ['VanillaBond', 'CallableBond', 'ConvertibleBond', 'MBSPool']: # Added MBSPool
            if self.actual_rate_pillars.size == 0 and self.tff_input_raw_factor_names:
                 print(f"Warning: 'actual_rate_pillars' was empty for {self.product_type_str}. Parsing from TFF input names.")
                 self.actual_rate_pillars = _parse_numeric_pillars_from_factor_names(self.tff_input_raw_factor_names)
            self.product_static_params_for_worker['actual_rate_pillars'] = self.actual_rate_pillars.tolist()

        if not self.tff_input_raw_factor_names or self.tff_input_raw_base_values.size == 0:
            raise RuntimeError(f"TFF input factors/base values not set for {self.product_type_str}")
        if len(self.tff_input_raw_factor_names) != len(self.tff_input_raw_base_values):
            raise RuntimeError(f"Mismatch TFF factor names ({len(self.tff_input_raw_factor_names)}) and base values ({len(self.tff_input_raw_base_values)}).")

        if not feature_generation is None:
            if not isinstance(feature_generation, FeatureGenerator):
                raise TypeError(f"feature_generation must be a FeatureGenerator instance, got {type(feature_generation)}.")
            self.feature_generation = feature_generation
        else:
            self.feature_generation = None
            

    def sample_and_fit(
        self, full_market_scenarios_for_tff_factors: np.ndarray,
        n_train: int = 64, n_test: int = 8,
        random_seed: int = 0, sampling_method: str = 'sobol', parallel_workers: int = None,
        option_feature_order: int = 0, **price_kwargs
    ) -> tuple[TensorFunctionalForm, np.ndarray, np.ndarray, float, dict]:

        rng_np = np.random.default_rng(random_seed)
        num_tff_factors = len(self.tff_input_raw_factor_names)

        if full_market_scenarios_for_tff_factors.ndim != 2 or \
           full_market_scenarios_for_tff_factors.shape[1] != num_tff_factors:
            raise ValueError(f"Shape error for scenarios. Expected (N, {num_tff_factors}), got {full_market_scenarios_for_tff_factors.shape}. Factors: {self.tff_input_raw_factor_names}")

        domain_min, domain_max = np.min(full_market_scenarios_for_tff_factors, axis=0), np.max(full_market_scenarios_for_tff_factors, axis=0)

        train_tff_inputs_raw = None
        if sampling_method in ['sobol', 'uniform']:
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    category=UserWarning,
                    message=".*power of 2.*"
                )
                sampler = Sobol(d=num_tff_factors, scramble=True, seed=random_seed)
                train_tff_inputs_raw = scale(
                    sampler.random(n=n_train-1), domain_min, domain_max
                )
        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}. Available methods are 'sobol' and 'uniform'.")

        # Append the first row of full_market_scenarios_for_tff_factors to ensure base values are included
        if self.tff_input_raw_base_values.ndim == 1:
            if self.tff_input_raw_base_values.shape[0] != num_tff_factors:
                raise ValueError(f"Base values shape {self.tff_input_raw_base_values.shape} does not match factor names length {num_tff_factors}.")
            train_tff_inputs_raw = np.vstack([self.tff_input_raw_base_values, train_tff_inputs_raw])
        
        worker_args_list = [(self.product_static_params_for_worker,
                             self.pricer_config_for_worker,
                             self.tff_input_raw_factor_names,
                             train_tff_inputs_raw[i],
                             self.valuation_date_for_ql_settings_in_worker.isoformat(),
                             price_kwargs) for i in range(n_train)]

        train_prices = np.array([_price_one_scenario_for_tff(args) for args in worker_args_list])

        if train_prices.ndim == 0 and n_train == 1: train_prices = np.array([train_prices])
        if train_prices.shape[0] != n_train: raise ValueError(f"Shape of train_prices {train_prices.shape} != n_train {n_train}")

        # 2) Prepare inputs for fitting
        if self.feature_generation is not None:
            # use user‐provided feature generator
            feat_vals_train, feat_names = self.feature_generation.create_features()
            feat_normed, means, stds = normalize_features(feat_vals_train)
            tff_inputs_for_fitting = feat_normed
            normalization_params = {
                'means': means.tolist(),
                'stds': stds.tolist(),
                'engineered_feature_names': feat_names,
                'is_engineered': True
            }
        else:
            # fallback: raw factors or built-in option featurization
            tff_inputs_for_fitting = train_tff_inputs_raw
            normalization_params = {
                'means': None, 'stds': None,
                'engineered_feature_names': self.tff_input_raw_factor_names,
                'is_engineered': False
            }
            if self.product_type_str == 'EuropeanOption' and option_feature_order > 0:
                eng_vals, eng_names = engineer_option_features(
                    train_tff_inputs_raw[:,0], train_tff_inputs_raw[:,1],
                    order=option_feature_order
                )
                feat_normed, means, stds = normalize_features(eng_vals)
                tff_inputs_for_fitting = feat_normed
                normalization_params = {
                    'means': means.tolist(),
                    'stds': stds.tolist(),
                    'engineered_feature_names': eng_names,
                    'is_engineered': True
                }

        # 3) build X_train from tff_inputs_for_fitting and fit …
        D_eff = tff_inputs_for_fitting.shape[1]
        X_train = np.hstack([np.array([np.outer(s,s).flatten() for s in tff_inputs_for_fitting]), tff_inputs_for_fitting, np.ones((n_train,1))])
        if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)): raise ValueError("NaN/Inf in X_train.")
        if np.any(np.isnan(train_prices)) or np.any(np.isinf(train_prices)): raise ValueError("NaN/Inf in train_prices.")

        try: coeffs,_,_,_ = np.linalg.lstsq(X_train, train_prices, rcond=None)
        except np.linalg.LinAlgError as e: raise np.linalg.LinAlgError(f"Lstsq failed: {e}.")

        A_flat = coeffs[:D_eff*D_eff]
        A_mat = A_flat.reshape(D_eff,D_eff); A_sym = 0.5*(A_mat+A_mat.T)
        b_vec, c_s = coeffs[D_eff*D_eff : D_eff*D_eff+D_eff], coeffs[-1]
        
        # Create the fitted TensorFunctionalForm instance
        fitted_tff = TensorFunctionalForm(A_sym, b_vec, c_s)

        # 4) generate test scenarios & true prices …
        test_idx = rng_np.choice(full_market_scenarios_for_tff_factors.shape[0], size=n_test-1, replace=False)
        test_idx = np.insert(test_idx, 0, 0)  # Ensure the first scenario is always included
        test_tff_inputs_raw = full_market_scenarios_for_tff_factors[test_idx]
        test_worker_args = [(self.product_static_params_for_worker, self.pricer_config_for_worker,
             self.tff_input_raw_factor_names, test_tff_inputs_raw[i],
             self.valuation_date_for_ql_settings_in_worker.isoformat(), price_kwargs) for i in range(n_test)]
        #print(f"   Generating {n_test} test prices sequentially...")
        test_true_prices = np.array([_price_one_scenario_for_tff(args) for args in test_worker_args])

        # apply same feature logic to test‐set
        if self.feature_generation is not None:
            feat_vals_test, _ = self.feature_generation.create_features()
            test_inputs_eval, _, _ = normalize_features(
                feat_vals_test,
                np.array(normalization_params['means']),
                np.array(normalization_params['stds'])
            )
        else:
            test_inputs_eval = test_tff_inputs_raw
            if self.product_type_str == 'EuropeanOption' and normalization_params.get('is_engineered', False):
                eng_vals_test, _ = engineer_option_features(
                    test_tff_inputs_raw[:,0], test_tff_inputs_raw[:,1],
                    order=option_feature_order
                )
                test_inputs_eval, _, _ = normalize_features(
                    eng_vals_test,
                    np.array(normalization_params['means']),
                    np.array(normalization_params['stds'])
                )

        test_pred_prices = fitted_tff(test_inputs_eval)
        
        base_value = test_true_prices[0]
        base_tff_value = fitted_tff(tff_inputs_for_fitting[0]) 
        
        if test_true_prices.ndim==0 and n_test==1: test_true_prices = np.array([test_true_prices])
        if test_pred_prices.ndim==0 and n_test==1: test_pred_prices = np.array([test_pred_prices])
        if test_true_prices.shape != test_pred_prices.shape: raise ValueError(f"Shape mismatch test prices: true {test_true_prices.shape}, pred {test_pred_prices.shape}")

        
        rmse = np.sqrt(np.mean((test_true_prices - test_pred_prices)**2))
        return fitted_tff, test_tff_inputs_raw, test_true_prices, rmse, normalization_params, base_value, base_tff_value

