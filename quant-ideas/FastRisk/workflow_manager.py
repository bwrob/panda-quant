# workflow_manager.py
"""
Contains classes to manage the workflow of instrument processing (TFF calibration),
portfolio construction, and portfolio analytics (e.g., VaR).
Includes TFFConfigurationFactory to centralize TFF setup logic, now handling MBS.
Corrected TFFCalibrate instantiation and parameter passing.
"""
import json
from datetime import date, datetime
from typing import Dict
import numpy as np
import QuantLib as ql
import time
import abc
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from tff_approximator import TensorFunctionalForm
from base_pricer import PricerBase

# Import custom product definitions and pricers
# defer Portfolio import to methods that need it
 
from product_definitions import (
    ProductStaticBase, QuantLibBondStaticBase, CallableBondStaticBase,
    ConvertibleBondStaticBase, EuropeanOptionStatic, MBSPoolStatic
)
# Specific pricers from their new modules
from black_scholes_pricer import BlackScholesPricer
from quantlib_bond_pricer import QuantLibBondPricer
from fast_bond_pricer import FastBondPricer
from mbs_pricer import MBSPricer
# Note: ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel are 
# correctly imported from prepayment_models.py further down in this file.

from scenario_generator import SimpleRandomScenarioGenerator
from prepayment_models import ConstantCPRModel, PSAModel, RefiIncentivePrepaymentModel

from tff_approximator import TensorFunctionalFormCalibrate, _parse_numeric_pillars_from_factor_names  # Ensure Portfolio is imported from the correct module


# --- JSON Serialization Helpers ---
def portfolio_json_serializer(obj):
    if isinstance(obj, (datetime, date)): return obj.isoformat()
    if isinstance(obj, np.ndarray): return obj.tolist()
    if hasattr(obj, 'to_dict') and callable(obj.to_dict): return obj.to_dict()
    if isinstance(obj, ProductStaticBase): return obj.to_dict()
    if isinstance(obj, ql.Date): return date(obj.year(), obj.month(), obj.day()).isoformat()
    if isinstance(obj, ql.Calendar): return obj.name()
    if isinstance(obj, ql.DayCounter): return obj.name()
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)): return int(obj)
    if isinstance(obj, dict):
        return {k: portfolio_json_serializer(v) for k, v in obj.items()}
    raise TypeError(f"Object of type {obj.__class__.__name__} ({obj}) is not JSON serializable by custom serializer")

def reconstruct_product_static(product_dict: dict) -> ProductStaticBase:
    product_type = product_dict.get('product_type')
    if not product_type:
        raise ValueError("Product dictionary must contain a 'product_type' field.")
    if product_type == 'VanillaBond':
        return QuantLibBondStaticBase.from_dict(product_dict)
    elif product_type == 'CallableBond':
        return CallableBondStaticBase.from_dict(product_dict)
    elif product_type == 'ConvertibleBond':
        return ConvertibleBondStaticBase.from_dict(product_dict)
    elif product_type == 'EuropeanOption':
        return EuropeanOptionStatic.from_dict(product_dict)
    elif product_type == 'MBSPool':
        return MBSPoolStatic.from_dict(product_dict)
    else:
        raise ValueError(f"Unknown product_type for reconstruction: {product_type}")

def generate_portfolio_specs_for_serialization(
    holdings_data: list[dict],
    model_registry: dict,
    instrument_definitions_data_for_pricer_params: list[dict] = None
    ) -> list[dict]:
    portfolio_specs_for_json = []
    if instrument_definitions_data_for_pricer_params is None:
        instrument_definitions_data_for_pricer_params = []

    for holding in holdings_data:
        instrument_id = holding.get("instrument_id")
        if not instrument_id:
            print(f"   Skipping holding due to missing instrument_id: {holding}")
            continue

        if instrument_id in model_registry and not model_registry[instrument_id].get('error'):
            entry = model_registry[instrument_id]
            spec_item = {
                "client_id": holding.get("client_id"),
                "instrument_id": instrument_id,
                "num_holdings": holding.get("num_holdings"),
                "pricing_engine_type": entry["pricing_method"].lower(),
                "product_static_object": entry["product_static_dict"]
            }
            if entry["pricing_method"] == 'TFF' and 'tff_model_dict' in entry:
                spec_item["direct_tff_config"] = {
                    "model_dict": entry["tff_model_dict"],
                    "raw_input_names": entry["tff_raw_input_names"],
                    "normalization_params": entry["tff_normalization_params"],
                    "option_feature_order": entry.get("tff_option_feature_order", 0)
                }
                if 'tff_fixed_pricer_params' in entry:
                    spec_item['pricer_params'] = entry['tff_fixed_pricer_params']

            if entry["pricing_method"] == 'FULL':
                if 'pricer_params' in entry:
                     spec_item['pricer_params'] = entry['pricer_params']
                else:
                    original_instrument_spec = next(
                        (item for item in instrument_definitions_data_for_pricer_params
                         if item.get("instrument_id") == instrument_id),
                        None
                    )
                    if original_instrument_spec and 'pricer_params' in original_instrument_spec:
                         spec_item['pricer_params'] = original_instrument_spec['pricer_params']

            portfolio_specs_for_json.append(spec_item)
        else:
            print(f"   Skipping instrument '{instrument_id}' for JSON spec generation: not in valid model_registry or had an error.")
    return portfolio_specs_for_json

class PortfolioBase(abc.ABC):
    """
    Abstract base class for a portfolio of financial instruments.
    """
    def __init__(self):
        # Stores details for each *position* in the portfolio.
        # Multiple positions can refer to the same underlying instrument_id if TFFs are cached.
        self.positions: list[dict] = []

    @abc.abstractmethod
    def add_position(self, *args, **kwargs):
        """Adds a position (instrument holding) to the portfolio."""
        pass

    @abc.abstractmethod
    def price_portfolio(self,
                        raw_market_scenarios: np.ndarray,
                        scenario_factor_names: list[str],
                        portfolio_rate_pillar_times: np.ndarray = None # For bond pricers
                        ) -> np.ndarray:
        """
        Prices all instruments in the portfolio for given market scenarios.

        Args:
            raw_market_scenarios (np.ndarray): 2D array of market scenarios
                                               (N_scenarios, N_total_market_factors).
            scenario_factor_names (list[str]): Names of the columns in raw_market_scenarios.
            portfolio_rate_pillar_times (np.ndarray, optional): 1D array of rate pillar times.
                                                               Required if portfolio contains bonds
                                                               priced with full QuantLibBondPricer.
        Returns:
            np.ndarray: 1D array of aggregated portfolio prices for each scenario (N_scenarios,).
        """
        pass
    
class Portfolio(PortfolioBase):
    """
    A portfolio where each instrument can be priced using either a pre-fitted
    Tensor Functional Form (TFF) model (retrieved from a cache) or its original full pricer.
    """
    def __init__(self):
        super().__init__()
        self.tff_model_cache: dict = {}
        # Cache structure:
        # { instrument_id: {
        #     'tff_model': TensorFunctionalForm_object,
        #     'raw_tff_input_names': list_of_names, # Raw factors TFF was trained on
        #     'normalization_params': dict_of_norm_params, # Includes engineered_feature_names
        #     'option_feature_order': int
        #   }, ...
        # }

    def to_dict(self) -> dict:
        """
        Returns a dictionary representation of the portfolio.

        Returns:
            dict: Dictionary representation of the portfolio.
        """
        return {
            'positions': [p.to_dict() if hasattr(p, 'to_dict') else p
                for p in self.positions],
            'tff_model_cache': self.tff_model_cache
        }

    def cache_tff_model(self,
                        instrument_id: str,
                        tff_model: TensorFunctionalForm,
                        raw_tff_input_names: list[str],
                        normalization_params: dict,
                        option_feature_order: int = 0):
        """
        Explicitly caches a fitted TFF model and its associated parameters.

        Args:
            instrument_id (str): Unique ID for the instrument type this TFF represents.
            tff_model (TensorFunctionalForm): The pre-fitted TFF model.
            raw_tff_input_names (list[str]): Names of raw market factors TFF is based on.
            normalization_params (dict): Normalization params from TFF calibration.
                                         Expected keys: 'means', 'stds', 'engineered_feature_names', 'is_engineered'.
            option_feature_order (int, optional): Order of feature engineering if option TFF.
        """
        if not instrument_id:
            raise ValueError("instrument_id must be provided for caching TFF model.")
        if not isinstance(tff_model, TensorFunctionalForm):
            raise TypeError("tff_model must be an instance of TensorFunctionalForm.")
        if raw_tff_input_names is None or not isinstance(raw_tff_input_names, list):
            raise ValueError("raw_tff_input_names (list of strings) must be provided for caching TFF model.")
        if normalization_params is None or not isinstance(normalization_params, dict):
            raise ValueError("normalization_params (dict) must be provided for caching TFF model.")

        self.tff_model_cache[instrument_id] = {
            'tff_model': tff_model,
            'raw_tff_input_names': raw_tff_input_names,
            'normalization_params': normalization_params,
            'option_feature_order': option_feature_order
        }

    def from_dict(self, portfolio_dict: dict):
        """
        Loads a portfolio from a dictionary representation. Similar to cache_tff_model

        Args:
            portfolio_dict (dict): Dictionary representation of the portfolio.
        """



    def add_position(self,
                       instrument_id: str,
                       product_static: ProductStaticBase,
                       num_holdings: int = 1,
                       pricing_engine_type: str = 'tff',
                       direct_tff_config: dict = None,
                       full_pricer_instance: PricerBase = None,
                       full_pricer_kwargs: dict = None):
        """
        Adds a position (an instrument holding) to the portfolio.
        If pricing_engine_type is 'tff', it retrieves the TFF model from the cache
        using the instrument_id. The TFF model must have been cached previously using `cache_tff_model`.

        Args:
            instrument_id (str): Unique ID for this instrument type. Used for TFF caching/lookup.
            product_static (ProductStaticBase): Static definition of the product for this position.
            num_holdings (int, optional): Number of units. Defaults to 1.
            pricing_engine_type (str, optional): 'tff' or 'full'. Defaults to 'tff'.
            full_pricer_instance (PricerBase, optional): A full pricer instance (required if type is 'full').
            full_pricer_kwargs (dict, optional): Keyword arguments for the full pricer's price method.
        """
        if not isinstance(num_holdings, int) or num_holdings <= 0:
                raise ValueError("num_holdings must be a positive integer.")
        if not instrument_id:
            raise ValueError("instrument_id must be provided for the position.")

        position_detail = {
            'instrument_id': instrument_id,
            'product_static': product_static,
            'num_holdings': num_holdings,
            'engine_type': pricing_engine_type.lower()
        }

        if position_detail['engine_type'] == 'tff':
            if direct_tff_config is not None:
                # A TFF model and its configuration are being provided directly as a dictionary
                if not isinstance(direct_tff_config, dict):
                    raise TypeError("direct_tff_config must be a dictionary.")

                model_dict = direct_tff_config.get('model_dict')
                raw_names = direct_tff_config.get('raw_input_names')
                norm_params = direct_tff_config.get('normalization_params')

                # Default option_feature_order to 0 if not explicitly in config
                opt_order = direct_tff_config.get('option_feature_order', 0)

                if model_dict is None or not isinstance(model_dict, dict):
                    raise ValueError("direct_tff_config is missing 'model_dict' or it's not a dictionary.")
                if raw_names is None or not isinstance(raw_names, list):
                    raise ValueError("direct_tff_config is missing 'raw_input_names' or it's not a list.")
                if norm_params is None or not isinstance(norm_params, dict):
                    raise ValueError("direct_tff_config is missing 'normalization_params' or it's not a dictionary.")

                # Deserialize the TFF model itself from model_dict
                try:
                    tff_model_instance = TensorFunctionalForm.from_dict(model_dict)
                except Exception as e:
                    raise ValueError(f"Failed to deserialize TFF model from direct_tff_config['model_dict']: {e}")

                position_detail['pricer_engine'] = tff_model_instance
                position_detail['raw_tff_input_names'] = raw_names
                position_detail['normalization_params'] = norm_params
                position_detail['option_feature_order'] = opt_order

            else:
                # No direct TFF config provided, so use the cache via instrument_id
                if instrument_id not in self.tff_model_cache:
                    raise ValueError(
                        f"TFF model for instrument_id '{instrument_id}' not found in cache. "
                        "Either fit and cache it first, or provide its configuration via 'direct_tff_config'."
                    )

                cached_data = self.tff_model_cache[instrument_id]
                position_detail['pricer_engine'] = cached_data['tff_model']
                position_detail['raw_tff_input_names'] = cached_data['raw_tff_input_names']
                position_detail['normalization_params'] = cached_data['normalization_params']
                position_detail['option_feature_order'] = cached_data['option_feature_order']

        elif position_detail['engine_type'] == 'full':
            if not isinstance(full_pricer_instance, PricerBase):
                raise TypeError("full_pricer_instance must be an instance of PricerBase if engine_type is 'full'.")
            position_detail['pricer_engine'] = full_pricer_instance
            position_detail['full_pricer_kwargs'] = full_pricer_kwargs or {}
        else:
            raise ValueError(f"Unsupported pricing_engine_type: {pricing_engine_type}. Choose 'tff' or 'full'.")

        self.positions.append(position_detail)

    def load_portfolio_from_specs(self, portfolio_specs: list[dict]):
        """
        Loads multiple positions into the portfolio from a list of specifications.

        Each specification in the list is a dictionary that should conform to
        the parameters expected by the `add_position` method.

        Args:
            portfolio_specs (list[dict]): A list of dictionaries, where each dictionary
                                          defines a position to be added.
                                          Expected keys in each dict:
                                            'instrument_id' (str)
                                            'product_static_object' (ProductStaticBase)
                                            'num_holdings' (int)
                                            'pricing_engine_type' (str: 'tff' or 'full')
                                            Optional for 'tff' type:
                                              'direct_tff_config' (dict, containing 'model_dict',
                                                                   'raw_input_names', 'normalization_params',
                                                                   'option_feature_order')
                                            Optional for 'full' type:
                                              'full_pricer_instance' (PricerBase)
                                              'full_pricer_kwargs' (dict)
        """
        if not isinstance(portfolio_specs, list):
            raise TypeError("portfolio_specs must be a list of dictionaries.")

        for i, item_spec in enumerate(portfolio_specs):
            if not isinstance(item_spec, dict):
                raise TypeError(f"Each item in portfolio_specs must be a dictionary. Found type {type(item_spec)} at index {i}.")

            try:
                self.add_position(
                    instrument_id=item_spec['instrument_id'],
                    product_static=item_spec['product_static_object'],
                    num_holdings=item_spec.get('num_holdings', 1), # Default if not specified
                    pricing_engine_type=item_spec.get('pricing_engine_type', 'tff'), # Default

                    direct_tff_config=item_spec.get('direct_tff_config'), # Will be None if key doesn't exist

                    full_pricer_instance=item_spec.get('full_pricer_instance'),
                    full_pricer_kwargs=item_spec.get('full_pricer_kwargs')
                )
            except KeyError as e:
                raise ValueError(f"Missing required key {e} in portfolio_spec at index {i}: {item_spec}")
            except Exception as e:
                raise RuntimeError(f"Error adding position from spec at index {i} ('{item_spec.get('instrument_id', 'Unknown ID')}'): {e}")

        print(f"Successfully loaded {len(self.positions)} positions into the portfolio from specifications.")

    def price_portfolio(self,
                        raw_market_scenarios: np.ndarray,
                        scenario_factor_names: list[str],
                        portfolio_rate_pillar_times: np.ndarray = None
                        ) -> np.ndarray:
        """
        Prices all positions in the portfolio for given market scenarios.

        Args:
            raw_market_scenarios (np.ndarray): 2D array of raw market scenarios
                                               (N_scenarios, N_total_market_factors).
            scenario_factor_names (list[str): Names of the columns in raw_market_scenarios.
            portfolio_rate_pillar_times (np.ndarray, optional): 1D array of rate pillar times (numeric tenors).
                                                               Required if portfolio contains bonds
                                                               priced with full QuantLibBondPricer or FastBondPricer.

        Returns:
            np.ndarray: 1D array of aggregated portfolio prices for each scenario (N_scenarios,).
        """
        if not self.positions:
            return np.array([])

        num_scenarios = raw_market_scenarios.shape[0]
        portfolio_prices_per_scenario = np.zeros(num_scenarios, dtype=float)

        for position_detail in self.positions:
            pricer_engine = position_detail['pricer_engine']
            engine_type   = position_detail['engine_type']
            num_holdings  = position_detail['num_holdings']

            instrument_prices_this_instrument = np.zeros(num_scenarios, dtype=float)

            if engine_type == 'tff':
                tff_model: TensorFunctionalForm = pricer_engine
                # These are the names of the RAW factors the TFF was originally trained on.
                raw_tff_input_factor_names_for_this_tff = position_detail['raw_tff_input_names']
                norm_params = position_detail['normalization_params']
                opt_feat_order = position_detail['option_feature_order']

                # Select the relevant columns from the global raw_market_scenarios
                try:
                    indices_of_raw_factors_in_global_scenarios = [scenario_factor_names.index(name) for name in raw_tff_input_factor_names_for_this_tff]
                except ValueError as e:
                    raise ValueError(f"A TFF input name in {raw_tff_input_factor_names_for_this_tff} not found in scenario_factor_names {scenario_factor_names} for instrument_id '{position_detail['instrument_id']}'. Error: {e}")

                current_raw_inputs_for_tff = raw_market_scenarios[:, indices_of_raw_factors_in_global_scenarios]

                inputs_for_tff_evaluation = current_raw_inputs_for_tff # Default

                # opt_feat_order should be available from position_detail if it's an option TFF
                opt_feat_order = position_detail.get('option_feature_order', 0)

                if isinstance(position_detail['product_static'], EuropeanOptionStatic) and \
                   norm_params.get('is_engineered', False): # Check if TFF was trained with engineered features

                    s0_factor_actual_name_port = None
                    vol_factor_actual_name_port = None
                    # These indices are relative to raw_tff_input_factor_names_for_this_tff
                    # and thus to the columns of current_raw_inputs_for_tff
                    s0_idx_in_tff_inputs = -1
                    vol_idx_in_tff_inputs = -1

                    # For an option TFF, raw_tff_input_factor_names_for_this_tff is expected to be [full_s0_name, full_vol_name]
                    if len(raw_tff_input_factor_names_for_this_tff) == 2:
                        for i, name in enumerate(raw_tff_input_factor_names_for_this_tff):
                            if name.upper().endswith("_S0"): # Using suffix matching
                                s0_factor_actual_name_port = name
                                s0_idx_in_tff_inputs = i
                            elif name.upper().endswith("_VOLATILITY") or name.upper().endswith("_VOL"): # Using suffix matching
                                vol_factor_actual_name_port = name
                                vol_idx_in_tff_inputs = i

                        if s0_idx_in_tff_inputs == -1 or vol_idx_in_tff_inputs == -1 or s0_idx_in_tff_inputs == vol_idx_in_tff_inputs:
                            raise ValueError(
                                f"Portfolio pricing: Could not identify distinct S0 and Volatility factors for option "
                                f"'{position_detail['instrument_id']}' from its TFF input names: "
                                f"{raw_tff_input_factor_names_for_this_tff}."
                            )
                    else:
                        raise ValueError(
                            f"Portfolio pricing: Option TFF for '{position_detail['instrument_id']}' with engineered features "
                            f"expects 2 raw input factors (S0, Vol), but "
                            f"TFF was trained on {len(raw_tff_input_factor_names_for_this_tff)} names: "
                            f"{raw_tff_input_factor_names_for_this_tff}."
                        )

                    # current_raw_inputs_for_tff columns are ordered as per raw_tff_input_factor_names_for_this_tff
                    s0_scenarios_raw = current_raw_inputs_for_tff[:, s0_idx_in_tff_inputs]
                    vol_scenarios_raw = current_raw_inputs_for_tff[:, vol_idx_in_tff_inputs]

                    engineered_features_test, _ = engineer_option_features( # Changed name for clarity
                        s0_scenarios_raw, vol_scenarios_raw, order=opt_feat_order
                    )
                    # Use normalization_params['means'] and ['stds'] from the cached TFF model data
                    inputs_for_tff_evaluation, _, _ = normalize_features(
                        engineered_features_test, # Changed name
                        norm_params['means'],
                        norm_params['stds']
                    )

                instrument_prices_this_instrument = tff_model(inputs_for_tff_evaluation)

            elif engine_type == 'full':
                # Delegate to each pricer's own scenario‐slicing logic
                full_pricer: PricerBase = pricer_engine
                pricer_kwargs = position_detail.get('full_pricer_kwargs', {})
                instrument_prices_this_instrument = full_pricer.price_scenarios(
                    raw_market_scenarios,
                    scenario_factor_names,
                    portfolio_rate_pillar_times,
                    **pricer_kwargs
                )
            else:
                raise ValueError(f"Unknown engine_type: {engine_type} for instrument_id '{position_detail['instrument_id']}'")

            # Ensure consistent shapes for aggregation
            if instrument_prices_this_instrument.ndim == 0:
                instrument_prices_this_instrument = np.full(num_scenarios, float(instrument_prices_this_instrument))
            elif len(instrument_prices_this_instrument) != num_scenarios:
                 instrument_prices_this_instrument = instrument_prices_this_instrument.flatten()
                 if len(instrument_prices_this_instrument) != num_scenarios:
                    raise ValueError(f"Price array shape mismatch for instrument '{position_detail['instrument_id']}'. Expected ({num_scenarios},), got {instrument_prices_this_instrument.shape}")

            portfolio_prices_per_scenario += instrument_prices_this_instrument * num_holdings

        return portfolio_prices_per_scenario

    def price_portfolio_from_strips(self,
                                    strips: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Aggregate portfolio values from pre-generated price strips.

        Args:
            strips: mapping from instrument_id to 1D np.ndarray of scenario prices.

        Returns:
            1D np.ndarray of aggregated portfolio price per scenario.
        """
        # nothing to do
        if not self.positions:
            return np.array([], dtype=float)

        # determine scenario count from first available strip
        try:
            first_strip = next(iter(strips.values()))
            num_scen = first_strip.shape[0]
        except StopIteration:
            raise ValueError("No price strips provided to aggregate.")

        result = np.zeros(num_scen, dtype=float)

        for pos in self.positions:
            inst_id = pos['instrument_id']
            nh     = pos['num_holdings']
            if pos['engine_type'] == 'full':
                # full‐pricer positions must have a strip
                if inst_id not in strips:
                    raise KeyError(f"Missing strip for instrument '{inst_id}'")
                prices = strips[inst_id]
                if prices.shape[0] != num_scen:
                    raise ValueError(f"Strip length mismatch for '{inst_id}': "
                                     f"expected {num_scen}, got {prices.shape[0]}")
                result += prices * nh
            else:
                # TFF‐engine positions cannot be aggregated from full‐pricer strips.
                raise RuntimeError(
                    f"Cannot aggregate TFF position '{inst_id}' from full‐pricer strips. "
                    "Use price_portfolio() or generate separate TFF strips."
                )

        return result
    
# --- TFF Configuration Factory (Defined BEFORE InstrumentProcessor) ---
class TFFConfigurationFactory:
    def __init__(self,
                 scenario_generator: SimpleRandomScenarioGenerator,
                 default_numeric_rate_tenors: np.ndarray,
                 default_bs_risk_free_rate: float = 0.025,
                 default_bs_dividend_yield: float = 0.0):
        self.scenario_generator = scenario_generator
        self.default_numeric_rate_tenors = default_numeric_rate_tenors
        self.default_bs_rfr = default_bs_risk_free_rate
        self.default_bs_div = default_bs_dividend_yield

    def _get_base_value(self, factor_name: str) -> float:
        """Helper to get base value from scenario generator's maps."""
        for m_map_name in ['base_rates_map', 'base_s0_map', 'base_vol_map', 'base_credit_spread_points_map']:
            m_map = getattr(self.scenario_generator, m_map_name, {})
            if factor_name in m_map:
                return m_map[factor_name]
        if hasattr(self.scenario_generator, 'base_s0_map') and factor_name in self.scenario_generator.base_s0_map:
             return self.scenario_generator.base_s0_map[factor_name]
        raise ValueError(f"Base value for TFF factor '{factor_name}' not found in scenario_generator's configured base maps.")

    def create_config(self, product_static: ProductStaticBase,
                      tff_behavior_params: dict = None,
                      instrument_pricer_params: dict = None) -> dict:
        if tff_behavior_params is None: tff_behavior_params = {}
        if instrument_pricer_params is None: instrument_pricer_params = {}

        raw_names = []
        raw_base_values = []
        fixed_params_for_training = {}
        opt_feature_order = 0
        pricer_cfg_worker = {}
        actual_pillars = np.array([])


        if isinstance(product_static, EuropeanOptionStatic):
            if not product_static.underlying_symbol or not product_static.currency:
                raise ValueError("EuropeanOptionStatic needs 'underlying_symbol' and 'currency'.")
            s0_fn = f"{product_static.currency}_{product_static.underlying_symbol}_S0"
            vol_fn = f"{product_static.currency}_{product_static.underlying_symbol}_VOL"
            raw_names = [s0_fn, vol_fn]
            raw_base_values = [self._get_base_value(s0_fn), self._get_base_value(vol_fn)]
            opt_feature_order = tff_behavior_params.get('option_feature_order', 0)
            pricer_cfg_worker['bs_pricer_config'] = {
                'risk_free_rate': instrument_pricer_params.get('bs_risk_free_rate'),
                'dividend_yield': instrument_pricer_params.get('bs_dividend_yield', 0.0)
            }
            if pricer_cfg_worker['bs_pricer_config']['risk_free_rate'] is None:
                raise ValueError("Missing 'bs_risk_free_rate' in pricer_params for EuropeanOption.")


        elif isinstance(product_static, QuantLibBondStaticBase) or isinstance(product_static, MBSPoolStatic):
            if not product_static.currency or not product_static.index_stub:
                raise ValueError(f"{product_static.__class__.__name__} needs 'currency' and a non-empty 'index_stub'.")
            if self.default_numeric_rate_tenors is None or self.default_numeric_rate_tenors.size == 0:
                raise ValueError("default_numeric_rate_tenors needed for TFF setup.")

            rate_factor_names = [f"{product_static.currency}_{product_static.index_stub}_{t:.2f}Y" for t in self.default_numeric_rate_tenors]
            base_rate_vals = [self._get_base_value(name) for name in rate_factor_names]
            raw_names.extend(rate_factor_names)
            raw_base_values.extend(base_rate_vals)
            actual_pillars = _parse_numeric_pillars_from_factor_names(rate_factor_names)

            pricer_cfg_worker['bond_pricer_config'] = {'method': 'discount'} # Default for Vanilla/MBS
            if isinstance(product_static, MBSPoolStatic):
                pricer_cfg_worker['mbs_pricer_config'] = { # For MBSPricer specific init if any (none currently)
                    'prepayment_model_type': product_static.prepayment_model_type,
                    'prepayment_rate_param': product_static.prepayment_rate_param
                }
                # Fixed params for MBSPricer.price() if not dynamic TFF factors
                # (e.g., fixed_market_mortgage_rate_for_prepay for simple Refi model)
                fixed_mbs_p = tff_behavior_params.get('fixed_mbs_params', {})
                if 'fixed_market_mortgage_rate_for_prepay' in fixed_mbs_p:
                     fixed_params_for_training['fixed_market_mortgage_rate_for_prepay'] = fixed_mbs_p['fixed_market_mortgage_rate_for_prepay']
                if 'refi_A' in fixed_mbs_p: fixed_params_for_training['refi_A'] = fixed_mbs_p['refi_A']
                if 'refi_B' in fixed_mbs_p: fixed_params_for_training['refi_B'] = fixed_mbs_p['refi_B']
                if 'refi_C' in fixed_mbs_p: fixed_params_for_training['refi_C'] = fixed_mbs_p['refi_C']
                if 'refi_D' in fixed_mbs_p: fixed_params_for_training['refi_D'] = fixed_mbs_p['refi_D']


            if hasattr(product_static, 'credit_spread_curve_name') and product_static.credit_spread_curve_name:
                cs_curve_name = product_static.credit_spread_curve_name
                cs_factor_names = [f"{cs_curve_name}_{t:.2f}Y" for t in self.default_numeric_rate_tenors]
                base_cs_vals = [self._get_base_value(name) for name in cs_factor_names]
                raw_names.extend(cs_factor_names)
                raw_base_values.extend(base_cs_vals)

            if isinstance(product_static, CallableBondStaticBase):
                pricer_cfg_worker['bond_pricer_config']['method'] = 'g2'
                pricer_cfg_worker['bond_pricer_config']['grid_steps'] = instrument_pricer_params.get('g2_grid_steps', 32)
                if instrument_pricer_params.get('g2_params'):
                    fixed_params_for_training['g2_params'] = instrument_pricer_params['g2_params']

            elif isinstance(product_static, ConvertibleBondStaticBase):
                pricer_cfg_worker['bond_pricer_config']['method'] = 'convertible_binomial'
                pricer_cfg_worker['bond_pricer_config']['convertible_engine_steps'] = instrument_pricer_params.get('conv_engine_steps', 128)

                if not product_static.underlying_symbol: raise ValueError("Convertible needs 'underlying_symbol'.")
                s0_fn_cb = f"{product_static.currency}_{product_static.underlying_symbol}_S0"
                if s0_fn_cb not in raw_names:
                    raw_names.append(s0_fn_cb); raw_base_values.append(self._get_base_value(s0_fn_cb))

                conv_all_dynamic = tff_behavior_params.get('convertible_tff_market_inputs_as_factors', False)

                if conv_all_dynamic:
                    div_fn = f"{product_static.currency}_{product_static.underlying_symbol}_DIVYIELD"
                    vol_fn = f"{product_static.currency}_{product_static.underlying_symbol}_EQVOL"
                    cs_fn_engine = f"{product_static.currency}_{product_static.underlying_symbol}_CS"

                    new_factors = []
                    new_base_values = []
                    if div_fn not in raw_names: new_factors.append(div_fn); new_base_values.append(self._get_base_value(div_fn))
                    if vol_fn not in raw_names: new_factors.append(vol_fn); new_base_values.append(self._get_base_value(vol_fn))
                    if cs_fn_engine not in raw_names: new_factors.append(cs_fn_engine); new_base_values.append(self._get_base_value(cs_fn_engine))

                    raw_names.extend(new_factors)
                    raw_base_values.extend(new_base_values)
                else:
                    fixed_cb_p = tff_behavior_params.get('fixed_cb_params', {})
                    fixed_params_for_training['dividend_yield'] = fixed_cb_p.get('dividend_yield')
                    fixed_params_for_training['equity_volatility'] = fixed_cb_p.get('equity_volatility')
                    fixed_params_for_training['credit_spread'] = fixed_cb_p.get('credit_spread')
                    if any(v is None for k,v in fixed_params_for_training.items() if k in ['dividend_yield', 'equity_volatility', 'credit_spread']):
                        raise ValueError(f"Missing fixed CB params (div,eq_vol,cs) when S0 is dynamic but others fixed. Got: {fixed_cb_p}")
        else:
            raise TypeError(f"Unsupported product type for TFF Configuration: {type(product_static)}")

        return {
            "tff_input_raw_factor_names": raw_names,
            "tff_input_raw_base_values": np.array(raw_base_values),
            "fixed_pricer_params_for_tff_training": fixed_params_for_training,
            "option_feature_order": opt_feature_order,
            "pricer_config_for_worker": pricer_cfg_worker,
            "actual_rate_pillars": actual_pillars
        }


# --- Workflow Classes ---
class InstrumentProcessor:
    def __init__(self, scenario_generator: SimpleRandomScenarioGenerator,
                 global_valuation_date: date,
                 default_numeric_rate_tenors: np.ndarray = None,
                 default_g2_params = None,
                 default_bs_risk_free_rate: float = 0.025,
                 default_bs_dividend_yield: float = 0.0,
                 parallel_workers_tff: int = None,
                 n_scenarios_for_tff_domain: int = 1000
                 ):
        self.scenario_generator = scenario_generator
        self.global_valuation_date = global_valuation_date
        self.default_numeric_rate_tenors = default_numeric_rate_tenors
        self.default_g2_params = default_g2_params
        self.default_bs_risk_free_rate = default_bs_risk_free_rate
        self.default_bs_dividend_yield = default_bs_dividend_yield
        self.num_instrument_processing_workers = parallel_workers_tff if parallel_workers_tff else 0
        self.n_scenarios_for_tff_domain = n_scenarios_for_tff_domain

        self.model_registry = {}
        self.tff_config_factory = TFFConfigurationFactory(
            scenario_generator=self.scenario_generator,
            default_numeric_rate_tenors=self.default_numeric_rate_tenors,
            default_bs_risk_free_rate=self.default_bs_risk_free_rate,
            default_bs_dividend_yield=self.default_bs_dividend_yield
        )

    def _create_pricer_template(self, product_static, instrument_spec: dict):
        pricer_params = instrument_spec.get('pricer_params', {})
        if isinstance(product_static, EuropeanOptionStatic):
            # instantiate with no rates
            pricer = BlackScholesPricer(product_static)
            # keep defaults for price() calls
            pricer._default_price_kwargs = {
              'risk_free_rate': pricer_params.get('bs_risk_free_rate', self.default_bs_risk_free_rate),
              'dividend_yield': pricer_params.get('bs_dividend_yield', self.default_bs_dividend_yield)
            }
            return pricer
        elif isinstance(product_static, CallableBondStaticBase):
            grid_steps = pricer_params.get('g2_grid_steps', 32)
            return QuantLibBondPricer(product_static, method='g2', grid_steps=grid_steps)
        elif isinstance(product_static, ConvertibleBondStaticBase):
            engine_steps = pricer_params.get('conv_engine_steps', 128)
            return QuantLibBondPricer(product_static, method='convertible_binomial', convertible_engine_steps=engine_steps)
        elif isinstance(product_static, MBSPoolStatic): # NEW
            prepayment_model_type = product_static.prepayment_model_type
            prepayment_rate_param = product_static.prepayment_rate_param
            prepayment_model_instance = None
            if prepayment_model_type == "CPR":
                prepayment_model_instance = ConstantCPRModel(prepayment_rate_param)
            elif prepayment_model_type == "PSA":
                prepayment_model_instance = PSAModel(prepayment_rate_param)
            elif prepayment_model_type == "RefiIncentive":
                # Get A,B,C,D from pricer_params if provided, else use model defaults
                refi_A = pricer_params.get('refi_A')
                refi_B = pricer_params.get('refi_B')
                refi_C = pricer_params.get('refi_C')
                refi_D = pricer_params.get('refi_D')
                if all(p is not None for p in [refi_A, refi_B, refi_C, refi_D]):
                    prepayment_model_instance = RefiIncentivePrepaymentModel(refi_A, refi_B, refi_C, refi_D)
                else:
                    prepayment_model_instance = RefiIncentivePrepaymentModel()
            else: raise ValueError(f"Unsupported prepayment_model_type: {prepayment_model_type} for MBS.")
            return MBSPricer(product_static, prepayment_model=prepayment_model_instance)
        elif isinstance(product_static, QuantLibBondStaticBase):
            if instrument_spec.get('pricer_type_preference', 'QuantLib').upper() == 'FAST':
                 return FastBondPricer(product_static)
            return QuantLibBondPricer(product_static, method='discount')
        else:
            raise ValueError(f"Unsupported product type for pricer template: {type(product_static)}")

    def _get_scenario_slice(self, all_scenarios, all_factor_names, target_factor_names_for_tff):
        if not target_factor_names_for_tff:
            return np.array([]).reshape(all_scenarios.shape[0],0)
        try:
            global_indices_map = {name: i for i, name in enumerate(all_factor_names)}
            ordered_indices = [global_indices_map[name] for name in target_factor_names_for_tff]
            return all_scenarios[:, ordered_indices]
        except KeyError as e:
            missing_factor = str(e).strip("'")
            raise ValueError(
                f"Error slicing scenarios: Factor name '{missing_factor}' required by TFF "
                f"not found in generated scenario factor names. "
                f"Required by TFF: {target_factor_names_for_tff}, "
                f"Available from generator: {all_factor_names}."
            )
        except Exception as e:
            raise RuntimeError(f"General error during scenario slicing for TFF factors {target_factor_names_for_tff}: {e}")

    def _process_single_instrument_spec(self, args_tuple):
        instrument_spec, global_market_scenarios, global_factor_names, ql_val_date_iso = args_tuple

        val_d_worker = date.fromisoformat(ql_val_date_iso)
        ql.Settings.instance().evaluationDate = ql.Date(val_d_worker.day, val_d_worker.month, val_d_worker.year)

        instrument_id = instrument_spec.get('instrument_id')
        product_type_str = instrument_spec.get('product_type')
        params = instrument_spec.get('params', {})
        pricing_preference = instrument_spec.get('pricing_preference', 'FULL').upper()

        registry_entry = {'instrument_id': instrument_id, 'pricing_method': pricing_preference}
        if 'valuation_date' not in params: params['valuation_date'] = self.global_valuation_date
        if 'product_type' not in params: params['product_type'] = product_type_str

        try:
            product_static_object = reconstruct_product_static(params)
            registry_entry['product_static_dict'] = product_static_object.to_dict()
        except Exception as e:
            print(f"    ERROR creating product static for {instrument_id} in worker: {e}")
            registry_entry.update({'error': str(e), 'pricing_method': 'ERROR'})
            return instrument_id, registry_entry

        pricer_template = self._create_pricer_template(product_static_object, instrument_spec)
        if 'pricer_params' in instrument_spec: registry_entry['pricer_params'] = instrument_spec['pricer_params']

        if pricing_preference == 'TFF':
            tff_config_from_spec = instrument_spec.get('tff_config', {})
            factory_behavior_params = tff_config_from_spec.copy()
            # Pass pricer_params from instrument_spec to factory for fixed_cb_params or fixed_bs_params
            factory_behavior_params['fixed_cb_params'] = instrument_spec.get('pricer_params', {})
            factory_behavior_params['fixed_bs_params'] = instrument_spec.get('pricer_params', {})
            if isinstance(product_static_object, MBSPoolStatic):
                 factory_behavior_params['fixed_mbs_params'] = instrument_spec.get('pricer_params',{})

            try:
                #print(f"    Calibrating TFF for {instrument_id} (in worker)...")
                tff_inputs = self.tff_config_factory.create_config(
                    product_static=product_static_object,
                    tff_behavior_params=factory_behavior_params,
                    instrument_pricer_params=instrument_spec.get('pricer_params', {})
                )

                tff_sample_fit_parallel_workers = False

                # Use the simplified TFFCalibrate constructor
                tff_calibrator = TensorFunctionalFormCalibrate(
                    pricer_template=pricer_template,
                    tff_input_raw_factor_names=tff_inputs["tff_input_raw_factor_names"],
                    tff_input_raw_base_values=tff_inputs["tff_input_raw_base_values"],
                    product_static_params_for_worker=product_static_object.to_dict(),
                    pricer_config_for_worker=tff_inputs["pricer_config_for_worker"],
                    actual_rate_pillars=tff_inputs["actual_rate_pillars"]
                )

                scenarios_for_this_tff = self._get_scenario_slice(global_market_scenarios, global_factor_names, tff_inputs["tff_input_raw_factor_names"])
                if scenarios_for_this_tff.size == 0 and tff_inputs["tff_input_raw_factor_names"]:
                     raise ValueError("Empty scenario slice for TFF fitting.")

                s_t = time.time()
                model_tff, _, _, rmse_tff, norm_params_tff, base_value_tff, base_tff_value = tff_calibrator.sample_and_fit(
                    full_market_scenarios_for_tff_factors=scenarios_for_this_tff,
                    n_train=tff_config_from_spec.get('n_train', 64),
                    n_test=tff_config_from_spec.get('n_test', 8),
                    random_seed=instrument_spec.get('seed', 42),
                    parallel_workers=tff_sample_fit_parallel_workers,
                    option_feature_order=tff_inputs["option_feature_order"],
                    **tff_inputs["fixed_pricer_params_for_tff_training"]
                )
                fit_time = time.time() - s_t
                if model_tff and norm_params_tff:
                    registry_entry.update({
                        'tff_model_dict': model_tff.to_dict(),
                        'tff_raw_input_names': tff_inputs["tff_input_raw_factor_names"],
                        'tff_normalization_params': norm_params_tff,
                        'tff_option_feature_order': tff_inputs["option_feature_order"],
                        'tff_rmse': rmse_tff, 'tff_fit_time_seconds': fit_time,
                        'tff_base_value': base_value_tff, 'tff_base_tff_value': base_tff_value
                    })
                    if tff_inputs["fixed_pricer_params_for_tff_training"]:
                        registry_entry['tff_fixed_pricer_params'] = tff_inputs["fixed_pricer_params_for_tff_training"]
                    #print(f"      TFF for {instrument_id} fitted. RMSE: {rmse_tff:.6f}, Time: {fit_time:.2f}s")
                else: raise RuntimeError("TFF fitting returned None for model or norm_params.")
            except Exception as e:
                print(f"    ERROR during TFF calibration for {instrument_id} in worker: {e}")
                registry_entry.update({'pricing_method': 'FULL', 'error_tff_calibration': str(e)})

        return instrument_id, registry_entry

    def _process_batch(self, args_list: list[tuple]) -> dict:
        """
        Worker helper: given a list of (spec, scenarios, names, date_iso)
        calls _process_single_instrument_spec on each and returns a
        dict[instrument_id, registry_entry] for that batch.
        """
        batch_registry = {}
        for args in args_list:
            instrument_id, registry_entry = self._process_single_instrument_spec(args)
            if instrument_id:
                batch_registry[instrument_id] = registry_entry
        return batch_registry

    def process_instruments(self, instrument_definitions: list[dict],
                            global_market_scenarios: np.ndarray,
                            global_factor_names: list[str],
                            batch_size: int = 1) -> dict:
        print(f"Processing {len(instrument_definitions)} instrument definitions...")
        worker_args_list = [
            (spec, global_market_scenarios, global_factor_names, self.global_valuation_date.isoformat())
            for spec in instrument_definitions
        ]

        if self.num_instrument_processing_workers > 0 \
           and len(instrument_definitions) > 1 \
           and batch_size == 1:
            print(f"  Processing instruments in parallel (workers={self.num_instrument_processing_workers}) in batch of 1 …")
            with ProcessPoolExecutor(max_workers=self.num_instrument_processing_workers) as executor:
                futures = [executor.submit(self._process_single_instrument_spec, args) for args in worker_args_list]
                for future in tqdm(as_completed(futures), total=len(instrument_definitions), desc="Processing Instruments"):
                    try:
                        instrument_id, registry_entry = future.result()
                        if instrument_id: self.model_registry[instrument_id] = registry_entry
                    except Exception as e: print(f"    ERROR processing an instrument in parallel (future result): {e}")
        elif self.num_instrument_processing_workers > 0 \
             and len(instrument_definitions) > 1 \
             and batch_size > 1:
            print(f"  Processing instruments in parallel (workers={self.num_instrument_processing_workers}) in batches of {batch_size} …")
            with ProcessPoolExecutor(max_workers=self.num_instrument_processing_workers) as executor:
                futures = []
                for i in range(0, len(worker_args_list), batch_size):
                    batch_args = worker_args_list[i : i + batch_size]
                    futures.append(executor.submit(self._process_batch, batch_args))

                for future in tqdm(as_completed(futures),
                                   total=(len(instrument_definitions) + batch_size - 1) // batch_size,
                                   desc="Processing Instruments"):
                    try:
                        registry_batch = future.result()
                        self.model_registry.update(registry_batch)
                    except Exception as e:
                        print(f"    ERROR processing a batch of instruments in parallel: {e}")
        else:
            print("  Processing instruments sequentially...")
            for args_tuple in tqdm(worker_args_list, total=len(instrument_definitions), desc="Processing Instruments"):
                try:
                    instrument_id, registry_entry = self._process_single_instrument_spec(args_tuple)
                    if instrument_id: self.model_registry[instrument_id] = registry_entry
                except Exception as e: print(f"    CRITICAL ERROR processing instrument spec {args_tuple[0].get('instrument_id', 'Unknown')}: {e}")
        print("Finished processing instrument definitions.")
        return self.model_registry

    def save_model_registry(self, filepath: str):
        print(f"Saving model registry to {filepath}...")
        try:
            with open(filepath, 'w') as f: json.dump(self.model_registry, f, indent=4, default=portfolio_json_serializer)
            print("  Model registry saved successfully.")
        except Exception as e: print(f"  ERROR saving model registry: {e}")

    @classmethod
    def load_model_registry(cls, filepath: str) -> dict:
        print(f"Loading model registry from {filepath}...")
        try:
            with open(filepath, 'r') as f: registry = json.load(f)
            print("  Model registry loaded successfully."); return registry
        except Exception as e: print(f"  ERROR loading model registry: {e}"); return {}


class PortfolioBuilder:
    def __init__(self, model_registry: dict = None):
        self.model_registry = model_registry if model_registry is not None else {}
        self.uncalculated_instruments = []

    def build_portfolios_from_specs(self, portfolio_specs_list: list[dict],
                                       global_valuation_date: date,
                                       default_g2_params=None,
                                       default_bs_rfr=0.025, default_bs_div=0.0
                                       ) -> dict[str, Portfolio]:
        print(f"Building portfolios from {len(portfolio_specs_list)} detailed specifications...")
        portfolios = {}
        self.uncalculated_instruments = []

        for spec_idx, spec in enumerate(portfolio_specs_list):
            client_id = spec.get('client_id')
            instrument_id = spec.get('instrument_id')
            num_holdings = spec.get('num_holdings')
            product_static_dict_from_spec = spec.get('product_static_object')
            pricing_method_from_spec = spec.get('pricing_engine_type', 'full').lower()
            direct_tff_config_from_spec = spec.get('direct_tff_config')
            pricer_params_from_spec = spec.get('pricer_params', {})

            if not client_id or not instrument_id or num_holdings is None or product_static_dict_from_spec is None:
                print(f"  Skipping spec at index {spec_idx}: missing essential fields.")
                continue

            try:
                if 'valuation_date' not in product_static_dict_from_spec:
                    product_static_dict_from_spec['valuation_date'] = global_valuation_date
                product_static_object = reconstruct_product_static(product_static_dict_from_spec)
            except Exception as e:
                print(f"  ERROR reconstructing product static for '{instrument_id}' from spec: {e}. Skipping.")
                if instrument_id not in self.uncalculated_instruments: self.uncalculated_instruments.append(instrument_id)
                continue

            if client_id not in portfolios:
                portfolios[client_id] = Portfolio()
            portfolio_instance = portfolios[client_id]

            final_pricing_method = pricing_method_from_spec
            final_direct_tff_config = direct_tff_config_from_spec
            final_full_pricer_instance = None
            final_pricer_kwargs = {}
            if pricer_params_from_spec: final_pricer_kwargs.update(pricer_params_from_spec)

            if final_pricing_method == 'tff':
                if not final_direct_tff_config:
                    if instrument_id in self.model_registry and self.model_registry[instrument_id].get('pricing_method', '').upper() == 'TFF':
                        entry = self.model_registry[instrument_id]
                        if all(k in entry for k in ['tff_model_dict', 'tff_raw_input_names', 'tff_normalization_params']):
                            final_direct_tff_config = {
                                'model_dict': entry['tff_model_dict'],
                                'raw_input_names': entry['tff_raw_input_names'],
                                'normalization_params': entry['tff_normalization_params'],
                                'option_feature_order': entry.get('tff_option_feature_order', 0)
                            }
                            if 'tff_fixed_pricer_params' in entry:
                                final_pricer_kwargs.update(entry['tff_fixed_pricer_params'])
                        else:
                            print(f"  WARNING: TFF data incomplete for '{instrument_id}' in registry. Fallback to FULL.")
                            final_pricing_method = 'full'
                    else:
                        print(f"  WARNING: TFF spec for '{instrument_id}' missing direct_tff_config and not found as TFF in registry. Fallback to FULL.")
                        final_pricing_method = 'full'
                elif isinstance(product_static_object, ConvertibleBondStaticBase) and final_direct_tff_config:
                     final_pricer_kwargs.update(pricer_params_from_spec)


            if final_pricing_method == 'full':
                current_pricer_params = final_pricer_kwargs
                try:
                    if isinstance(product_static_object, EuropeanOptionStatic):
                        # instantiate pricer with only the static def
                        final_full_pricer_instance = BlackScholesPricer(product_static_object)
                        # push rates into price() kwargs
                        rfr = current_pricer_params.get('bs_risk_free_rate', default_bs_rfr)
                        div = current_pricer_params.get('bs_dividend_yield', default_bs_div)
                        final_pricer_kwargs = {
                            'risk_free_rate': rfr,
                            'dividend_yield': div
                        }
                    elif isinstance(product_static_object, MBSPoolStatic):
                        prepayment_model_type = product_static_object.prepayment_model_type
                        prepayment_rate_param = product_static_object.prepayment_rate_param
                        prepayment_model_instance = None
                        if prepayment_model_type == "CPR":
                            prepayment_model_instance = ConstantCPRModel(prepayment_rate_param)
                        elif prepayment_model_type == "PSA":
                            prepayment_model_instance = PSAModel(prepayment_rate_param)
                        elif prepayment_model_type == "RefiIncentive":
                            refi_A = current_pricer_params.get('refi_A')
                            refi_B = current_pricer_params.get('refi_B')
                            refi_C = current_pricer_params.get('refi_C')
                            refi_D = current_pricer_params.get('refi_D')
                            if all(p is not None for p in [refi_A, refi_B, refi_C, refi_D]):
                                prepayment_model_instance = RefiIncentivePrepaymentModel(refi_A, refi_B, refi_C, refi_D)
                            else:
                                prepayment_model_instance = RefiIncentivePrepaymentModel()
                        else: raise ValueError(f"Unsupported prepayment_model_type: {prepayment_model_type} for MBS.")
                        final_full_pricer_instance = MBSPricer(product_static_object, prepayment_model=prepayment_model_instance)
                    elif isinstance(product_static_object, CallableBondStaticBase):
                        grid_steps = current_pricer_params.get('g2_grid_steps', 32)
                        final_full_pricer_instance = QuantLibBondPricer(product_static_object, method='g2', grid_steps=grid_steps)
                        if current_pricer_params.get('g2_params', default_g2_params):
                             final_pricer_kwargs['g2_params'] = current_pricer_params.get('g2_params', default_g2_params)
                    elif isinstance(product_static_object, ConvertibleBondStaticBase):
                        cb_full_kwargs_needed = {
                            's0_val': current_pricer_params.get('s0_val', current_pricer_params.get('initial_stock_price')),
                            'dividend_yield': current_pricer_params.get('dividend_yield'),
                            'equity_volatility': current_pricer_params.get('equity_volatility'),
                            'credit_spread': current_pricer_params.get('credit_spread')
                        }
                        if any(val is None for val in cb_full_kwargs_needed.values()):
                            raise ValueError(f"Missing required pricer_params for FULL CB pricing of {instrument_id}.")
                        final_pricer_kwargs.update(cb_full_kwargs_needed)
                        final_full_pricer_instance = QuantLibBondPricer(
                            product_static_object, method='convertible_binomial',
                            convertible_engine_steps=current_pricer_params.get('conv_engine_steps', 128)
                        )
                    elif isinstance(product_static_object, QuantLibBondStaticBase):
                        final_full_pricer_instance = QuantLibBondPricer(product_static_object, method='discount')
                    else: raise ValueError("Unknown product type for full pricer reconstruction.")
                except Exception as e_pricer:
                    print(f"  WARNING: Cannot create full pricer for '{instrument_id}': {e_pricer}. Skipping.")
                    if instrument_id not in self.uncalculated_instruments: self.uncalculated_instruments.append(instrument_id)
                    continue

            try:
                portfolio_instance.add_position(
                    instrument_id=instrument_id, product_static=product_static_object,
                    num_holdings=num_holdings, pricing_engine_type=final_pricing_method,
                    direct_tff_config=final_direct_tff_config if final_pricing_method == 'tff' else None,
                    full_pricer_instance=final_full_pricer_instance if final_pricing_method == 'full' else None,
                    full_pricer_kwargs=final_pricer_kwargs
                )
            except Exception as e:
                print(f"  ERROR adding position '{instrument_id}' to portfolio for '{client_id}': {e}")
                if instrument_id not in self.uncalculated_instruments: self.uncalculated_instruments.append(instrument_id)

        if self.uncalculated_instruments:
            print(f"  Summary: Uncalculated instruments during build_portfolios_from_specs: {self.uncalculated_instruments}")
        print(f"Finished building {len(portfolios)} portfolios from detailed specs.")
        return portfolios


class PortfolioAnalytics:
    def __init__(self,
                 client_portfolios: dict[str, Portfolio],
                 global_market_scenarios: np.ndarray,
                 global_factor_names: list[str],
                 numeric_rate_tenors: np.ndarray,
                 scenario_generator_for_base_values: SimpleRandomScenarioGenerator
                 ):
        self.client_portfolios = client_portfolios
        self.global_market_scenarios = global_market_scenarios
        self.global_factor_names = global_factor_names
        self.numeric_rate_tenors = numeric_rate_tenors
        self.scenario_generator_for_base_values = scenario_generator_for_base_values
        self.results = {}

    def calculate_base_portfolio_values(self) -> dict[str, float]:
        base_values = {}
        base_value_scenario_list = []
        sg_for_base = self.scenario_generator_for_base_values
        for factor_name in self.global_factor_names:
            val_found = False
            for current_map_name in ['base_rates_map', 'base_s0_map', 'base_vol_map', 'base_credit_spread_points_map']:
                current_map = getattr(sg_for_base, current_map_name, {})
                if factor_name in current_map:
                    base_value_scenario_list.append(current_map[factor_name])
                    val_found = True
                    break
            if not val_found:
                if hasattr(sg_for_base, 'base_s0_map') and factor_name in sg_for_base.base_s0_map:
                     base_value_scenario_list.append(sg_for_base.base_s0_map[factor_name]); val_found = True
                elif hasattr(sg_for_base, 'base_vol_map') and factor_name in sg_for_base.base_vol_map:
                     base_value_scenario_list.append(sg_for_base.base_vol_map[factor_name]); val_found = True
            if not val_found:
                print(f"Warning: Factor '{factor_name}' for base value not found in generator base maps. Using 0.0.")
                base_value_scenario_list.append(0.0)
        base_value_scenario_np = np.array([base_value_scenario_list])

        for client_id, portfolio_obj in self.client_portfolios.items():
            if portfolio_obj.positions:
                try:
                    base_val = portfolio_obj.price_portfolio(
                        raw_market_scenarios=base_value_scenario_np,
                        scenario_factor_names=self.global_factor_names,
                        portfolio_rate_pillar_times=self.numeric_rate_tenors
                    )[0]
                    base_values[client_id] = base_val
                except Exception as e:
                    print(f"  ERROR calculating base value for portfolio {client_id}: {e}")
                    base_values[client_id] = np.nan
            else:
                base_values[client_id] = 0.0
        return base_values

    def run_var_analysis(self, var_percentiles: list[float] = None):
        if var_percentiles is None:
            var_percentiles = [1.0, 5.0]

        print(f"Running VaR Analysis for percentiles: {[f'{(100-p):.0f}%' for p in var_percentiles]}")

        base_portfolio_values = self.calculate_base_portfolio_values()
        self.results = {}

        for client_id, portfolio_obj in self.client_portfolios.items():
            client_results = {'base_value': base_portfolio_values.get(client_id, np.nan)}
            if portfolio_obj.positions and not np.isnan(client_results['base_value']):
                print(f"  Analyzing portfolio for {client_id}...")
                try:
                    portfolio_values_scenarios = portfolio_obj.price_portfolio(
                        raw_market_scenarios=self.global_market_scenarios,
                        scenario_factor_names=self.global_factor_names,
                        portfolio_rate_pillar_times=self.numeric_rate_tenors
                    )

                    client_results['mean_scenario_value'] = np.mean(portfolio_values_scenarios)
                    client_results['std_dev_scenario_value'] = np.std(portfolio_values_scenarios)
                    pnl_distribution = portfolio_values_scenarios - client_results['base_value']
                    client_results['pnl_distribution_mean'] = np.mean(pnl_distribution)
                    client_results['pnl_distribution_std_dev'] = np.std(pnl_distribution)
                    client_results['pnl_distribution'] = pnl_distribution

                    vars_calculated = {}
                    for p in var_percentiles:
                        var_value = np.percentile(pnl_distribution, p)
                        vars_calculated[f"var_{(100-p):.0f}pct"] = -var_value
                    client_results['var_values'] = vars_calculated

                    print(f"    Client {client_id}: Base Value={client_results['base_value']:.2f}, "
                          f"Mean Scen. Value={client_results['mean_scenario_value']:.2f}, "
                          f"VaRs: {vars_calculated}")

                except Exception as e:
                    print(f"    ERROR during VaR analysis for portfolio {client_id}: {e}")
                    client_results['error_var_analysis'] = str(e)
            else:
                if not portfolio_obj.positions: print(f"  Portfolio for {client_id} is empty, skipping VaR.")
                else: print(f"  Base value for portfolio {client_id} could not be calculated (was NaN), skipping VaR.")
                client_results['var_values'] = {f"var_{(100-p):.0f}pct": np.nan for p in var_percentiles}
            self.results[client_id] = client_results
        return self.results

    def calculate_base_value(self, portfolio):
        """
        Compute the portfolio’s base‐value by pricing it on the generator’s
        “base” scenario (all factors = their base map values).
        """
        try:
            # get a single‐row array of the base factors
            base_scenario, _ = self.scenario_generator_for_base_values.generate_scenarios(
                n_scenarios=1,
                factor_names=self.global_factor_names
            )
            # price_portfolio expects (N, factors), names, and optionally rate pillars
            base_price = portfolio.price_portfolio(
                raw_market_scenarios=base_scenario,
                scenario_factor_names=self.global_factor_names,
                portfolio_rate_pillar_times=self.numeric_rate_tenors
            )
            # price_portfolio returns array with one element
            return float(base_price[0])
        except Exception as e:
            print(f"ERROR calculating base value for portfolio {portfolio.client_id}: {e}")
            return float('nan')

# Helper function for parallel execution in generate_price_strips
def _generate_single_strip_worker(
    spec: dict,
    global_market_scenarios: np.ndarray,
    global_factor_names: list[str],
    iproc: InstrumentProcessor # Pass the InstrumentProcessor instance
) -> tuple[str, np.ndarray | None, str | None]:
    """
    Worker function to generate a price strip for a single instrument.
    Returns (instrument_id, prices_array, error_message).
    """
    inst_id = spec['instrument_id']
    try:
        # Prepare the dictionary for reconstruct_product_static
        params_for_reconstruction = spec.get('params', {}).copy()
        if 'product_type' not in params_for_reconstruction:
            if 'product_type' in spec:
                params_for_reconstruction['product_type'] = spec['product_type']
            else:
                raise ValueError(f"Instrument spec for '{inst_id}' must contain a 'product_type' field.")

        product_static = reconstruct_product_static(params_for_reconstruction)
        
        # Use iproc's method to create the pricer template
        pricer = iproc._create_pricer_template(product_static, spec)
        
        price_scenarios_kwargs = spec.get('pricer_params', {}).copy()

        # Ensure default BS RFR and Div Yield are passed if not in spec,
        # similar to how InstrumentProcessor sets up _default_price_kwargs
        if isinstance(pricer, BlackScholesPricer):
            if 'risk_free_rate' not in price_scenarios_kwargs:
                price_scenarios_kwargs['risk_free_rate'] = iproc.default_bs_risk_free_rate
            if 'dividend_yield' not in price_scenarios_kwargs:
                price_scenarios_kwargs['dividend_yield'] = iproc.default_bs_dividend_yield
        
        # For G2 bonds, ensure g2_params are passed if method is g2
        if isinstance(pricer, QuantLibBondPricer) and pricer.method == 'g2':
            if 'g2_params' not in price_scenarios_kwargs and iproc.default_g2_params:
                price_scenarios_kwargs['g2_params'] = iproc.default_g2_params

        prices = pricer.price_scenarios(
            raw_market_scenarios    = global_market_scenarios,
            scenario_factor_names   = global_factor_names,
            rate_pillars            = iproc.default_numeric_rate_tenors,
            **price_scenarios_kwargs
        )
        return inst_id, np.atleast_1d(prices).flatten(), None
    except Exception as e:
        # Error handling: return instrument_id, None for prices, and the error message
        return inst_id, None, str(e)
    
    
def generate_price_strips(
    instrument_specs: list[dict],
    global_market_scenarios: np.ndarray,
    global_factor_names: list[str],
    iproc: InstrumentProcessor,
    num_workers: int = None,
    batch_size: int = 1  # NEW: Add batch_size parameter
) -> Dict[str, np.ndarray]:
    """
    Generate price strips for each instrument across all scenarios using full pricers.
    Can run in parallel if num_workers is provided and > 1.
    Now supports batch processing for better memory management and progress tracking.

    Args:
        instrument_specs (list[dict]): Each spec must have:
            - "instrument_id": str
            - "product_type": str (e.g., "VanillaBond", "EuropeanOptionStatic")
            - "params": dict for reconstruct_product_static (constructor args for the static product)
            - optional "pricer_params": dict for pricer kwargs
        global_market_scenarios (np.ndarray): shape (N_scenarios, N_factors)
        global_factor_names (list[str]): names of the columns in scenarios
        iproc (InstrumentProcessor): InstrumentProcessor instance.
        num_workers (int, optional): Number of parallel workers. 
                                     Defaults to None (sequential processing).
                                     If 0 or 1, runs sequentially.
                                     If > 1, uses ProcessPoolExecutor.
        batch_size (int, optional): Number of instruments to process per batch.
                                    Defaults to 1 (process one at a time).
                                    Only applies to parallel processing.

    Returns:
        Dict mapping instrument_id to 1D np.ndarray of prices (length N_scenarios).
    """
    strips: Dict[str, np.ndarray] = {}
    
    effective_num_workers = num_workers if num_workers is not None else 0

    if effective_num_workers > 1 and len(instrument_specs) > 1:
        if batch_size > 1:
            print(f"Generating price strips in parallel with {effective_num_workers} workers in batches of {batch_size}...")
            # Process instruments in batches
            with ProcessPoolExecutor(max_workers=effective_num_workers) as executor:
                futures = []
                
                # Create batches of instrument specs
                for i in range(0, len(instrument_specs), batch_size):
                    batch_specs = instrument_specs[i:i + batch_size]
                    futures.append(
                        executor.submit(
                            _generate_batch_strips_worker,
                            batch_specs,
                            global_market_scenarios,
                            global_factor_names,
                            iproc
                        )
                    )
                
                # Process completed batches
                for future in tqdm(as_completed(futures), 
                                 total=len(futures), 
                                 desc=f"Processing {len(futures)} batches"):
                    batch_strips, batch_errors = future.result()
                    strips.update(batch_strips)
                    
                    # Report any errors from the batch
                    for inst_id, error_msg in batch_errors.items():
                        print(f"  ERROR generating price strip for {inst_id}: {error_msg}")
        else:
            print(f"Generating price strips in parallel with {effective_num_workers} workers (individual processing)...")
            # Original single-instrument parallel processing
            with ProcessPoolExecutor(max_workers=effective_num_workers) as executor:
                futures = [
                    executor.submit(
                        _generate_single_strip_worker,
                        spec,
                        global_market_scenarios,
                        global_factor_names,
                        iproc
                    )
                    for spec in instrument_specs
                ]
                for future in tqdm(as_completed(futures), total=len(instrument_specs), desc="Generating Price Strips"):
                    inst_id, prices_array, error_msg = future.result()
                    if error_msg:
                        print(f"  ERROR generating price strip for {inst_id}: {error_msg}")
                    elif prices_array is not None:
                        strips[inst_id] = prices_array
    else:
        if len(instrument_specs) > 1:
            print("Generating price strips sequentially...")
        for spec in tqdm(instrument_specs, desc="Generating Price Strips", disable=len(instrument_specs) <= 1):
            inst_id = spec['instrument_id']
            try:
                params_for_reconstruction = spec.get('params', {}).copy()
                if 'product_type' not in params_for_reconstruction:
                    if 'product_type' in spec:
                        params_for_reconstruction['product_type'] = spec['product_type']
                    else:
                        raise ValueError(f"Instrument spec for '{inst_id}' must contain a 'product_type' field.")

                product_static = reconstruct_product_static(params_for_reconstruction)
                pricer = iproc._create_pricer_template(product_static, spec)
                price_scenarios_kwargs = spec.get('pricer_params', {}).copy()

                if isinstance(pricer, BlackScholesPricer):
                    if 'risk_free_rate' not in price_scenarios_kwargs:
                        price_scenarios_kwargs['risk_free_rate'] = iproc.default_bs_risk_free_rate
                    if 'dividend_yield' not in price_scenarios_kwargs:
                        price_scenarios_kwargs['dividend_yield'] = iproc.default_bs_dividend_yield
                
                if isinstance(pricer, QuantLibBondPricer) and pricer.method == 'g2':
                    if 'g2_params' not in price_scenarios_kwargs and iproc.default_g2_params:
                        price_scenarios_kwargs['g2_params'] = iproc.default_g2_params

                prices = pricer.price_scenarios(
                    raw_market_scenarios    = global_market_scenarios,
                    scenario_factor_names   = global_factor_names,
                    rate_pillars            = iproc.default_numeric_rate_tenors,
                    **price_scenarios_kwargs
                )
                strips[inst_id] = np.atleast_1d(prices).flatten()
            except Exception as e:
                print(f"  ERROR generating price strip for {inst_id} (sequential): {e}")
    
    return strips

def _generate_batch_strips_worker(
    batch_specs: list[dict],
    global_market_scenarios: np.ndarray,
    global_factor_names: list[str],
    iproc: InstrumentProcessor
) -> tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Worker function to generate price strips for a batch of instruments.
    Returns (batch_strips_dict, batch_errors_dict).
    """
    batch_strips = {}
    batch_errors = {}
    
    for spec in batch_specs:
        inst_id, prices_array, error_msg = _generate_single_strip_worker(
            spec, global_market_scenarios, global_factor_names, iproc
        )
        
        if error_msg:
            batch_errors[inst_id] = error_msg
        elif prices_array is not None:
            batch_strips[inst_id] = prices_array
    
    return batch_strips, batch_errors

# portfolio.py
"""
Contains classes for defining and analyzing portfolios of financial instruments.
The Portfolio class allows pricing using either TFF models (retrieved from a cache)
or full pricers.
"""
import numpy as np
import abc
# Import custom classes for product definitions and pricers
from features_generator import engineer_option_features, normalize_features
from mbs_pricer import MBSPricer
from product_definitions import ProductStaticBase, EuropeanOptionStatic
# Ensure that the custom classes are imported correctly
from quantlib_custom_serializer import custom_json_serializer
from datetime import date
# Ensure that ProductStaticBase is imported if we are checking for it in to_dict
from product_definitions import ProductStaticBase
# Ensure that PricerBase and its derivatives are imported for pricer handling
from base_pricer import PricerBase
from quantlib_bond_pricer import QuantLibBondPricer
from fast_bond_pricer import FastBondPricer
from black_scholes_pricer import BlackScholesPricer
from tff_approximator import TensorFunctionalForm

from concurrent.futures import ThreadPoolExecutor, as_completed
import os


class PortfolioBuilder:
    """
    A class to build portfolios from specifications, where each specification
    defines a position in the portfolio. This class is designed to handle
    specifications that include both TFF (Tensor Functional Form) and full
    pricing methods for financial instruments.
    
    It allows for the construction of portfolios based on a list of dictionaries,
    each specifying an instrument to include in the portfolio.
    
    The specifications should include the instrument ID, product static object,
    number of holdings, and the type of pricing engine to use (either 'tff' or 'full').
    The class also supports optional configurations for TFF and full pricers.
    """
    def __init__(self, model_registry: dict = None):
        self.model_registry = model_registry if model_registry is not None else {}
        self.uncalculated_instruments = []
       
    def build_portfolios_from_specs(self, portfolio_specs_list: list[dict],
                                       global_valuation_date: date,
                                       default_g2_params=None,
                                       default_bs_rfr=0.025, default_bs_div=0.0
                                       ) -> dict[str, Portfolio]:
        print(f"Building portfolios from {len(portfolio_specs_list)} detailed specifications...")
        portfolios = {}
        self.uncalculated_instruments = []

        for spec_idx, spec in enumerate(portfolio_specs_list):
            client_id = spec.get('client_id')
            instrument_id = spec.get('instrument_id')
            num_holdings = spec.get('num_holdings')
            product_static_dict_from_spec = spec.get('product_static_object')
            pricing_method_from_spec = spec.get('pricing_engine_type', 'full').lower()
            direct_tff_config_from_spec = spec.get('direct_tff_config')
            pricer_params_from_spec = spec.get('pricer_params', {})

            if not client_id or not instrument_id or num_holdings is None or product_static_dict_from_spec is None:
                print(f"  Skipping spec at index {spec_idx}: missing essential fields.")
                continue

            try:
                if 'valuation_date' not in product_static_dict_from_spec:
                    product_static_dict_from_spec['valuation_date'] = global_valuation_date
                product_static_object = reconstruct_product_static(product_static_dict_from_spec)
            except Exception as e:
                print(f"  ERROR reconstructing product static for '{instrument_id}' from spec: {e}. Skipping.")
                if instrument_id not in self.uncalculated_instruments: self.uncalculated_instruments.append(instrument_id)
                continue

            if client_id not in portfolios:
                portfolios[client_id] = Portfolio()
            portfolio_instance = portfolios[client_id]

            final_pricing_method = pricing_method_from_spec
            final_direct_tff_config = direct_tff_config_from_spec
            final_full_pricer_instance = None
            final_pricer_kwargs = {}
            if pricer_params_from_spec: final_pricer_kwargs.update(pricer_params_from_spec)

            if final_pricing_method == 'tff':
                if not final_direct_tff_config:
                    if instrument_id in self.model_registry and self.model_registry[instrument_id].get('pricing_method', '').upper() == 'TFF':
                        entry = self.model_registry[instrument_id]
                        if all(k in entry for k in ['tff_model_dict', 'tff_raw_input_names', 'tff_normalization_params']):
                            final_direct_tff_config = {
                                'model_dict': entry['tff_model_dict'],
                                'raw_input_names': entry['tff_raw_input_names'],
                                'normalization_params': entry['tff_normalization_params'],
                                'option_feature_order': entry.get('tff_option_feature_order', 0)
                            }
                            if 'tff_fixed_pricer_params' in entry:
                                final_pricer_kwargs.update(entry['tff_fixed_pricer_params'])
                        else:
                            print(f"  WARNING: TFF data incomplete for '{instrument_id}' in registry. Fallback to FULL.")
                            final_pricing_method = 'full'
                    else:
                        print(f"  WARNING: TFF spec for '{instrument_id}' missing direct_tff_config and not found as TFF in registry. Fallback to FULL.")
                        final_pricing_method = 'full'
                elif isinstance(product_static_object, ConvertibleBondStaticBase) and final_direct_tff_config:
                     final_pricer_kwargs.update(pricer_params_from_spec)


            if final_pricing_method == 'full':
                current_pricer_params = final_pricer_kwargs
                try:
                    if isinstance(product_static_object, EuropeanOptionStatic):
                        # instantiate pricer with only the static def
                        final_full_pricer_instance = BlackScholesPricer(product_static_object)
                        # push rates into price() kwargs
                        rfr = current_pricer_params.get('bs_risk_free_rate', default_bs_rfr)
                        div = current_pricer_params.get('bs_dividend_yield', default_bs_div)
                        final_pricer_kwargs = {
                            'risk_free_rate': rfr,
                            'dividend_yield': div
                        }
                    elif isinstance(product_static_object, MBSPoolStatic):
                        prepayment_model_type = product_static_object.prepayment_model_type
                        prepayment_rate_param = product_static_object.prepayment_rate_param
                        prepayment_model_instance = None
                        if prepayment_model_type == "CPR":
                            prepayment_model_instance = ConstantCPRModel(prepayment_rate_param)
                        elif prepayment_model_type == "PSA":
                            prepayment_model_instance = PSAModel(prepayment_rate_param)
                        elif prepayment_model_type == "RefiIncentive":
                            refi_A = current_pricer_params.get('refi_A')
                            refi_B = current_pricer_params.get('refi_B')
                            refi_C = current_pricer_params.get('refi_C')
                            refi_D = current_pricer_params.get('refi_D')
                            if all(p is not None for p in [refi_A, refi_B, refi_C, refi_D]):
                                prepayment_model_instance = RefiIncentivePrepaymentModel(refi_A, refi_B, refi_C, refi_D)
                            else:
                                prepayment_model_instance = RefiIncentivePrepaymentModel()
                        else: raise ValueError(f"Unsupported prepayment_model_type: {prepayment_model_type} for MBS.")
                        final_full_pricer_instance = MBSPricer(product_static_object, prepayment_model=prepayment_model_instance)
                    elif isinstance(product_static_object, CallableBondStaticBase):
                        grid_steps = current_pricer_params.get('g2_grid_steps', 32)
                        final_full_pricer_instance = QuantLibBondPricer(product_static_object, method='g2', grid_steps=grid_steps)
                        if current_pricer_params.get('g2_params', default_g2_params):
                             final_pricer_kwargs['g2_params'] = current_pricer_params.get('g2_params', default_g2_params)
                    elif isinstance(product_static_object, ConvertibleBondStaticBase):
                        cb_full_kwargs_needed = {
                            's0_val': current_pricer_params.get('s0_val', current_pricer_params.get('initial_stock_price')),
                            'dividend_yield': current_pricer_params.get('dividend_yield'),
                            'equity_volatility': current_pricer_params.get('equity_volatility'),
                            'credit_spread': current_pricer_params.get('credit_spread')
                        }
                        if any(val is None for val in cb_full_kwargs_needed.values()):
                            raise ValueError(f"Missing required pricer_params for FULL CB pricing of {instrument_id}.")
                        final_pricer_kwargs.update(cb_full_kwargs_needed)
                        final_full_pricer_instance = QuantLibBondPricer(
                            product_static_object, method='convertible_binomial',
                            convertible_engine_steps=current_pricer_params.get('conv_engine_steps', None)
                        )
                    elif isinstance(product_static_object, QuantLibBondStaticBase):
                        final_full_pricer_instance = QuantLibBondPricer(product_static_object, method='discount')
                    else: raise ValueError("Unknown product type for full pricer reconstruction.")
                except Exception as e_pricer:
                    print(f"  WARNING: Cannot create full pricer for '{instrument_id}': {e_pricer}. Skipping.")
                    if instrument_id not in self.uncalculated_instruments: self.uncalculated_instruments.append(instrument_id)
                    continue

            try:
                portfolio_instance.add_position(
                    instrument_id=instrument_id, product_static=product_static_object,
                    num_holdings=num_holdings, pricing_engine_type=final_pricing_method,
                    direct_tff_config=final_direct_tff_config if final_pricing_method == 'tff' else None,
                    full_pricer_instance=final_full_pricer_instance if final_pricing_method == 'full' else None,
                    full_pricer_kwargs=final_pricer_kwargs
                )
            except Exception as e:
                print(f"  ERROR adding position '{instrument_id}' to portfolio for '{client_id}': {e}")
                if instrument_id not in self.uncalculated_instruments: self.uncalculated_instruments.append(instrument_id)

        if self.uncalculated_instruments:
            print(f"  Summary: Uncalculated instruments during build_portfolios_from_specs: {self.uncalculated_instruments}")
        print(f"Finished building {len(portfolios)} portfolios from detailed specs.")
        return portfolios


class PortfolioAnalytics:
    def __init__(self,
                 client_portfolios: dict[str, Portfolio],
                 global_market_scenarios: np.ndarray,
                 global_factor_names: list[str],
                 numeric_rate_tenors: np.ndarray,
                 scenario_generator_for_base_values: SimpleRandomScenarioGenerator
                 ):
        self.client_portfolios = client_portfolios
        self.global_market_scenarios = global_market_scenarios
        self.global_factor_names = global_factor_names
        self.numeric_rate_tenors = numeric_rate_tenors
        self.scenario_generator_for_base_values = scenario_generator_for_base_values
        self.results = {}

    def calculate_base_portfolio_values(self) -> dict[str, float]:
        base_values = {}
        base_value_scenario_list = []
        sg_for_base = self.scenario_generator_for_base_values
        for factor_name in self.global_factor_names:
            val_found = False
            for current_map_name in ['base_rates_map', 'base_s0_map', 'base_vol_map', 'base_credit_spread_points_map']:
                current_map = getattr(sg_for_base, current_map_name, {})
                if factor_name in current_map:
                    base_value_scenario_list.append(current_map[factor_name])
                    val_found = True
                    break
            if not val_found:
                if hasattr(sg_for_base, 'base_s0_map') and factor_name in sg_for_base.base_s0_map:
                     base_value_scenario_list.append(sg_for_base.base_s0_map[factor_name]); val_found = True
                elif hasattr(sg_for_base, 'base_vol_map') and factor_name in sg_for_base.base_vol_map:
                     base_value_scenario_list.append(sg_for_base.base_vol_map[factor_name]); val_found = True
            if not val_found:
                print(f"Warning: Factor '{factor_name}' for base value not found in generator base maps. Using 0.0.")
                base_value_scenario_list.append(0.0)
        base_value_scenario_np = np.array([base_value_scenario_list])

        for client_id, portfolio_obj in self.client_portfolios.items():
            if portfolio_obj.positions:
                try:
                    base_val = portfolio_obj.price_portfolio(
                        raw_market_scenarios=base_value_scenario_np,
                        scenario_factor_names=self.global_factor_names,
                        portfolio_rate_pillar_times=self.numeric_rate_tenors
                    )[0]
                    base_values[client_id] = base_val
                except Exception as e:
                    print(f"  ERROR calculating base value for portfolio {client_id}: {e}")
                    base_values[client_id] = np.nan
            else:
                base_values[client_id] = 0.0
        return base_values

    def run_var_analysis(self, var_percentiles: list[float] = None):
        if var_percentiles is None:
            var_percentiles = [1.0, 5.0]

        print(f"Running VaR Analysis for percentiles: {[f'{(100-p):.0f}%' for p in var_percentiles]}")

        base_portfolio_values = self.calculate_base_portfolio_values()
        self.results = {}

        for client_id, portfolio_obj in self.client_portfolios.items():
            client_results = {'base_value': base_portfolio_values.get(client_id, np.nan)}
            if portfolio_obj.positions and not np.isnan(client_results['base_value']):
                print(f"  Analyzing portfolio for {client_id}...")
                try:
                    portfolio_values_scenarios = portfolio_obj.price_portfolio(
                        raw_market_scenarios=self.global_market_scenarios,
                        scenario_factor_names=self.global_factor_names,
                        portfolio_rate_pillar_times=self.numeric_rate_tenors
                    )

                    client_results['mean_scenario_value'] = np.mean(portfolio_values_scenarios)
                    client_results['std_dev_scenario_value'] = np.std(portfolio_values_scenarios)
                    pnl_distribution = portfolio_values_scenarios - client_results['base_value']
                    client_results['pnl_distribution_mean'] = np.mean(pnl_distribution)
                    client_results['pnl_distribution_std_dev'] = np.std(pnl_distribution)
                    client_results['pnl_distribution'] = pnl_distribution

                    vars_calculated = {}
                    for p in var_percentiles:
                        var_value = np.percentile(pnl_distribution, p)
                        vars_calculated[f"var_{(100-p):.0f}pct"] = -var_value
                    client_results['var_values'] = vars_calculated

                    print(f"    Client {client_id}: Base Value={client_results['base_value']:.2f}, "
                          f"Mean Scen. Value={client_results['mean_scenario_value']:.2f}, "
                          f"VaRs: {vars_calculated}")

                except Exception as e:
                    print(f"    ERROR during VaR analysis for portfolio {client_id}: {e}")
                    client_results['error_var_analysis'] = str(e)
            else:
                if not portfolio_obj.positions: print(f"  Portfolio for {client_id} is empty, skipping VaR.")
                else: print(f"  Base value for portfolio {client_id} could not be calculated (was NaN), skipping VaR.")
                client_results['var_values'] = {f"var_{(100-p):.0f}pct": np.nan for p in var_percentiles}
            self.results[client_id] = client_results
        return self.results

    def calculate_base_value(self, portfolio):
        """
        Compute the portfolio’s base‐value by pricing it on the generator’s
        “base” scenario (all factors = their base map values).
        """
        try:
            # get a single‐row array of the base factors
            base_scenario, _ = self.scenario_generator_for_base_values.generate_scenarios(
                n_scenarios=1,
                factor_names=self.global_factor_names
            )
            # price_portfolio expects (N, factors), names, and optionally rate pillars
            base_price = portfolio.price_portfolio(
                raw_market_scenarios=base_scenario,
                scenario_factor_names=self.global_factor_names,
                portfolio_rate_pillar_times=self.numeric_rate_tenors
            )
            # price_portfolio returns array with one element
            return float(base_price[0])
        except Exception as e:
            print(f"ERROR calculating base value for portfolio {portfolio.client_id}: {e}")
            return float('nan')
